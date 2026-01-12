import lldb
import re


def _sbvalue_to_u64(v: lldb.SBValue) -> int:
    """Best-effort parse of pointer/int SBValue into an unsigned int."""
    if not v or not v.IsValid():
        return 0
    try:
        x = v.GetValueAsUnsigned()
        if x:
            return int(x)
    except Exception:
        pass

    s = v.GetValue() or ""
    # Common form: "0x1234..."
    m = re.search(r"0x[0-9a-fA-F]+", s)
    if m:
        return int(m.group(0), 16)

    # Sometimes LLDB prints pointers oddly; try summary too
    s2 = v.GetSummary() or ""
    m2 = re.search(r"0x[0-9a-fA-F]+", s2)
    if m2:
        return int(m2.group(0), 16)

    return 0


def _get_child_any(obj: lldb.SBValue, *names):
    """Return first valid child member with one of these names."""
    for n in names:
        c = obj.GetChildMemberWithName(n)
        if c and c.IsValid():
            return c
    return None


def _extract_vec_u32_limbs(vec: lldb.SBValue):
    """
    Try to get Vec<u32> elements via synthetic children first.
    If none are found, return [].
    """
    limbs = []
    if not vec or not vec.IsValid():
        return limbs

    for i in range(vec.GetNumChildren()):
        ch = vec.GetChildAtIndex(i)
        if not ch or not ch.IsValid():
            continue
        nm = ch.GetName() or ""

        # Accept common synthetic element naming patterns:
        # [0], [1] ... OR 0, 1 ... OR __0, __1 ...
        if (
            (nm.startswith("[") and nm.endswith("]"))
            or nm.isdigit()
            or nm.startswith("__")
        ):
            try:
                limbs.append(int(ch.GetValueAsUnsigned()))
            except Exception:
                pass

    return limbs


def _read_vec_u32_from_memory(vec: lldb.SBValue, any_valobj: lldb.SBValue):
    """
    Fallback: read Vec<u32> from memory using len + buf/ptr.
    Returns limbs in *memory order* (which for BigUint digits is typically little-endian / least-significant first).
    """
    limbs = []
    if not vec or not vec.IsValid():
        return limbs

    # len might be directly "len" or wrapped (len.__0)
    len_obj = vec.GetChildMemberWithName("len")
    if not len_obj or not len_obj.IsValid():
        return limbs

    try:
        length = int(len_obj.GetValueAsUnsigned())
    except Exception:
        length = 0
    if length <= 0:
        return limbs

    # Navigate to the pointer:
    # vec.buf.inner.ptr.pointer   (and sometimes another .pointer wrapper)
    buf = vec.GetChildMemberWithName("buf")
    if not buf or not buf.IsValid():
        return limbs
    inner = buf.GetChildMemberWithName("inner")
    if not inner or not inner.IsValid():
        return limbs
    ptr = inner.GetChildMemberWithName("ptr")
    if not ptr or not ptr.IsValid():
        return limbs
    p = ptr.GetChildMemberWithName("pointer")
    if not p or not p.IsValid():
        return limbs

    # Some builds show pointer wrapped again as .pointer
    p2 = p.GetChildMemberWithName("pointer")
    if p2 and p2.IsValid():
        p = p2

    addr = _sbvalue_to_u64(p)
    if addr == 0:
        return limbs

    process = any_valobj.GetProcess()
    if not process or not process.IsValid():
        return limbs

    err = lldb.SBError()
    raw = process.ReadMemory(addr, length * 4, err)
    if err.Fail() or raw is None:
        return limbs

    # Decode u32 limbs, little-endian per limb
    for i in range(length):
        chunk = raw[i * 4 : (i + 1) * 4]
        limbs.append(int.from_bytes(chunk, "little", signed=False))

    return limbs


def bigdecimal_summary(valobj, internal_dict):
    """
    Pretty-print bigdecimal::BigDecimal.
    BigDecimal { int_val: BigInt, scale: i64 }
    BigInt magnitude is stored in base 2^32 digits (u32).
    """
    try:
        # scale
        scale_obj = valobj.GetChildMemberWithName("scale")
        scale = scale_obj.GetValueAsSigned() if scale_obj and scale_obj.IsValid() else 0

        # sign
        int_val = valobj.GetChildMemberWithName("int_val")
        sign_obj = (
            int_val.GetChildMemberWithName("sign")
            if int_val and int_val.IsValid()
            else None
        )
        sign_txt = (
            (sign_obj.GetValue() or sign_obj.GetSummary() or "")
            if sign_obj and sign_obj.IsValid()
            else ""
        )
        negative = "Minus" in str(sign_txt)

        # vec path in your dump: int_val.data.data  (Vec<u32>)
        data_outer = int_val.GetChildMemberWithName("data")
        vec = (
            data_outer.GetChildMemberWithName("data")
            if data_outer and data_outer.IsValid()
            else data_outer
        )
        if not vec or not vec.IsValid():
            return "0"

        # 1) Try synthetic children
        limbs = _extract_vec_u32_limbs(vec)

        # 2) Fallback: raw memory read
        if not limbs:
            limbs = _read_vec_u32_from_memory(vec, valobj)

        if not limbs:
            return "0"

        # BigUint digits are typically least-significant first.
        mag = 0
        for limb in reversed(limbs):
            mag = (mag << 32) | int(limb)

        if negative:
            mag = -mag

        # Apply decimal scale (BigDecimal uses value = int_val * 10^(-scale) in many implementations;
        # but your observed behaviour earlier matched "insert decimal when scale is negative".
        s = str(abs(mag))
        if scale > 0:
            # scale = number of fractional digits
            if len(s) > scale:
                s = s[:-scale] + "." + s[-scale:]
            else:
                s = "0." + ("0" * (scale - len(s))) + s
        elif scale < 0:
            # negative scale -> multiply by 10^(-scale) (append zeros)
            s = s + ("0" * (-scale))
        else:
            # scale == 0 -> integer, leave as-is
            pass

        if negative and s != "0":
            s = "-" + s
        return s

    except Exception as e:
        return f"<BigDecimal error: {e}>"


def _is_leap(year: int) -> bool:
    # Same leap-year rules as proleptic Gregorian
    if year % 4 != 0:
        return False
    if year % 100 != 0:
        return True
    return year % 400 == 0


def _ordinal_to_month_day(year: int, ordinal: int):
    # ordinal: 1..365/366  →  (month, day)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if _is_leap(year):
        days_in_month[1] = 29

    o = int(ordinal)
    month = 1
    for dim in days_in_month:
        if o <= dim:
            day = o
            return month, day
        o -= dim
        month += 1

    # Should not happen if chrono invariants hold
    return 1, 1


def chrono_naivedate_summary(valobj, internal_dict):
    """
    Pretty-print chrono::naive::date::NaiveDate as YYYY-MM-DD
    by decoding the internal yof: NonZeroI32 = (year << 13) | (ordinal << 4) | flags
    """

    # 1. Get the yof field
    yof = valobj.GetChildMemberWithName("yof")
    if not yof.IsValid():
        return "<invalid NaiveDate: no yof>"

    # 2. Drill down through the NonZeroI32 wrapper(s) to the i32
    inner = yof
    # Your printout showed yof = { __0 = (__0 = 16593129) },
    # so we walk through a couple of single-child layers.
    for _ in range(3):
        if inner.GetNumChildren() == 1:
            inner = inner.GetChildAtIndex(0)
        else:
            break

    if not inner.IsValid():
        return "<invalid NaiveDate: yof inner>"

    yof_int = inner.GetValueAsSigned()

    # 3. Decode year and ordinal
    # Comment from chrono source: yof: NonZeroI32, // (year << 13) | of
    # and of = (ordinal << 4) | flags
    ORDINAL_MASK = 0x1FF  # 9 bits

    year = yof_int >> 13
    ordinal = (yof_int >> 4) & ORDINAL_MASK

    if ordinal <= 0:
        return f"{year:04d}-??-??"

    month, day = _ordinal_to_month_day(year, ordinal)
    return f"{year:04d}-{month:02d}-{day:02d}"


def chrono_option_naivedate_summary(valobj, internal_dict):
    """
    Pretty-print Option<NaiveDate> when LLDB shows it as:

      $variants$ = {
        $variant$0 = (... None ...)
        $variant$  = { value = (__0 = <NaiveDate>) }
      }

    Result:
      - "None"
      - "Some(YYYY-MM-DD)"
    """
    try:
        variants = valobj.GetChildMemberWithName("$variants$")
        if not variants or not variants.IsValid():
            return "<Option<NaiveDate>: no $variants$>"

        # Try the "Some" side first: $variant$ -> value -> __0 (NaiveDate)
        some_variant = variants.GetChildMemberWithName("$variant$")
        if some_variant and some_variant.IsValid():
            value = some_variant.GetChildMemberWithName("value")
            if value and value.IsValid():
                # tuple struct, child is usually index 0 or "__0"
                inner = value.GetChildAtIndex(0)
                if not inner or not inner.IsValid():
                    inner = value.GetChildMemberWithName("__0")

                if inner and inner.IsValid():
                    # inner *is* the NaiveDate that already works
                    return (
                        "Some(" + chrono_naivedate_summary(inner, internal_dict) + ")"
                    )

        # If we didn’t find a valid "Some" payload, treat it as None.
        # (The $variant$0 branch describes the None variant.)
        return "None"
    except Exception as e:
        return f"<Option<NaiveDate> error: {e}>"


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        "type summary add -F lldb_rust_formatters.bigdecimal_summary bigdecimal::BigDecimal"
    )
    debugger.HandleCommand(
        'type summary add -F lldb_rust_formatters.bigdecimal_summary "bigdecimal::bigdecimal::BigDecimal"'
    )

    # NaiveDate itself
    debugger.HandleCommand(
        "type summary add -F lldb_rust_formatters.chrono_naivedate_summary "
        "chrono::naive::date::NaiveDate"
    )

    # Various ways Rust might spell Option<NaiveDate>
    debugger.HandleCommand(
        "type summary add -F lldb_rust_formatters.chrono_option_naivedate_summary "
        '"core::option::Option<chrono::naive::date::NaiveDate>"'
    )
    debugger.HandleCommand(
        "type summary add -F lldb_rust_formatters.chrono_option_naivedate_summary "
        '"std::option::Option<chrono::naive::date::NaiveDate>"'
    )
    debugger.HandleCommand(
        "type summary add -F lldb_rust_formatters.chrono_option_naivedate_summary "
        '"Option<chrono::naive::date::NaiveDate>"'
    )
