def strtobool(val: str) -> int:
    """Convert a string truth value to 1 or 0.

    Replacement for the removed distutils.util.strtobool (gone from the stdlib
    in Python 3.12). True values are "y", "yes", "t", "true", "on", "1"; false
    values are "n", "no", "f", "false", "off", "0". Raises ValueError otherwise.
    """
    val = str(val).strip().lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {val!r}")
