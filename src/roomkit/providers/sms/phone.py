"""Phone number normalization utilities."""

from __future__ import annotations


def normalize_phone(number: str, default_region: str = "US") -> str:
    """Normalize a phone number to E.164 format.

    Args:
        number: Phone number in any common format.
        default_region: ISO 3166-1 alpha-2 country code for numbers without
            country code (default: "US").

    Returns:
        Phone number in E.164 format (e.g., "+14185551234").

    Raises:
        ImportError: If phonenumbers library is not installed.
        ValueError: If the number cannot be parsed or is invalid.

    Example:
        >>> normalize_phone("418-555-1234", "CA")
        '+14185551234'
        >>> normalize_phone("+1 (418) 555-1234")
        '+14185551234'
        >>> normalize_phone("14185551234")
        '+14185551234'
    """
    try:
        import phonenumbers
    except ImportError as exc:
        raise ImportError(
            "phonenumbers is required for phone normalization. "
            "Install it with: pip install roomkit[phonenumbers]"
        ) from exc

    # Handle numbers that start with country code but no +
    cleaned = number.strip()
    if cleaned and cleaned[0].isdigit() and len(cleaned) >= 10:
        # Try parsing with + prefix first
        try:
            parsed = phonenumbers.parse(f"+{cleaned}", None)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except phonenumbers.NumberParseException:
            pass

    try:
        parsed = phonenumbers.parse(cleaned, default_region)
    except phonenumbers.NumberParseException as exc:
        raise ValueError(f"Cannot parse phone number: {number}") from exc

    if not phonenumbers.is_valid_number(parsed):
        raise ValueError(f"Invalid phone number: {number}")

    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def is_valid_phone(number: str, default_region: str = "US") -> bool:
    """Check if a phone number is valid.

    Args:
        number: Phone number in any common format.
        default_region: ISO 3166-1 alpha-2 country code for numbers without
            country code (default: "US").

    Returns:
        True if the number is valid, False otherwise.

    Note:
        Returns False if phonenumbers library is not installed.
    """
    try:
        normalize_phone(number, default_region)
        return True
    except (ImportError, ValueError):
        return False
