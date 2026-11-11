from app.core import validators as V


def test_utr_formats():
    assert V.validate_utr("4123 9876 5432").kind == "upi_rrn"
    assert V.validate_utr("SBIN423123456789").kind == "neft_utr"
    assert V.validate_utr("T2611111042123456789012").kind == "app_txn_id"
    assert not V.validate_utr("12345").ok
    assert V.validate_utr("೪೧೨೩೯೮೭೬೫೪೩೨").normalized == "412398765432"  # Kannada digits


def test_phone_upi_ifsc():
    assert V.validate_phone("98765 43210").normalized == "+919876543210"
    assert V.validate_phone("+91-9876543210").normalized == "+919876543210"
    assert V.validate_phone("+92 300 1234567").kind == "international"
    assert not V.validate_phone("12345").ok
    assert V.validate_upi("CBI.Verify2@ybl").normalized == "cbi.verify2@ybl"
    assert not V.validate_upi("not an upi").ok
    assert V.validate_ifsc("sbin0001234").normalized == "SBIN0001234"


def test_amounts_indian_units():
    assert V.parse_amount("₹1,20,000").normalized == "120000.00"
    assert V.parse_amount("Rs. 50,000.50").normalized == "50000.50"
    assert V.parse_amount("1.5 lakh").normalized == "150000.00"
    assert V.parse_amount("50 ಸಾವಿರ").normalized == "50000.00"
    assert V.parse_amount("2 करोड़").normalized == "20000000.00"
    assert not V.parse_amount("no money").ok


def test_format_and_spoken_digits():
    assert V.format_inr("1234567.5") == "₹12,34,567.50"
    assert V.format_inr(500) == "₹500.00"
    assert V.spoken_digits_to_numbers("my number is nine eight seven six five four three two one zero ok") \
        == "my number is 9876543210 ok"
    assert V.spoken_digits_to_numbers("I have one son") == "I have one son"


def test_datetime_parse():
    assert V.to_iso_ist("11 Nov 2026, 10:42 am").startswith("2026-11-11T10:42:00+05:30")
    assert V.to_iso_ist("ಸುಮಾರು 11 ಗಂಟೆ") is None
