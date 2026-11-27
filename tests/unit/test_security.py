import pytest

from backend.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_roundtrip():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)
    assert not verify_password("wrong", h)


def test_password_too_long():
    with pytest.raises(ValueError):
        hash_password("я" * 40)  # 80 байт в UTF-8


def test_token_roundtrip():
    assert decode_access_token(create_access_token(42)) == 42


def test_bad_token():
    assert decode_access_token("not-a-token") is None
