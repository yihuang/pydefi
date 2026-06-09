from __future__ import annotations

from typing import Any

from eth_account import Account
from eth_account.messages import encode_typed_data


def sign_typed_data(typed_data: dict[str, Any], private_key: str) -> bytes:
    """Sign EIP-712 typed-data; returns the 65-byte signature."""
    signed = Account.from_key(private_key).sign_message(encode_typed_data(full_message=typed_data))
    return bytes(signed.signature)
