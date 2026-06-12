from __future__ import annotations

import pytest

from pydefi.types import Address
from pydefi.vm.yield_deposit import (
    AMOUNT_SENTINEL,
    build_deposit_program,
    build_initializer,
    find_amount_offset,
)

_OWNER = Address("0x" + "AA" * 20)
_TOKEN = Address("0x" + "a0" * 20)
_PROTOCOL = Address("0x" + "CE" * 20)
_DEFIVM = Address("0x" + "DD" * 20)
_WORD = AMOUNT_SENTINEL.to_bytes(32, "big")
# Stand-in supply calldata: selector + one arg word + the sentinel amount word.
_TEMPLATE = b"\xde\xad\xbe\xef" + b"\x00" * 28 + _WORD


def test_find_amount_offset():
    assert find_amount_offset(_TEMPLATE) == 32


@pytest.mark.parametrize("template", [b"\x12\x34\x56", 2 * _WORD], ids=["missing", "duplicate"])
def test_find_amount_offset_requires_exactly_one(template):
    with pytest.raises(ValueError, match="exactly once"):
        find_amount_offset(template)


def test_deposit_program_embeds_template_and_is_deterministic():
    program = build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 10**6)
    assert _TEMPLATE in program  # call_raw embeds the template; the amount is patched at runtime
    assert program == build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 10**6)
    # the fee and reset legs change the program (and therefore the deposit address)
    assert program != build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 0)
    assert program != build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 10**6, approve_reset=True)


def test_initializer_commits_owner_and_program():
    """The Safe setup calldata — and therefore the CREATE2 address — binds the
    owner and the exact program."""
    program = build_deposit_program(_TOKEN, _PROTOCOL, _TEMPLATE, 0)
    init = build_initializer(_OWNER, _DEFIVM, program)
    assert program in init  # embedded via DeFiVM.execute(program)
    assert bytes(_OWNER) in init
    assert init != build_initializer(Address("0x" + "BB" * 20), _DEFIVM, program)
    assert init != build_initializer(_OWNER, _DEFIVM, program + b"\x00")
