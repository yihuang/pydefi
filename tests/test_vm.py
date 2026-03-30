"""Unit tests for pydefi.vm — Program builder and ABI helpers.

These tests are pure Python (no network, no Solidity compilation) and verify:

 - That :class:`~pydefi.vm.builder.Program` produces the same bytecode as
   the equivalent low-level functional builders from :mod:`pydefi.vm.program`.
 - Label-based jump resolution in :meth:`~pydefi.vm.builder.Program.build`.
 - The :meth:`~pydefi.vm.builder.Program.call_contract` high-level helper.
 - ABI calldata helpers in :mod:`pydefi.vm.abi`.
 - Error cases (duplicate label, undefined label, invalid arguments).
"""

from __future__ import annotations

import struct

import pytest

from pydefi.vm import (
    Program,
    erc20_approve,
    erc20_balance_of,
    erc20_transfer,
    erc20_transfer_from,
)
from pydefi.vm.program import (
    OP_CALL,
    OP_JUMP,
    OP_JUMPI,
    OP_PUSH_ADDR,
    OP_PUSH_BYTES,
    OP_PUSH_U256,
    assert_ge,
    assert_le,
    balance_of,
    call,
    dup,
    jump,
    jumpi,
    load_reg,
    patch_addr,
    patch_u256,
    pop,
    push_addr,
    push_bytes,
    push_u256,
    ret_slice,
    ret_u256,
    revert_if,
    self_addr,
    store_reg,
    sub,
    swap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADDR_A = "0x" + "aA" * 20
ADDR_B = "0x" + "bB" * 20
ADDR_ZERO = "0x" + "00" * 20


# ---------------------------------------------------------------------------
# Program: basic instruction emission
# ---------------------------------------------------------------------------


class TestProgramInstructionEmission:
    """Verify that every Program method emits the same bytes as its low-level counterpart."""

    def test_push_u256(self):
        assert Program().push_u256(42).build() == push_u256(42)

    def test_push_addr(self):
        assert Program().push_addr(ADDR_A).build() == push_addr(ADDR_A)

    def test_push_bytes(self):
        data = b"\xde\xad\xbe\xef"
        assert Program().push_bytes(data).build() == push_bytes(data)

    def test_dup(self):
        assert Program().dup().build() == dup()

    def test_swap(self):
        assert Program().swap().build() == swap()

    def test_pop(self):
        assert Program().pop().build() == pop()

    def test_load_reg(self):
        assert Program().load_reg(3).build() == load_reg(3)

    def test_store_reg(self):
        assert Program().store_reg(7).build() == store_reg(7)

    def test_jump_int(self):
        assert Program().jump(0).build() == jump(0)

    def test_jumpi_int(self):
        assert Program().jumpi(0).build() == jumpi(0)

    def test_revert_if(self):
        assert Program().revert_if("oops").build() == revert_if("oops")

    def test_assert_ge(self):
        assert Program().assert_ge("min not met").build() == assert_ge("min not met")

    def test_assert_le(self):
        assert Program().assert_le("max exceeded").build() == assert_le("max exceeded")

    def test_call(self):
        assert Program().call().build() == call()
        assert Program().call(require_success=False).build() == call(require_success=False)

    def test_balance_of(self):
        assert Program().balance_of().build() == balance_of()

    def test_self_addr(self):
        assert Program().self_addr().build() == self_addr()

    def test_sub(self):
        assert Program().sub().build() == sub()

    def test_patch_u256(self):
        assert Program().patch_u256(4).build() == patch_u256(4)

    def test_patch_addr(self):
        assert Program().patch_addr(16).build() == patch_addr(16)

    def test_ret_u256(self):
        assert Program().ret_u256(0).build() == ret_u256(0)

    def test_ret_slice(self):
        assert Program().ret_slice(0, 32).build() == ret_slice(0, 32)


# ---------------------------------------------------------------------------
# Program: method chaining
# ---------------------------------------------------------------------------


class TestProgramChaining:
    def test_chain_produces_correct_concat(self):
        expected = push_u256(1) + push_u256(2) + sub()
        actual = Program().push_u256(1).push_u256(2).sub().build()
        assert actual == expected

    def test_len_matches_bytecode_length(self):
        p = Program().push_u256(0).push_addr(ADDR_A)
        assert len(p) == len(push_u256(0) + push_addr(ADDR_A))

    def test_bytes_builtin(self):
        p = Program().push_u256(99)
        assert bytes(p) == push_u256(99)

    def test_repr_contains_label_info(self):
        p = Program().push_u256(0).label("done")
        r = repr(p)
        assert "done" in r


# ---------------------------------------------------------------------------
# Program: labels and jumps
# ---------------------------------------------------------------------------


class TestProgramLabels:
    def test_jump_label_resolves_to_correct_offset(self):
        # Layout: JUMP(3 bytes) | target instruction (push_u256 = 33 bytes)
        # Label "start" placed at byte 3 (after JUMP)
        p = Program().jump("start").label("start").push_u256(0)
        bytecode = p.build()
        # The JUMP target (bytes 1-2) should be 3
        assert bytecode[0] == OP_JUMP
        target = struct.unpack(">H", bytecode[1:3])[0]
        assert target == 3  # one JUMP instruction = 3 bytes

    def test_jumpi_label_resolves_to_correct_offset(self):
        # push_u256(1) [33 bytes] then JUMPI -> label "skip" [3 bytes] then push_u256(99) [33]
        # label "skip" placed after the JUMPI
        p = (
            Program()
            .push_u256(1)
            .jumpi("skip")
            .push_u256(99)
            .label("skip")
            .push_u256(0)
        )
        bytecode = p.build()
        # JUMPI instruction starts at byte 33
        assert bytecode[33] == OP_JUMPI
        target = struct.unpack(">H", bytecode[34:36])[0]
        # label "skip" is placed after: push_u256(1)[33] + JUMPI[3] + push_u256(99)[33] = 69
        assert target == 69

    def test_forward_jump_skips_instruction(self):
        # Build: JUMP("end") | push_u256(99) | <label end> | push_u256(1)
        # After jump, push_u256(99) is unreachable; the resolved bytecode is just
        # the three concatenated instructions with correct target offset embedded.
        p = (
            Program()
            .jump("end")
            .push_u256(99)     # would be skipped at runtime
            .label("end")
            .push_u256(1)
        )
        bytecode = p.build()
        assert bytecode[0] == OP_JUMP
        target = struct.unpack(">H", bytecode[1:3])[0]
        # jump(3 bytes) + push_u256(99)(33 bytes) = 36 bytes before label
        assert target == 36

    def test_duplicate_label_raises(self):
        with pytest.raises(ValueError, match="duplicate label"):
            Program().label("x").label("x")

    def test_undefined_label_raises_at_build(self):
        with pytest.raises(ValueError, match="undefined label"):
            Program().jump("nowhere").build()

    def test_multiple_jumps_to_same_label(self):
        p = (
            Program()
            .push_u256(0)
            .jumpi("end")
            .push_u256(1)
            .jump("end")
            .label("end")
            .push_u256(2)
        )
        bytecode = p.build()
        # Both jumps should resolve to the same target
        # Layout: push_u256(0)[33] + JUMPI[3] + push_u256(1)[33] + JUMP[3] + push_u256(2)[33]
        # label "end" is at offset 33+3+33+3 = 72
        assert bytecode[33] == OP_JUMPI
        t1 = struct.unpack(">H", bytecode[34:36])[0]
        assert bytecode[69] == OP_JUMP
        t2 = struct.unpack(">H", bytecode[70:72])[0]
        assert t1 == t2 == 72


# ---------------------------------------------------------------------------
# Program: call_contract helper
# ---------------------------------------------------------------------------


class TestCallContractHelper:
    def test_call_contract_matches_manual_sequence(self):
        calldata = bytes.fromhex("a9059cbb" + "00" * 12 + "bb" * 20 + "00" * 31 + "64")
        expected = (
            push_bytes(calldata)
            + push_u256(0)
            + push_addr(ADDR_A)
            + push_u256(0)
            + call(require_success=True)
        )
        actual = Program().call_contract(ADDR_A, calldata).build()
        assert actual == expected

    def test_call_contract_with_value_and_gas(self):
        calldata = b"\x12\x34\x56\x78"
        expected = (
            push_bytes(calldata)
            + push_u256(10**18)
            + push_addr(ADDR_B)
            + push_u256(50000)
            + call(require_success=True)
        )
        actual = (
            Program()
            .call_contract(ADDR_B, calldata, value=10**18, gas=50000)
            .build()
        )
        assert actual == expected

    def test_call_contract_no_require_success(self):
        calldata = b"\xab\xcd"
        expected = (
            push_bytes(calldata)
            + push_u256(0)
            + push_addr(ADDR_A)
            + push_u256(0)
            + call(require_success=False)
        )
        actual = (
            Program()
            .call_contract(ADDR_A, calldata, require_success=False)
            .build()
        )
        assert actual == expected

    def test_call_contract_push_bytes_opcode(self):
        # First byte of the output should be OP_PUSH_BYTES
        bytecode = Program().call_contract(ADDR_A, b"\x00").build()
        assert bytecode[0] == OP_PUSH_BYTES

    def test_call_contract_address_embedded(self):
        # The address should be present in the bytecode
        bytecode = Program().call_contract(ADDR_A, b"\x00").build()
        assert bytes.fromhex(ADDR_A[2:]) in bytecode


# ---------------------------------------------------------------------------
# ABI helpers
# ---------------------------------------------------------------------------


class TestABIHelpers:
    def test_erc20_transfer_selector(self):
        cd = erc20_transfer(ADDR_A, 100)
        assert cd[:4] == bytes.fromhex("a9059cbb")

    def test_erc20_transfer_total_length(self):
        cd = erc20_transfer(ADDR_A, 100)
        assert len(cd) == 4 + 32 + 32  # selector + address_word + uint256

    def test_erc20_transfer_address_encoding(self):
        cd = erc20_transfer(ADDR_A, 0)
        # Address is right-aligned in a 32-byte word (bytes 4..35)
        addr_word = cd[4:36]
        assert addr_word[:12] == b"\x00" * 12
        assert addr_word[12:] == bytes.fromhex(ADDR_A[2:])

    def test_erc20_transfer_amount_encoding(self):
        amount = 1_000_000
        cd = erc20_transfer(ADDR_A, amount)
        amount_word = cd[36:68]
        assert int.from_bytes(amount_word, "big") == amount

    def test_erc20_approve_selector(self):
        cd = erc20_approve(ADDR_B, 2**256 - 1)
        assert cd[:4] == bytes.fromhex("095ea7b3")

    def test_erc20_approve_max_approval(self):
        cd = erc20_approve(ADDR_B, 2**256 - 1)
        amount_word = cd[36:68]
        assert amount_word == b"\xff" * 32

    def test_erc20_transfer_from_selector(self):
        cd = erc20_transfer_from(ADDR_A, ADDR_B, 500)
        assert cd[:4] == bytes.fromhex("23b872dd")

    def test_erc20_transfer_from_total_length(self):
        cd = erc20_transfer_from(ADDR_A, ADDR_B, 500)
        assert len(cd) == 4 + 32 + 32 + 32

    def test_erc20_transfer_from_addresses(self):
        cd = erc20_transfer_from(ADDR_A, ADDR_B, 0)
        from_word = cd[4:36]
        to_word = cd[36:68]
        assert from_word[12:] == bytes.fromhex(ADDR_A[2:])
        assert to_word[12:] == bytes.fromhex(ADDR_B[2:])

    def test_erc20_balance_of_selector(self):
        cd = erc20_balance_of(ADDR_A)
        assert cd[:4] == bytes.fromhex("70a08231")

    def test_erc20_balance_of_total_length(self):
        cd = erc20_balance_of(ADDR_A)
        assert len(cd) == 4 + 32

    def test_erc20_helpers_bad_address(self):
        with pytest.raises(ValueError, match="bad address length"):
            erc20_transfer("0x1234", 100)  # too short

    def test_erc20_approve_zero_amount(self):
        cd = erc20_approve(ADDR_A, 0)
        assert cd[36:68] == b"\x00" * 32


# ---------------------------------------------------------------------------
# Integration: compose helpers together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_approve_then_balance_check(self):
        """Program that approves and then checks balance — pure byte verification."""
        approve_cd = erc20_approve(ADDR_B, 10**18)
        bytecode = (
            Program()
            .call_contract(ADDR_A, approve_cd)
            .pop()
            .push_addr(ADDR_A)
            .push_addr(ADDR_B)
            .balance_of()
            .push_u256(0)
            .assert_ge("balance too low")
            .build()
        )
        # Verify it's non-empty and starts with PUSH_BYTES
        assert len(bytecode) > 0
        assert bytecode[0] == OP_PUSH_BYTES

    def test_conditional_skip_with_label(self):
        """Verify label resolution in a real conditional flow."""
        p = (
            Program()
            .push_u256(0)          # condition = false
            .jumpi("skip")
            .push_u256(99)         # unreachable path
            .label("skip")
            .push_u256(1)
        )
        bytecode = p.build()
        # The JUMPI target should point past the push_u256(99)
        # push_u256(0)=33 bytes, JUMPI=3 bytes => label at 33+3+33=69
        assert bytecode[33] == OP_JUMPI
        target = struct.unpack(">H", bytecode[34:36])[0]
        assert target == 69

    def test_multi_call_program(self):
        """Three sequential calls produce a valid byte sequence."""
        cd1 = erc20_approve(ADDR_B, 100)
        cd2 = erc20_transfer(ADDR_A, 100)
        cd3 = erc20_balance_of(ADDR_A)
        bytecode = (
            Program()
            .call_contract(ADDR_A, cd1)
            .pop()
            .call_contract(ADDR_A, cd2)
            .pop()
            .call_contract(ADDR_A, cd3)
            .pop()
            .build()
        )
        assert len(bytecode) > 0
        # Starts with PUSH_BYTES
        assert bytecode[0] == OP_PUSH_BYTES
