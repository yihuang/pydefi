"""Unit tests for pydefi.vm — Program builder and ABI helpers.

These tests are pure Python (no network, no Solidity compilation) and verify:

 - That :class:`~pydefi.vm.builder.Program` produces the same bytecode as
   the equivalent low-level functional builders from :mod:`pydefi.vm.program`.
 - Label-based jump resolution in :meth:`~pydefi.vm.builder.Program.build`.
 - The :meth:`~pydefi.vm.builder.Program.call_contract` high-level helper.
 - Program composition via :meth:`~pydefi.vm.builder.Program.extend`,
   ``+``, ``+=`` and :meth:`~pydefi.vm.builder.Program.compose`.
 - Calldata surgery via :meth:`~pydefi.vm.builder.Program.call_with_patches`.
 - ABI calldata helpers in :mod:`pydefi.vm.abi`.
 - Error cases (duplicate label, undefined label, invalid arguments).
"""

from __future__ import annotations

import struct

import pytest

from pydefi.vm import Program
from pydefi.vm.program import (
    OP_ADD,
    OP_AND,
    OP_DIV,
    OP_EQ,
    OP_GT,
    OP_ISZERO,
    OP_JUMP,
    OP_JUMPI,
    OP_LT,
    OP_MOD,
    OP_MUL,
    OP_NOT,
    OP_OR,
    OP_SHL,
    OP_SHR,
    OP_SUB,
    OP_XOR,
    add,
    assert_ge,
    assert_le,
    balance_of,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    call,
    div,
    dup,
    eq,
    gas_opcode,
    gt,
    iszero,
    jump,
    jumpi,
    load_reg,
    lt,
    mod,
    mul,
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
    shl,
    shr,
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

    def test_add(self):
        assert Program().add().build() == add()

    def test_mul(self):
        assert Program().mul().build() == mul()

    def test_div(self):
        assert Program().div().build() == div()

    def test_mod(self):
        assert Program().mod().build() == mod()

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
        # Layout: jump("start")[4 bytes: PUSH2 hi lo JUMP] | JUMPDEST[1] | push_u256(0)[33]
        # JUMPDEST at byte 4 — that is the jump target
        p = Program().jump("start").label("start").push_u256(0)
        bytecode = p.build()
        # First byte is PUSH2 (0x61) which is OP_JUMP in the new scheme
        assert bytecode[0] == OP_JUMP
        target = struct.unpack(">H", bytecode[1:3])[0]
        assert target == 4  # JUMPDEST is at byte 4

    def test_jumpi_label_resolves_to_correct_offset(self):
        # push_u256(1) [33 bytes] then jumpi("skip") [4 bytes] then push_u256(99) [33]
        # then JUMPDEST [1] then push_u256(0) [33]
        p = Program().push_u256(1).jumpi("skip").push_u256(99).label("skip").push_u256(0)
        bytecode = p.build()
        # jumpi sequence starts at byte 33: PUSH2(1) hi(1) lo(1) JUMPI(1)
        assert bytecode[33] == OP_JUMPI
        target = struct.unpack(">H", bytecode[34:36])[0]
        # JUMPDEST at: push_u256(1)[33] + jumpi[4] + push_u256(99)[33] = 70
        assert target == 70

    def test_forward_jump_skips_instruction(self):
        # Build: jump("end")[4] | push_u256(99)[33] | JUMPDEST[1] | push_u256(1)[33]
        p = (
            Program()
            .jump("end")
            .push_u256(99)  # would be skipped at runtime
            .label("end")
            .push_u256(1)
        )
        bytecode = p.build()
        assert bytecode[0] == OP_JUMP
        target = struct.unpack(">H", bytecode[1:3])[0]
        # JUMPDEST at: jump[4] + push_u256(99)[33] = 37
        assert target == 37

    def test_duplicate_label_raises(self):
        with pytest.raises(ValueError, match="duplicate label"):
            Program().label("x").label("x")

    def test_undefined_label_raises_at_build(self):
        with pytest.raises(ValueError, match="undefined label"):
            Program().jump("nowhere").build()

    def test_multiple_jumps_to_same_label(self):
        p = Program().push_u256(0).jumpi("end").push_u256(1).jump("end").label("end").push_u256(2)
        bytecode = p.build()
        # Both jumps should resolve to the same target
        # Layout: push_u256(0)[33] + jumpi[4] + push_u256(1)[33] + jump[4] + JUMPDEST[1] + push_u256(2)[33]
        # JUMPDEST (label "end") is at offset 33+4+33+4 = 74
        assert bytecode[33] == OP_JUMPI
        t1 = struct.unpack(">H", bytecode[34:36])[0]
        assert bytecode[70] == OP_JUMP
        t2 = struct.unpack(">H", bytecode[71:73])[0]
        assert t1 == t2 == 74


# ---------------------------------------------------------------------------
# Program: call_contract helper
# ---------------------------------------------------------------------------


class TestCallContractHelper:
    def test_call_contract_matches_manual_sequence(self):
        calldata = bytes.fromhex("a9059cbb" + "00" * 12 + "bb" * 20 + "00" * 31 + "64")
        expected = (
            push_u256(0)
            + push_u256(0)
            + push_bytes(calldata)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()
            + call(require_success=True)
        )
        actual = Program().call_contract(ADDR_A, calldata).build()
        assert actual == expected

    def test_call_contract_with_value_and_gas(self):
        calldata = b"\x12\x34\x56\x78"
        expected = (
            push_u256(0)
            + push_u256(0)
            + push_bytes(calldata)
            + push_u256(10**18)
            + push_addr(ADDR_B)
            + push_u256(50000)
            + call(require_success=True)
        )
        actual = Program().call_contract(ADDR_B, calldata, value=10**18, gas=50000).build()
        assert actual == expected

    def test_call_contract_no_require_success(self):
        calldata = b"\xab\xcd"
        expected = (
            push_u256(0)
            + push_u256(0)
            + push_bytes(calldata)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()
            + call(require_success=False)
        )
        actual = Program().call_contract(ADDR_A, calldata, require_success=False).build()
        assert actual == expected

    def test_call_contract_push_bytes_opcode(self):
        # push_bytes embeds calldata as PUSH32 immediates; verify data is present
        bytecode = Program().call_contract(ADDR_A, b"\x00").build()
        assert b"\x00" * 32 in bytecode  # zero-padded chunk embedded via PUSH32

    def test_call_contract_address_embedded(self):
        # The address should be present in the bytecode
        bytecode = Program().call_contract(ADDR_A, b"\x00").build()
        assert bytes.fromhex(ADDR_A[2:]) in bytecode


# ---------------------------------------------------------------------------
# ABI helpers (via eth-contract ERC20 contract object)
# ---------------------------------------------------------------------------


class TestABIHelpers:
    def test_erc20_transfer_selector(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.transfer(ADDR_A, 100).data)
        assert cd[:4] == bytes.fromhex("a9059cbb")

    def test_erc20_transfer_total_length(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.transfer(ADDR_A, 100).data)
        assert len(cd) == 4 + 32 + 32  # selector + address_word + uint256

    def test_erc20_transfer_address_encoding(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.transfer(ADDR_A, 0).data)
        # Address is right-aligned in a 32-byte word (bytes 4..35)
        addr_word = cd[4:36]
        assert addr_word[:12] == b"\x00" * 12
        assert addr_word[12:] == bytes.fromhex(ADDR_A[2:])

    def test_erc20_transfer_amount_encoding(self):
        from eth_contract.erc20 import ERC20

        amount = 1_000_000
        cd = bytes(ERC20.fns.transfer(ADDR_A, amount).data)
        amount_word = cd[36:68]
        assert int.from_bytes(amount_word, "big") == amount

    def test_erc20_approve_selector(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.approve(ADDR_B, 2**256 - 1).data)
        assert cd[:4] == bytes.fromhex("095ea7b3")

    def test_erc20_approve_max_approval(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.approve(ADDR_B, 2**256 - 1).data)
        amount_word = cd[36:68]
        assert amount_word == b"\xff" * 32

    def test_erc20_transfer_from_selector(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.transferFrom(ADDR_A, ADDR_B, 500).data)
        assert cd[:4] == bytes.fromhex("23b872dd")

    def test_erc20_transfer_from_total_length(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.transferFrom(ADDR_A, ADDR_B, 500).data)
        assert len(cd) == 4 + 32 + 32 + 32

    def test_erc20_transfer_from_addresses(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.transferFrom(ADDR_A, ADDR_B, 0).data)
        from_word = cd[4:36]
        to_word = cd[36:68]
        assert from_word[12:] == bytes.fromhex(ADDR_A[2:])
        assert to_word[12:] == bytes.fromhex(ADDR_B[2:])

    def test_erc20_balance_of_selector(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.balanceOf(ADDR_A).data)
        assert cd[:4] == bytes.fromhex("70a08231")

    def test_erc20_balance_of_total_length(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.balanceOf(ADDR_A).data)
        assert len(cd) == 4 + 32

    def test_erc20_approve_zero_amount(self):
        from eth_contract.erc20 import ERC20

        cd = bytes(ERC20.fns.approve(ADDR_A, 0).data)
        assert cd[36:68] == b"\x00" * 32


# ---------------------------------------------------------------------------
# Integration: compose helpers together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_approve_then_balance_check(self):
        """Program that approves and then checks balance — pure byte verification."""
        from eth_contract.erc20 import ERC20

        approve_cd = bytes(ERC20.fns.approve(ADDR_B, 10**18).data)
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
        # Verify it's non-empty and contains calldata bytes
        assert len(bytecode) > 0
        assert bytes(approve_cd[:4]) in bytecode  # selector embedded via PUSH32

    def test_conditional_skip_with_label(self):
        """Verify label resolution in a real conditional flow."""
        p = (
            Program()
            .push_u256(0)  # condition = false
            .jumpi("skip")
            .push_u256(99)  # unreachable path
            .label("skip")
            .push_u256(1)
        )
        bytecode = p.build()
        # The JUMPI target should point past the push_u256(99)
        # push_u256(0)=33 bytes, jumpi=4 bytes, push_u256(99)=33 => JUMPDEST at 33+4+33=70
        assert bytecode[33] == OP_JUMPI
        target = struct.unpack(">H", bytecode[34:36])[0]
        assert target == 70

    def test_multi_call_program(self):
        """Three sequential calls produce a valid byte sequence."""
        from eth_contract.erc20 import ERC20

        cd1 = bytes(ERC20.fns.approve(ADDR_B, 100).data)
        cd2 = bytes(ERC20.fns.transfer(ADDR_A, 100).data)
        cd3 = bytes(ERC20.fns.balanceOf(ADDR_A).data)
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
        # Contains calldata bytes (selectors embedded via PUSH32 in push_bytes)


# ---------------------------------------------------------------------------
# Program: composition
# ---------------------------------------------------------------------------


class TestProgramComposition:
    def test_extend_concatenates_bytecode(self):
        p1 = Program().push_u256(1)
        p2 = Program().push_u256(2)
        p1.extend(p2)
        assert p1.build() == push_u256(1) + push_u256(2)

    def test_add_returns_new_program(self):
        p1 = Program().push_u256(1)
        p2 = Program().push_u256(2)
        result = p1 + p2
        # Original programs unchanged
        assert p1.build() == push_u256(1)
        assert p2.build() == push_u256(2)
        # Combined result is correct
        assert result.build() == push_u256(1) + push_u256(2)

    def test_iadd_modifies_in_place(self):
        p1 = Program().push_u256(10)
        p2 = Program().push_u256(20)
        p1 += p2
        assert p1.build() == push_u256(10) + push_u256(20)

    def test_compose_list(self):
        parts = [
            Program().push_u256(1),
            Program().push_u256(2),
            Program().push_u256(3),
        ]
        result = Program.compose(parts)
        assert result.build() == push_u256(1) + push_u256(2) + push_u256(3)

    def test_compose_empty_list(self):
        result = Program.compose([])
        assert result.build() == b""

    def test_extend_label_offset_adjusted(self):
        """Labels in the appended program have their positions shifted correctly."""
        # Layout: jump("here")[4 bytes: PUSH2+hi+lo+JUMP] | JUMPDEST[1] + push_u256(1) from p2
        # After extend, "here" should resolve to offset 4 (JUMPDEST right after the jump).
        p1 = Program().jump("here")
        p2 = Program().label("here").push_u256(1)
        p1.extend(p2)
        bytecode = p1.build()
        assert bytecode[0] == OP_JUMP
        target = struct.unpack(">H", bytecode[1:3])[0]
        assert target == 4  # jump(4 bytes) → JUMPDEST right after

    def test_extend_fixup_offset_adjusted(self):
        """JUMP fixup offsets in appended program are shifted correctly."""
        # p2 has a forward jump: jump[4 bytes] | JUMPDEST[1] | push_u256(0)
        p2 = Program().jump("done").label("done").push_u256(0)

        p1_size = len(Program().push_u256(999))  # 33 bytes
        p1 = Program().push_u256(999)
        p1.extend(p2)

        bytecode = p1.build()
        # The PUSH2 (start of jump sequence) is at byte p1_size (33)
        assert bytecode[p1_size] == OP_JUMP
        target = struct.unpack(">H", bytecode[p1_size + 1 : p1_size + 3])[0]
        # JUMPDEST is at p1_size + 4 (after PUSH2+hi+lo+JUMP)
        assert target == p1_size + 4

    def test_extend_cross_program_label_resolution(self):
        """Jump in p1 to label defined in p2 resolves correctly."""
        p1 = Program().jump("end")
        p2 = Program().push_u256(1).label("end").push_u256(2)
        p1.extend(p2)
        bytecode = p1.build()
        # "end" is at: jump[4] + push_u256(1)[33] = offset 37
        target = struct.unpack(">H", bytecode[1:3])[0]
        assert target == 37

    def test_extend_duplicate_label_raises(self):
        p1 = Program().label("x")
        p2 = Program().label("x")
        with pytest.raises(ValueError, match="duplicate label"):
            p1.extend(p2)

    def test_add_duplicate_label_raises(self):
        p1 = Program().label("x")
        p2 = Program().label("x")
        with pytest.raises(ValueError, match="duplicate label"):
            _ = p1 + p2

    def test_compose_preserves_order(self):
        """compose([A, B, C]) == A + B + C."""
        a = Program().push_u256(1)
        b = Program().push_u256(2)
        c = Program().push_u256(3)
        composed = Program.compose([a, b, c])
        chained = Program().push_u256(1).push_u256(2).push_u256(3)
        assert composed.build() == chained.build()

    def test_compose_with_labels(self):
        """Labels from multiple sub-programs are correctly resolved after compose."""
        # Layout: push_u256(1)[33] | jump("end")[4: PUSH2+hi+lo+JUMP] | JUMPDEST[1] + push_u256(2)[33]
        # "end" (JUMPDEST) is at offset 33+4 = 37
        p1 = Program().push_u256(1)
        p2 = Program().jump("end")
        p3 = Program().label("end").push_u256(2)
        result = Program.compose([p1, p2, p3])
        bytecode = result.build()
        assert bytecode[33] == OP_JUMP
        target = struct.unpack(">H", bytecode[34:36])[0]
        assert target == 37

    def test_sub_programs_independent(self):
        """Combining programs via + does not mutate the originals."""
        p1 = Program().push_u256(7)
        p2 = Program().push_u256(8)
        _ = p1 + p2
        assert p1.build() == push_u256(7)
        assert p2.build() == push_u256(8)


# ---------------------------------------------------------------------------
# Program: call_with_patches (calldata surgery)
# ---------------------------------------------------------------------------


class TestCallWithPatches:
    def _template(self) -> bytes:
        """4-byte selector + two 32-byte zero placeholders."""
        return bytes.fromhex("deadbeef") + b"\x00" * 64

    def test_no_patches_equals_call_contract(self):
        """With an empty patches list, call_with_patches == call_contract."""
        cd = self._template()
        expected = Program().call_contract(ADDR_A, cd).build()
        actual = Program().call_with_patches(ADDR_A, cd, []).build()
        assert actual == expected

    def test_bytes_u256_patch(self):
        """Bytes opcodes for u256 patch: emits opcodes + patch_u256."""
        cd = self._template()
        # Manually build equivalent low-level sequence
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + push_u256(42)
            + patch_u256(4)
            + push_u256(0)  # value
            + push_addr(ADDR_A)
            + gas_opcode()  # gas (forward all)
            + call(True)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [("u256", 4, push_u256(42))]).build()
        assert actual == expected

    def test_bytes_addr_patch(self):
        """Bytes opcodes for addr patch: emits opcodes + patch_addr."""
        cd = self._template()
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + push_addr(ADDR_B)
            + patch_addr(16)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()  # gas (forward all)
            + call(True)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [("addr", 16, push_addr(ADDR_B))]).build()
        assert actual == expected

    def test_ret_u256_patch(self):
        """ret_u256(offset) bytes emits ret_u256 + patch_u256."""
        cd = self._template()
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + ret_u256(0)
            + patch_u256(4)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()
            + call(True)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [("u256", 4, ret_u256(0))]).build()
        assert actual == expected

    def test_reg_patch(self):
        """load_reg(idx) bytes emits load_reg + patch_u256."""
        cd = self._template()
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + load_reg(3)
            + patch_u256(4)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()
            + call(True)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [("u256", 4, load_reg(3))]).build()
        assert actual == expected

    def test_reg_patch_addr(self):
        """load_reg(idx) with kind='addr' emits load_reg + patch_addr."""
        cd = self._template()
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + load_reg(5)
            + patch_addr(16)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()
            + call(True)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [("addr", 16, load_reg(5))]).build()
        assert actual == expected

    def test_multiple_patches(self):
        """Multiple patches are applied in order."""
        cd = self._template()
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + push_u256(100)
            + patch_u256(4)
            + push_addr(ADDR_B)
            + patch_addr(4 + 32 + 12)
            + push_u256(0)
            + push_addr(ADDR_A)
            + gas_opcode()  # gas (forward all)
            + call(True)
        )
        actual = (
            Program()
            .call_with_patches(
                ADDR_A,
                cd,
                [
                    ("u256", 4, push_u256(100)),
                    ("addr", 4 + 32 + 12, push_addr(ADDR_B)),
                ],
            )
            .build()
        )
        assert actual == expected

    def test_value_and_gas_forwarded(self):
        """value and gas parameters are reflected in CALL prologue."""
        cd = self._template()
        expected = (
            push_u256(0)
            + push_u256(0)  # retSize, retOffset
            + push_bytes(cd)
            + push_u256(10**18)  # value
            + push_addr(ADDR_A)
            + push_u256(50_000)  # gas
            + call(True)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [], value=10**18, gas=50_000).build()
        assert actual == expected

    def test_require_success_false(self):
        """require_success=False emits CALL without the success-check block."""
        cd = self._template()
        expected = (
            push_u256(0) + push_u256(0) + push_bytes(cd) + push_u256(0) + push_addr(ADDR_A) + gas_opcode() + call(False)
        )
        actual = Program().call_with_patches(ADDR_A, cd, [], require_success=False).build()
        assert actual == expected

    def test_unknown_patch_kind_raises(self):
        with pytest.raises(ValueError, match="unknown patch kind"):
            Program().call_with_patches(ADDR_A, self._template(), [("bytes32", 4, push_u256(0))]).build()

    def test_non_bytes_opcodes_raises(self):
        """Passing a non-bytes source raises TypeError."""
        with pytest.raises(TypeError, match="opcodes must be bytes or bytearray"):
            Program().call_with_patches(ADDR_A, self._template(), [("u256", 4, 42)]).build()

    def test_non_bytes_str_opcodes_raises(self):
        """Passing a string raises TypeError."""
        with pytest.raises(TypeError, match="opcodes must be bytes or bytearray"):
            Program().call_with_patches(ADDR_A, self._template(), [("u256", 4, "0xdeadbeef")]).build()

    def test_non_bytes_none_opcodes_raises(self):
        """Passing None raises TypeError."""
        with pytest.raises(TypeError, match="opcodes must be bytes or bytearray"):
            Program().call_with_patches(ADDR_A, self._template(), [("u256", 4, None)]).build()

    def test_chained_with_composition(self):
        """call_with_patches works correctly when composed with other programs."""
        cd = self._template()
        step1 = Program().call_contract(ADDR_A, cd).pop()
        step2 = Program().call_with_patches(ADDR_A, cd, [("u256", 4, ret_u256(0))]).pop()
        combined = step1 + step2
        bytecode = combined.build()
        assert len(bytecode) > 0
        assert bytes(cd[:4]) in bytecode  # selector embedded via PUSH32 in push_bytes


# ---------------------------------------------------------------------------
# Arithmetic opcodes
# ---------------------------------------------------------------------------


class TestArithmeticOpcodes:
    """Verify bytecode emitted by arithmetic helpers and the Program builder."""

    def test_add_emitter(self):
        assert add() == bytes([OP_ADD])

    def test_mul_emitter(self):
        assert mul() == bytes([OP_MUL])

    def test_div_emitter(self):
        assert div() == bytes([OP_DIV])

    def test_mod_emitter(self):
        assert mod() == bytes([OP_MOD])

    def test_builder_add(self):
        assert Program().add().build() == add()

    def test_builder_mul(self):
        assert Program().mul().build() == mul()

    def test_builder_div(self):
        assert Program().div().build() == div()

    def test_builder_mod(self):
        assert Program().mod().build() == mod()

    def test_arithmetic_chain(self):
        """push(100) MUL push(60) DIV push(100) emits the correct byte sequence."""
        expected = push_u256(100) + push_u256(60) + mul() + push_u256(100) + div()
        actual = Program().push_u256(100).push_u256(60).mul().push_u256(100).div().build()
        assert actual == expected


# ---------------------------------------------------------------------------
# Integration: split-swap composition example
# ---------------------------------------------------------------------------


class TestSplitSwapComposition:
    """Verify the split-swap pattern builds valid bytecode via Program.compose."""

    # Fake calldata templates (4-byte selector + 64 bytes placeholders)
    _SWAP01 = bytes.fromhex("aabbccdd") + b"\x00" * 64
    _SWAP12 = bytes.fromhex("11223344") + b"\x00" * 64
    _SWAP13 = bytes.fromhex("55667788") + b"\x00" * 64

    AMOUNT_OFFSET = 4  # first ABI slot after 4-byte selector

    def _build_split_swap(self, numerator: int, denominator: int) -> bytes:
        """
        Build a split-swap program:
          1. Call SWAP01; store output in reg[0]
          2. Compute share0 = output * numerator / denominator → reg[1]
          3. Compute share1 = output - share0 → reg[2]
          4. Call SWAP12 with share0 from reg[1]
          5. Call SWAP13 with share1 from reg[2]
        """
        step1 = Program().call_with_patches(ADDR_A, self._SWAP01, []).pop().ret_u256(0).store_reg(0)

        split = (
            Program()
            .load_reg(0)
            .push_u256(numerator)
            .mul()
            .push_u256(denominator)
            .div()
            .store_reg(1)
            .load_reg(0)
            .load_reg(1)
            .sub()
            .store_reg(2)
        )

        step4 = (
            Program()
            .call_with_patches(
                ADDR_B,
                self._SWAP12,
                [("u256", self.AMOUNT_OFFSET, load_reg(1))],
            )
            .pop()
        )

        step5 = (
            Program()
            .call_with_patches(
                ADDR_B,
                self._SWAP13,
                [("u256", self.AMOUNT_OFFSET, load_reg(2))],
            )
            .pop()
        )

        return Program.compose([step1, split, step4, step5]).build()

    def test_split_swap_builds_without_error(self):
        bytecode = self._build_split_swap(60, 100)
        assert len(bytecode) > 0

    def test_split_swap_starts_with_push_bytes(self):
        bytecode = self._build_split_swap(60, 100)
        # push_bytes embeds calldata as PUSH32 immediates; selector bytes must appear
        assert bytes(self._SWAP01[:4]) in bytecode

    def test_split_swap_contains_mul_and_div(self):
        bytecode = self._build_split_swap(60, 100)
        assert OP_MUL in bytecode
        assert OP_DIV in bytecode

    def test_split_swap_contains_sub(self):
        bytecode = self._build_split_swap(60, 100)
        assert OP_SUB in bytecode

    def test_split_swap_50_50(self):
        """50/50 split also builds correctly."""
        bytecode = self._build_split_swap(50, 100)
        assert OP_MUL in bytecode
        assert OP_DIV in bytecode

    def test_split_swap_equals_manual_compose(self):
        """Program.compose([A, B]) == A + B for the split-swap sub-programs."""
        numerator, denominator = 60, 100

        step1 = Program().call_with_patches(ADDR_A, self._SWAP01, []).pop().ret_u256(0).store_reg(0)
        split = (
            Program()
            .load_reg(0)
            .push_u256(numerator)
            .mul()
            .push_u256(denominator)
            .div()
            .store_reg(1)
            .load_reg(0)
            .load_reg(1)
            .sub()
            .store_reg(2)
        )

        via_compose = Program.compose([step1, split]).build()
        via_add = (step1 + split).build()
        assert via_compose == via_add


# ---------------------------------------------------------------------------
# call_contract_abi and ContractFunction.from_abi integration
# ---------------------------------------------------------------------------


class TestEncodeCalldata:
    """Verify that ContractFunction.from_abi encodes calldata correctly."""

    def test_no_args_selector(self):
        """A zero-argument function yields only the 4-byte selector."""
        from eth_contract.contract import ContractFunction
        from eth_utils import keccak

        result = ContractFunction.from_abi("function totalSupply() view returns (uint256)")().data
        expected_selector = keccak(text="totalSupply()")[:4]
        assert result == expected_selector

    def test_transfer_selector_and_encoding(self):
        """transfer(address,uint256) matches the known ERC-20 selector."""
        from eth_contract.contract import ContractFunction
        from eth_contract.erc20 import ERC20

        calldata = ContractFunction.from_abi("function transfer(address,uint256)")(ADDR_A, 1000).data
        assert calldata[:4].hex() == "a9059cbb"
        assert calldata == ERC20.fns.transfer(ADDR_A, 1000).data

    def test_approve_selector_and_encoding(self):
        """approve(address,uint256) matches the known ERC-20 selector."""
        from eth_contract.contract import ContractFunction
        from eth_contract.erc20 import ERC20

        calldata = ContractFunction.from_abi(
            "function approve(address spender, uint256 amount) external returns (bool)"
        )(ADDR_B, 2**256 - 1).data
        assert calldata[:4].hex() == "095ea7b3"
        assert calldata == ERC20.fns.approve(ADDR_B, 2**256 - 1).data

    def test_function_keyword_optional(self):
        """Both bare and 'function'-prefixed signatures yield identical calldata."""
        from eth_contract.contract import ContractFunction

        bare = ContractFunction.from_abi("function transfer(address,uint256)")(ADDR_A, 42).data
        full = ContractFunction.from_abi("function transfer(address to, uint256 amount)")(ADDR_A, 42).data
        assert bare == full

    def test_param_names_optional(self):
        """Signatures with and without parameter names are equivalent."""
        from eth_contract.contract import ContractFunction

        with_names = ContractFunction.from_abi("function transfer(address to, uint256 amount)")(ADDR_A, 7).data
        without_names = ContractFunction.from_abi("function transfer(address,uint256)")(ADDR_A, 7).data
        assert with_names == without_names

    def test_tuple_type_encoding(self):
        """Tuple (struct) arguments are encoded correctly."""
        from eth_abi import encode
        from eth_contract.contract import ContractFunction
        from eth_utils import keccak

        sig = (
            "function exactInputSingle("
            "(address tokenIn, address tokenOut, uint24 fee, address recipient,"
            " uint256 deadline, uint256 amountIn, uint256 amountOutMinimum,"
            " uint160 sqrtPriceLimitX96) params"
            ")"
        )
        params = (ADDR_A, ADDR_B, 3000, ADDR_A, 9999, 10**18, 0, 0)
        calldata = ContractFunction.from_abi(sig)(params).data

        canonical = "exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))"
        expected_selector = keccak(text=canonical)[:4]
        assert calldata[:4] == expected_selector

        abi_type = "(address,address,uint24,address,uint256,uint256,uint256,uint160)"
        expected_payload = encode([abi_type], [params])
        assert calldata[4:] == expected_payload

    def test_result_starts_with_selector_length(self):
        """The result is at least 4 bytes (selector) for any function."""
        from eth_contract.contract import ContractFunction

        result = ContractFunction.from_abi("function fallback_()")().data
        assert len(result) == 4  # only selector, no args

    def test_uint256_arg(self):
        """A uint256 argument is encoded as a 32-byte big-endian word."""
        from eth_contract.contract import ContractFunction

        result = ContractFunction.from_abi("function foo(uint256 x)")(0xDEAD).data
        assert len(result) == 4 + 32
        assert int.from_bytes(result[4:], "big") == 0xDEAD


class TestCallContractAbi:
    """Verify that Program.call_contract_abi delegates correctly."""

    def test_produces_same_bytecode_as_call_contract(self):
        """call_contract_abi(to, sig, *args) == call_contract(to, ContractFunction.from_abi(sig)(*args).data)."""
        from eth_contract.contract import ContractFunction

        sig = "function transfer(address,uint256)"
        args = [ADDR_B, 500]

        via_abi = Program().call_contract_abi(ADDR_A, sig, *args).build()
        via_manual = Program().call_contract(ADDR_A, ContractFunction.from_abi(sig)(*args).data).build()
        assert via_abi == via_manual

    def test_selector_in_bytecode(self):
        """The ERC-20 transfer selector 0xa9059cbb appears inside the built bytecode."""
        bytecode = Program().call_contract_abi(ADDR_A, "transfer(address,uint256)", ADDR_B, 1000).pop().build()
        # The selector bytes should be somewhere inside the push_bytes payload
        assert bytes.fromhex("a9059cbb") in bytecode

    def test_chaining(self):
        """call_contract_abi returns self so it can be chained."""
        p = Program()
        result = p.call_contract_abi(ADDR_A, "transfer(address,uint256)", ADDR_B, 1)
        assert result is p

    def test_no_args_function(self):
        """call_contract_abi works for a zero-argument function."""
        bytecode = Program().call_contract_abi(ADDR_A, "function totalSupply() view returns (uint256)").pop().build()
        assert len(bytecode) > 0

    def test_value_and_gas_forwarded(self):
        """ETH value and gas limit are forwarded to the underlying call_contract."""
        from eth_contract.contract import ContractFunction

        sig = "function transfer(address,uint256)"
        args = [ADDR_B, 42]

        via_abi = Program().call_contract_abi(ADDR_A, sig, *args, value=100, gas=50000).build()
        via_manual = (
            Program().call_contract(ADDR_A, ContractFunction.from_abi(sig)(*args).data, value=100, gas=50000).build()
        )
        assert via_abi == via_manual

    def test_require_success_false(self):
        """require_success=False is forwarded correctly."""
        from eth_contract.contract import ContractFunction

        sig = "function transfer(address,uint256)"
        args = [ADDR_B, 1]

        via_abi = Program().call_contract_abi(ADDR_A, sig, *args, require_success=False).build()
        via_manual = (
            Program().call_contract(ADDR_A, ContractFunction.from_abi(sig)(*args).data, require_success=False).build()
        )
        assert via_abi == via_manual

    def test_compose_with_other_programs(self):
        """call_contract_abi works correctly when composed with other sub-programs."""
        sig = "transfer(address,uint256)"
        step1 = Program().call_contract_abi(ADDR_A, sig, ADDR_B, 100).pop()
        step2 = Program().call_contract_abi(ADDR_B, sig, ADDR_A, 50).pop()
        combined = Program.compose([step1, step2])
        bytecode = combined.build()
        assert len(bytecode) > 0
        # ERC-20 transfer selector appears at least twice (once per call)
        assert bytecode.count(bytes.fromhex("a9059cbb")) >= 2


# ---------------------------------------------------------------------------
# EVM-native opcodes: comparison, bitwise, and shift
# ---------------------------------------------------------------------------


class TestEvmNativeOpcodes:
    """Verify bytecode emitted by EVM-native opcode helpers and the Program builder."""

    # -- Emitter correctness ---------------------------------------------------

    def test_lt_emitter(self):
        assert lt() == bytes([OP_LT])

    def test_gt_emitter(self):
        assert gt() == bytes([OP_GT])

    def test_eq_emitter(self):
        assert eq() == bytes([OP_EQ])

    def test_iszero_emitter(self):
        assert iszero() == bytes([OP_ISZERO])

    def test_and_emitter(self):
        assert bitwise_and() == bytes([OP_AND])

    def test_or_emitter(self):
        assert bitwise_or() == bytes([OP_OR])

    def test_xor_emitter(self):
        assert bitwise_xor() == bytes([OP_XOR])

    def test_not_emitter(self):
        assert bitwise_not() == bytes([OP_NOT])

    def test_shl_emitter(self):
        assert shl() == bytes([OP_SHL])

    def test_shr_emitter(self):
        assert shr() == bytes([OP_SHR])

    # -- Program builder methods -----------------------------------------------

    def test_builder_lt(self):
        assert Program().lt().build() == lt()

    def test_builder_gt(self):
        assert Program().gt().build() == gt()

    def test_builder_eq(self):
        assert Program().eq().build() == eq()

    def test_builder_iszero(self):
        assert Program().iszero().build() == iszero()

    def test_builder_bitwise_and(self):
        assert Program().bitwise_and().build() == bitwise_and()

    def test_builder_bitwise_or(self):
        assert Program().bitwise_or().build() == bitwise_or()

    def test_builder_bitwise_xor(self):
        assert Program().bitwise_xor().build() == bitwise_xor()

    def test_builder_bitwise_not(self):
        assert Program().bitwise_not().build() == bitwise_not()

    def test_builder_shl(self):
        assert Program().shl().build() == shl()

    def test_builder_shr(self):
        assert Program().shr().build() == shr()

    # -- Opcode constant values ------------------------------------------------

    def test_opcode_values(self):
        # Native EVM opcode values
        assert OP_LT == 0x10
        assert OP_GT == 0x11
        assert OP_EQ == 0x14
        assert OP_ISZERO == 0x15
        assert OP_AND == 0x16
        assert OP_OR == 0x17
        assert OP_XOR == 0x18
        assert OP_NOT == 0x19
        assert OP_SHL == 0x1B
        assert OP_SHR == 0x1C

    # -- Bytecode composition --------------------------------------------------

    def test_comparison_chain(self):
        """push(5) push(10) LT emits the correct byte sequence."""
        expected = push_u256(5) + push_u256(10) + lt()
        actual = Program().push_u256(5).push_u256(10).lt().build()
        assert actual == expected

    def test_bitwise_chain(self):
        """push(0xFF) push(0x0F) AND XOR emits the correct byte sequence."""
        expected = push_u256(0xFF) + push_u256(0x0F) + bitwise_and() + push_u256(0xAA) + bitwise_xor()
        actual = Program().push_u256(0xFF).push_u256(0x0F).bitwise_and().push_u256(0xAA).bitwise_xor().build()
        assert actual == expected

    def test_shift_chain(self):
        """push(1) push(8) SHL emits the correct byte sequence (shift value left by 8)."""
        expected = push_u256(1) + push_u256(8) + shl()
        actual = Program().push_u256(1).push_u256(8).shl().build()
        assert actual == expected

    def test_iszero_after_eq(self):
        """EQ followed by ISZERO implements != comparison."""
        expected = push_u256(5) + push_u256(5) + eq() + iszero()
        actual = Program().push_u256(5).push_u256(5).eq().iszero().build()
        assert actual == expected
