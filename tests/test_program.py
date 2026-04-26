"""Tests for :class:`pydefi.vm.Program` (SSA-style program builder)."""

from __future__ import annotations

import pytest
from hexbytes import HexBytes

from pydefi.types import BasePool, RouteDAG, SwapProtocol, Token
from pydefi.vm import (
    STDLIB,
    Placeholder,
    Program,
    Value,
    build_execution_program_for_dag,
    build_quote_program_for_dag,
)
from tests.conftest import mini_evm

# Common test address — mini_evm makes any CALL succeed.
_TARGET = HexBytes("0x" + "aa" * 20)


def _run_int(prog: Program, **build_kwargs) -> int:
    """Build *prog* and execute via mini_evm; assert success and return TOS as int."""
    bytecode = prog.build(**build_kwargs)
    result = mini_evm(bytecode)
    assert not result.is_error, f"unexpected revert: {result.output.hex()}"
    assert len(result.output) == 32, f"expected 32-byte word, got {len(result.output)}"
    return int.from_bytes(result.output, "big")


def _run_assert_revert(prog: Program, **build_kwargs) -> bytes:
    """Build *prog* and execute via mini_evm; assert revert and return raw output."""
    bytecode = prog.build(**build_kwargs)
    result = mini_evm(bytecode)
    assert result.is_error, "expected revert, got success"
    return result.output


# ---------------------------------------------------------------------------
# Constants & arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_const_returns_value(self):
        v = Program().const(42)
        assert isinstance(v, Value)

    def test_add(self):
        p = Program()
        p.return_word(p.add(p.const(7), p.const(5)))
        assert _run_int(p) == 12

    def test_sub_normal(self):
        p = Program()
        p.return_word(p.sub(p.const(10), p.const(3)))
        assert _run_int(p) == 7

    def test_sub_saturates_on_underflow(self):
        p = Program()
        p.return_word(p.sub(p.const(3), p.const(10)))
        assert _run_int(p) == 0

    def test_mul(self):
        p = Program()
        p.return_word(p.mul(p.const(6), p.const(7)))
        assert _run_int(p) == 42

    def test_div(self):
        p = Program()
        p.return_word(p.div(p.const(100), p.const(7)))
        assert _run_int(p) == 14

    def test_div_by_zero_returns_zero(self):
        p = Program()
        p.return_word(p.div(p.const(100), p.const(0)))
        assert _run_int(p) == 0

    def test_mod(self):
        p = Program()
        p.return_word(p.mod(p.const(100), p.const(7)))
        assert _run_int(p) == 2

    def test_const_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Program().const(-1)

    def test_const_overflow_rejected(self):
        with pytest.raises(ValueError, match="uint256"):
            Program().const(1 << 256)


# ---------------------------------------------------------------------------
# Comparison & boolean
# ---------------------------------------------------------------------------


class TestComparison:
    def test_lt_true(self):
        p = Program()
        p.return_word(p.lt(p.const(3), p.const(10)))
        assert _run_int(p) == 1

    def test_lt_false(self):
        p = Program()
        p.return_word(p.lt(p.const(10), p.const(3)))
        assert _run_int(p) == 0

    def test_gt(self):
        p = Program()
        p.return_word(p.gt(p.const(10), p.const(3)))
        assert _run_int(p) == 1

    def test_eq(self):
        p = Program()
        p.return_word(p.eq(p.const(7), p.const(7)))
        assert _run_int(p) == 1

    def test_is_zero(self):
        p = Program()
        p.return_word(p.is_zero(p.const(0)))
        assert _run_int(p) == 1


# ---------------------------------------------------------------------------
# Bitwise
# ---------------------------------------------------------------------------


class TestBitwise:
    def test_bit_and(self):
        p = Program()
        p.return_word(p.bit_and(p.const(0b1100), p.const(0b1010)))
        assert _run_int(p) == 0b1000

    def test_bit_or(self):
        p = Program()
        p.return_word(p.bit_or(p.const(0b1100), p.const(0b1010)))
        assert _run_int(p) == 0b1110

    def test_bit_xor(self):
        p = Program()
        p.return_word(p.bit_xor(p.const(0b1100), p.const(0b1010)))
        assert _run_int(p) == 0b0110

    def test_shl(self):
        p = Program()
        p.return_word(p.shl(p.const(1), p.const(8)))  # 1 << 8
        assert _run_int(p) == 256

    def test_shr(self):
        p = Program()
        p.return_word(p.shr(p.const(256), p.const(8)))  # 256 >> 8
        assert _run_int(p) == 1


# ---------------------------------------------------------------------------
# Memory slots
# ---------------------------------------------------------------------------


class TestMemorySlots:
    def test_store_load_roundtrip(self):
        p = Program()
        s = p.alloc_slot()
        p.store_slot(s, p.const(42))
        p.return_word(p.load_slot(s))
        assert _run_int(p, disable_constant_folding=True) == 42

    def test_multiple_slots(self):
        p = Program()
        a = p.alloc_slot()
        b = p.alloc_slot()
        p.store_slot(a, p.const(11))
        p.store_slot(b, p.const(22))
        p.return_word(p.add(p.load_slot(a), p.load_slot(b)))
        assert _run_int(p, disable_constant_folding=True) == 33


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


class TestAssert:
    def test_assert_pass(self):
        p = Program()
        p.assert_(p.const(1))
        p.return_word(p.const(99))
        assert _run_int(p) == 99

    def test_assert_revert_no_msg(self):
        p = Program()
        p.assert_(p.const(0))
        p.stop()
        out = _run_assert_revert(p, disable_constant_folding=True)
        assert out == b""

    def test_assert_revert_with_msg(self):
        p = Program()
        p.assert_(p.const(0), "oops")
        p.stop()
        out = _run_assert_revert(p, disable_constant_folding=True)
        # Solidity Error(string) ABI: selector(4) + offset(32) + length(32) + msg
        assert out[:4] == bytes.fromhex("08c379a0")
        length = int.from_bytes(out[36:68], "big")
        assert out[68 : 68 + length] == b"oops"

    def test_assert_message_too_long_rejected(self):
        p = Program()
        with pytest.raises(ValueError, match="too long"):
            p.assert_(p.const(0), "x" * 33)

    def test_assert_ge_pass(self):
        p = Program()
        p.assert_ge(p.const(10), p.const(3))
        p.return_word(p.const(1))
        assert _run_int(p) == 1

    def test_assert_ge_fail(self):
        p = Program()
        p.assert_ge(p.const(3), p.const(10), "below min")
        p.stop()
        out = _run_assert_revert(p, disable_constant_folding=True)
        length = int.from_bytes(out[36:68], "big")
        assert out[68 : 68 + length] == b"below min"

    def test_assert_le_pass(self):
        p = Program()
        p.assert_le(p.const(3), p.const(10))
        p.return_word(p.const(1))
        assert _run_int(p) == 1

    def test_assert_le_fail(self):
        p = Program()
        p.assert_le(p.const(10), p.const(3))
        p.stop()
        _ = _run_assert_revert(p, disable_constant_folding=True)

    def test_assert_le_fail_with_msg(self):
        """assert_le(a, b, msg) must revert with the Error(string) payload."""
        p = Program()
        p.assert_le(p.const(10), p.const(3), "above max")
        p.stop()
        out = _run_assert_revert(p, disable_constant_folding=True)
        assert out[:4] == bytes.fromhex("08c379a0")
        length = int.from_bytes(out[36:68], "big")
        assert out[68 : 68 + length] == b"above max"

    def test_stdlib_singleton_survives_repeated_builds(self):
        """ModuleBuilder.merge clones — STDLIB stays intact across many Programs."""
        before = sorted(fn.name.value for fn in STDLIB.ctx.functions.values())
        for _ in range(3):
            p = Program()
            p.assert_(p.const(1), "x")
            p.stop()
            p.build()
        after = sorted(fn.name.value for fn in STDLIB.ctx.functions.values())
        assert before == after

    def test_codesize_optimization_shares_revert_helper(self):
        """Under Os, per-extra-messaged-assert cost should be cheaper than under
        O2 — FunctionInlinerPass keeps stdlib_revert_if shared at codesize-opt."""
        from vyper.compiler.settings import OptimizationLevel

        def size(n: int, opt: OptimizationLevel) -> int:
            p = Program()
            cond = p.stack_param()  # runtime value — SCCP can't elide
            for _ in range(n):
                p.assert_(cond, "msg")
            p.return_word(p.const(7))
            return len(p.build(optimize=opt))

        gas_growth = size(8, OptimizationLevel.GAS) - size(2, OptimizationLevel.GAS)
        os_growth = size(8, OptimizationLevel.CODESIZE) - size(2, OptimizationLevel.CODESIZE)
        assert os_growth < gas_growth, f"expected Os ({os_growth}b) < O2 ({gas_growth}b)"


# ---------------------------------------------------------------------------
# External calls
# ---------------------------------------------------------------------------


class TestCallContract:
    def test_call_returns_success_flag(self):
        p = Program()
        success = p.call_contract(_TARGET, b"\x12\x34\x56\x78")
        p.return_word(success)
        # mini_evm makes all calls succeed.
        assert _run_int(p) == 1

    def test_call_with_explicit_gas(self):
        p = Program()
        # value=0 since mini_evm contract has no ETH balance to forward.
        success = p.call_contract(_TARGET, b"\x12\x34", gas=50000)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_call_with_patches(self):
        p = Program()
        template = b"\x12\x34\x56\x78" + b"\x00" * 32
        success = p.call_contract(_TARGET, template, patches={4: p.const(0xCAFEBABE)})
        p.return_word(success)
        assert _run_int(p) == 1

    def test_call_patch_offset_out_of_bounds(self):
        p = Program()
        with pytest.raises(ValueError, match="out of bounds"):
            p.call_contract(_TARGET, b"\x12\x34", patches={100: p.const(0)})


# ---------------------------------------------------------------------------
# call_contract_abi (ABI-encoded calldata builder)
# ---------------------------------------------------------------------------


class TestCallContractAbi:
    def test_static_args_no_patches(self):
        p = Program()
        success = p.call_contract_abi(_TARGET, "transfer(address,uint256)", _TARGET, 10**18)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_function_keyword_optional(self):
        p = Program()
        s1 = p.call_contract_abi(_TARGET, "transfer(address,uint256)", _TARGET, 1)
        p.assert_(s1)
        s2 = p.call_contract_abi(_TARGET, "function transfer(address,uint256)", _TARGET, 2)
        p.return_word(s2)
        assert _run_int(p) == 1

    def test_no_args(self):
        p = Program()
        success = p.call_contract_abi(_TARGET, "ping()")
        p.return_word(success)
        assert _run_int(p) == 1

    def test_value_placeholder_uint256(self):
        p = Program()
        amount = p.const(0xDEADBEEF)
        success = p.call_contract_abi(_TARGET, "set(uint256)", Placeholder(amount))
        p.return_word(success)
        assert _run_int(p) == 1

    def test_value_placeholder_address(self):
        p = Program()
        addr = p.addr(_TARGET)
        success = p.call_contract_abi(_TARGET, "ping(address)", Placeholder(addr))
        p.return_word(success)
        assert _run_int(p) == 1

    def test_mixed_static_and_value(self):
        p = Program()
        amount = p.const(7)
        success = p.call_contract_abi(_TARGET, "transfer(address,uint256)", _TARGET, Placeholder(amount))
        p.return_word(success)
        assert _run_int(p) == 1

    def test_arity_mismatch_rejected(self):
        p = Program()
        with pytest.raises(ValueError, match="expected 2"):
            p.call_contract_abi(_TARGET, "transfer(address,uint256)", _TARGET)


# ---------------------------------------------------------------------------
# DAG / swap composer (SSA versions)
# ---------------------------------------------------------------------------


def _mk_v2_dag() -> RouteDAG:
    token_in = Token(chain_id=1, address=HexBytes("0x" + "11" * 20), symbol="TKA", decimals=18)
    token_out = Token(chain_id=1, address=HexBytes("0x" + "22" * 20), symbol="TKB", decimals=18)

    class _DummyPool(BasePool):
        protocol = SwapProtocol.UNISWAP_V2
        pool_address = HexBytes("0x" + "33" * 20)
        fee_bps = 30

        def __init__(self) -> None:
            self.token_in = token_in
            self.token_out = token_out

        def zero_for_one(self, _token_out: HexBytes) -> bool:
            return True

    return RouteDAG().from_token(token_in).swap(token_out, _DummyPool())


class TestSsaDagBuilder:
    def test_execution_program_builds(self):
        prog = build_execution_program_for_dag(
            _mk_v2_dag(),
            amount_in=10**18,
            vm_address="0x" + "44" * 20,
            recipient="0x" + "55" * 20,
        )
        assert isinstance(prog, Program)
        bytecode = prog.build()
        assert len(bytecode) > 0

    def test_quote_program_builds(self):
        prog = build_quote_program_for_dag(_mk_v2_dag(), amount_in=10**18)
        assert isinstance(prog, Program)
        bytecode = prog.build()
        assert len(bytecode) > 0


# ---------------------------------------------------------------------------
# Returndata access
# ---------------------------------------------------------------------------


class TestReturndata:
    def test_returndata_word_compiles(self):
        # mini_evm has empty returndata so RETURNDATACOPY would revert;
        # only verify the IR builds without error.
        p = Program()
        success = p.call_contract(_TARGET, b"\x12\x34")
        p.assert_(success)
        word = p.returndata_word(0)
        p.return_word(word)
        bytecode = p.build()
        assert len(bytecode) > 0


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class TestTermination:
    def test_implicit_stop(self):
        # Build with no explicit terminator — should auto-stop.
        p = Program()
        s = p.alloc_slot()
        p.store_slot(s, p.const(99))  # produces no stack effect
        bytecode = p.build()
        result = mini_evm(bytecode)
        assert not result.is_error
        assert result.output == b""

    def test_explicit_stop(self):
        p = Program()
        p.stop()
        result = mini_evm(p.build())
        assert not result.is_error

    def test_revert_no_msg(self):
        p = Program()
        p.revert()
        result = mini_evm(p.build())
        assert result.is_error


# ---------------------------------------------------------------------------
# Operand coercion
# ---------------------------------------------------------------------------


class TestOperandCoercion:
    def test_int_accepted_directly(self):
        p = Program()
        p.return_word(p.add(7, 5))  # both ints
        assert _run_int(p) == 12

    def test_address_bytes_accepted(self):
        p = Program()
        # 20-byte address used in a value position — interpreted as uint256.
        p.return_word(p.bit_and(_TARGET, p.const((1 << 8) - 1)))  # low byte
        assert _run_int(p) == 0xAA

    def test_wrong_length_bytes_rejected(self):
        p = Program()
        with pytest.raises(ValueError, match="20"):
            p.add(b"\xff", 0)

    def test_negative_int_rejected(self):
        p = Program()
        with pytest.raises(ValueError, match="non-negative"):
            p.add(-1, 0)


# ---------------------------------------------------------------------------
# SSA optimization quality
# ---------------------------------------------------------------------------


class TestOptimization:
    def test_constant_fold_at_compile_time(self):
        """Two-constant arithmetic should fold at compile time."""
        p = Program()
        p.return_word(p.add(p.const(7), p.const(5)))
        bytecode = p.build()
        # Optimized output: PUSH1 12, PUSH0, MSTORE, PUSH1 32, PUSH0, RETURN = 8 bytes.
        assert len(bytecode) <= 16, f"expected ≤ 16 bytes, got {len(bytecode)}"

    def test_no_dup_swap_pop_in_simple_flow(self):
        """SSA → linear dataflow shouldn't need stack juggling for a simple add."""
        p = Program()
        p.return_word(p.add(p.const(7), p.const(5)))
        bytecode = p.build()
        # No DUP (0x80..0x8F), SWAP (0x90..0x9F), or POP (0x50).
        for byte in bytecode:
            assert not (0x80 <= byte <= 0x9F), f"unexpected DUP/SWAP byte 0x{byte:02x}"
            assert byte != 0x50, f"unexpected POP byte 0x{byte:02x}"


# ---------------------------------------------------------------------------
# Program.build(prefix_length=) — composer-prologue compatibility
# ---------------------------------------------------------------------------


def _build_codecopy_program() -> Program:
    """Dense calldata forces the CODECOPY strategy in call_contract —
    exercises label-resolved PUSH_OFST for the data-section offset."""
    dense = bytes((i * 17 + 31) & 0xFF for i in range(612))
    p = Program()
    p.return_word(p.call_contract(_TARGET, dense))
    return p


def _build_assert_program() -> Program:
    """assert_(cond, msg) emits jnz + ok/revert basic blocks — exercises
    PUSHLABEL immediates for the branch targets."""
    p = Program()
    s = p.alloc_slot()
    p.store_slot(s, 1)
    p.assert_(p.load_slot(s), "should not revert")
    p.return_word(99)
    return p


# Prefix shapes that leave an empty stack by the time the user program runs.
_PREFIX_PUSH1_POP = bytes([0x60, 0x42, 0x50])  # PUSH1 0x42; POP — net stack effect = 0
_PREFIX_CCTP_SHAPE = (
    bytes([0x7F])
    + bytes(32)  # PUSH32 0 (mimics amountReceived)
    + bytes([0x7F])
    + bytes(32)  # PUSH32 0 (mimics sourceDomain)
    + bytes([0x60, 0x80, 0x52, 0x60, 0xA0, 0x52])  # drain both into scratch mem
)


class TestBuildPrefixLength:
    """``prefix_length`` shifts label-resolved PUSH_OFST / PUSHLABEL immediates so
    a program embedded behind a runtime stack-push prologue still executes
    correctly.  The parametrized matrix checks every combination of label
    category (CODECOPY data-section, assert jnz/label targets) against every
    prefix shape we care about."""

    @pytest.mark.parametrize(
        "build_prog",
        [_build_codecopy_program, _build_assert_program],
        ids=["codecopy_data_section", "assert_jnz_branches"],
    )
    @pytest.mark.parametrize(
        "prefix",
        [_PREFIX_PUSH1_POP, _PREFIX_CCTP_SHAPE],
        ids=["push1_pop", "cctp_shape"],
    )
    def test_output_matches_after_shift(self, build_prog, prefix):
        ref_out = mini_evm(build_prog().build()).output
        shifted = prefix + build_prog().build(prefix_length=len(prefix))
        shifted_out = mini_evm(shifted).output
        assert shifted_out == ref_out, f"ref={ref_out.hex()} shifted={shifted_out.hex()}"

    def test_prefix_length_zero_is_noop(self):
        assert _build_codecopy_program().build() == _build_codecopy_program().build(prefix_length=0)
