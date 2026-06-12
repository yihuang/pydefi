"""Tests for :class:`pydefi.vm.Program` (SSA-style program builder)."""

from __future__ import annotations

import pytest
from eth_contract.contract import ContractFunction
from hexbytes import HexBytes

from pydefi.bridge.eureka import encode_send_transfer_calldata
from pydefi.pathfinder.dag import RouteBridge, RouteDAG
from pydefi.pathfinder.graph import BasePool
from pydefi.types import Address, ChainId, SwapProtocol, Token
from pydefi.vm import (
    IIBC_SENDER_CALLBACKS_INTERFACE_ID,
    Operand,
    Program,
    approve_then_send_transfer,
    build_execution_program_for_dag,
    build_quote_program_for_dag,
    encode_send_and_compose_calldata,
    send_transfer,
)
from tests.conftest import mini_evm

_TRANSFER_FN = ContractFunction.from_abi("function transfer(address,uint256)")
_PING_FN = ContractFunction.from_abi("function ping()")
_PING_ADDR_FN = ContractFunction.from_abi("function ping(address)")
_SET_UINT_FN = ContractFunction.from_abi("function set(uint256)")

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
        assert isinstance(v, Operand)

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
        p.builder.stop()
        out = _run_assert_revert(p, disable_constant_folding=True)
        assert out == b""

    def test_assert_revert_with_msg(self):
        p = Program()
        p.assert_(p.const(0), "oops")
        p.builder.stop()
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
        p.builder.stop()
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
        p.builder.stop()
        _ = _run_assert_revert(p, disable_constant_folding=True)

    def test_assert_le_fail_with_msg(self):
        """assert_le(a, b, msg) must revert with the Error(string) payload."""
        p = Program()
        p.assert_le(p.const(10), p.const(3), "above max")
        p.builder.stop()
        out = _run_assert_revert(p, disable_constant_folding=True)
        assert out[:4] == bytes.fromhex("08c379a0")
        length = int.from_bytes(out[36:68], "big")
        assert out[68 : 68 + length] == b"above max"

    def test_codesize_optimization_shares_revert_helper(self):
        """Under Os, per-extra-messaged-assert cost should be cheaper than under
        O2 — FunctionInlinerPass keeps stdlib_revert_if shared at codesize-opt."""
        from vyper.compiler.settings import OptimizationLevel

        def size(n: int, opt: OptimizationLevel) -> int:
            p = Program()
            cond = p.builder.param()  # runtime value — SCCP can't elide
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
        success = p.call_raw(_TARGET, b"\x12\x34\x56\x78")
        p.return_word(success)
        # mini_evm makes all calls succeed.
        assert _run_int(p) == 1

    def test_call_with_explicit_gas(self):
        p = Program()
        # value=0 since mini_evm contract has no ETH balance to forward.
        success = p.call_raw(_TARGET, b"\x12\x34", gas=50000)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_call_with_patches(self):
        p = Program()
        template = b"\x12\x34\x56\x78" + b"\x00" * 32
        success = p.call_raw(_TARGET, template, patches={4: p.const(0xCAFEBABE)})
        p.return_word(success)
        assert _run_int(p) == 1

    def test_call_patch_offset_out_of_bounds(self):
        p = Program()
        with pytest.raises(ValueError, match="out of bounds"):
            p.call_raw(_TARGET, b"\x12\x34", patches={100: p.const(0)})


# ---------------------------------------------------------------------------
# call_contract (ABI-encoded calldata builder)
# ---------------------------------------------------------------------------


class TestCallContractAbi:
    def test_static_args(self):
        p = Program()
        success = p.call_contract(_TARGET, _TRANSFER_FN, _TARGET, 10**18)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_no_args(self):
        p = Program()
        success = p.call_contract(_TARGET, _PING_FN)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_runtime_uint256_arg(self):
        p = Program()
        amount = p.const(0xDEADBEEF)
        success = p.call_contract(_TARGET, _SET_UINT_FN, amount)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_runtime_address_arg(self):
        p = Program()
        addr = p.addr(_TARGET)
        success = p.call_contract(_TARGET, _PING_ADDR_FN, addr)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_mixed_static_and_runtime(self):
        p = Program()
        amount = p.const(7)
        success = p.call_contract(_TARGET, _TRANSFER_FN, _TARGET, amount)
        p.return_word(success)
        assert _run_int(p) == 1

    def test_arity_mismatch_rejected(self):
        p = Program()
        with pytest.raises(ValueError, match="expected 2"):
            p.call_contract(_TARGET, _TRANSFER_FN, _TARGET)


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
# RouteBridge — Eureka edge in a RouteDAG
# ---------------------------------------------------------------------------


_BRIDGE_SRC = Token(chain_id=1, address=HexBytes("0x" + "11" * 20), symbol="USDC", decimals=6)
_BRIDGE_MID = Token(chain_id=1, address=HexBytes("0x" + "22" * 20), symbol="WETH", decimals=18)
_BRIDGE_DST = Token(chain_id=ChainId.MANTRA, address=HexBytes("0x" + "33" * 20), symbol="USDC", decimals=6)
_BRIDGE_ICS20 = Address(HexBytes("0x" + "AA" * 20))


class _V2DummyPool(BasePool):
    """Minimal V2 pool for the swap-then-bridge case (always zero-for-one)."""

    protocol = SwapProtocol.UNISWAP_V2
    pool_address = HexBytes("0x" + "44" * 20)
    fee_bps = 30

    def __init__(self, token_in: Token, token_out: Token) -> None:
        self.token_in = token_in
        self.token_out = token_out

    def zero_for_one(self, _t: HexBytes) -> bool:
        return True


def _bridge_dag(*, with_upstream_swap: bool = False) -> RouteDAG:
    dag = RouteDAG().from_token(_BRIDGE_SRC)
    if with_upstream_swap:
        dag = dag.swap(_BRIDGE_MID, _V2DummyPool(_BRIDGE_SRC, _BRIDGE_MID))
    return dag.bridge(
        _BRIDGE_DST,
        transfer_addr=_BRIDGE_ICS20,
        source_client="07-tendermint-0",
        receiver="mantra1qq",
        timeout_seconds=600,
    )


_ROUTE_BRIDGE_KWARGS = dict(
    denom=Address(HexBytes("0x" + "11" * 20)),
    token_out=_BRIDGE_DST,
    transfer_addr=_BRIDGE_ICS20,
    source_client="07-tendermint-0",
    receiver="mantra1qq",
)


class TestRouteBridge:
    def test_builder_infers_denom_from_running_token(self):
        dag = (
            RouteDAG()
            .from_token(_BRIDGE_SRC)
            .bridge(
                _BRIDGE_DST,
                transfer_addr=_BRIDGE_ICS20,
                source_client="07-tendermint-0",
                receiver="mantra1qq",
                timeout_seconds=600,
            )
        )
        assert len(dag.actions) == 1
        action = dag.actions[0]
        assert isinstance(action, RouteBridge)
        assert action.denom == _BRIDGE_SRC.address
        assert action.token_out == _BRIDGE_DST

    @pytest.mark.parametrize(
        "action",
        ["swap", "bridge", "split"],
    )
    def test_builder_rejects_action_after_bridge(self, action):
        dag = _bridge_dag()
        with pytest.raises(ValueError, match="cannot follow .bridge"):
            if action == "swap":
                dag.swap(
                    Token(chain_id=1, address=HexBytes("0x" + "44" * 20), symbol="X", decimals=18),
                    pool=_V2DummyPool(_BRIDGE_DST, _BRIDGE_DST),
                )
            elif action == "bridge":
                dag.bridge(
                    _BRIDGE_DST,
                    transfer_addr=_BRIDGE_ICS20,
                    source_client="c",
                    receiver="r",
                    timeout_seconds=1,
                )
            else:
                dag.split()

    @pytest.mark.parametrize(
        "extra, ok",
        [
            ({}, False),
            ({"timeout_seconds": 10, "timeout_timestamp": 1}, False),
            ({"timeout_seconds": 10}, True),
            ({"timeout_timestamp": 1}, True),
        ],
    )
    def test_route_bridge_requires_exactly_one_timeout(self, extra, ok):
        if ok:
            RouteBridge(**_ROUTE_BRIDGE_KWARGS, **extra)
        else:
            with pytest.raises(ValueError, match="exactly one of"):
                RouteBridge(**_ROUTE_BRIDGE_KWARGS, **extra)

    @pytest.mark.parametrize("with_upstream_swap", [False, True])
    def test_execution_program_builds(self, with_upstream_swap):
        prog = build_execution_program_for_dag(
            _bridge_dag(with_upstream_swap=with_upstream_swap),
            amount_in=10**6,
            vm_address="0x" + "44" * 20,
            recipient="0x" + "55" * 20,
        )
        assert isinstance(prog, Program)
        assert len(prog.build()) > 100  # approve + sendTransfer + assert at minimum

    def test_quote_program_passes_amount_through_bridge(self):
        # ICS-20 is amount-preserving on the source; the quote walker must
        # thread ``amount_in`` through unchanged. We just verify the program
        # compiles — quote-time eval of the new action variant doesn't choke.
        assert len(build_quote_program_for_dag(_bridge_dag(), amount_in=10**6).build()) > 0

    def test_to_dict_preserves_bridge_action(self):
        dag = _bridge_dag()
        out = dag.to_dict()
        assert len(out["actions"]) == 1
        assert isinstance(out["actions"][0], RouteBridge)


# ---------------------------------------------------------------------------
# Returndata access
# ---------------------------------------------------------------------------


class TestReturndata:
    def test_returndata_word_compiles(self):
        # mini_evm has empty returndata so RETURNDATACOPY would revert;
        # only verify the IR builds without error.
        p = Program()
        success = p.call_raw(_TARGET, b"\x12\x34")
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
        p.builder.stop()
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
# Eureka (IBC v2) helpers — send_transfer / approve_then_send_transfer
# end-to-end through mini_evm.
# ---------------------------------------------------------------------------

_EUREKA_TRANSFER = Address(b"\xaa" * 20)  # mini_evm makes any CALL to this succeed
_EUREKA_DENOM = Address(b"\x22" * 20)
_SEND_KW = dict(
    transfer_addr=_EUREKA_TRANSFER,
    denom=_EUREKA_DENOM,
    receiver="mantra1qq...",
    source_client="07-tendermint-0",
)


class TestSendTransfer:
    @pytest.mark.parametrize(
        "timeout_kw",
        [{"timeout_timestamp": 1_700_000_000}, {"timeout_seconds": 600}],
        ids=["absolute", "relative"],
    )
    def test_send_transfer_runs(self, timeout_kw):
        # mini_evm makes any CALL succeed; we just verify the program builds
        # and returns success regardless of which timeout flavour was used.
        p = Program()
        ok = send_transfer(p, amount=10**6, **_SEND_KW, **timeout_kw)
        p.return_word(ok)
        assert _run_int(p) == 1

    def test_runtime_amount_runs_through_call_raw_patches(self):
        # Runtime amount goes through ``call_raw`` patches; the canary must
        # land in the bytecode (Venom may pick a narrower PUSHn so we look
        # for its minimum byte representation, not a fixed 32-byte form).
        canary = 0xDEADBEEF_CAFEBABE_FEEDFACE_BADC0DEE
        p = Program()
        ok = send_transfer(p, amount=p.const(canary), timeout_timestamp=1, **_SEND_KW)
        p.return_word(ok)
        bytecode = p.build()
        assert _run_int(p) == 1
        min_width = (canary.bit_length() + 7) // 8
        assert canary.to_bytes(min_width, "big") in bytecode

    def test_approve_then_send_transfer_compiles_and_runs(self):
        p = Program()
        ok = approve_then_send_transfer(p, amount=p.const(10**6), timeout_seconds=600, **_SEND_KW)
        p.return_word(ok)
        assert _run_int(p) == 1

    @pytest.mark.parametrize(
        "extra",
        [{}, {"timeout_seconds": 10, "timeout_timestamp": 1}],
        ids=["neither", "both"],
    )
    def test_timeout_arg_validation(self, extra):
        with pytest.raises(ValueError, match="exactly one of"):
            send_transfer(Program(), amount=1, **_SEND_KW, **extra)

    def test_static_calldata_matches_off_chain_encoder(self):
        # The bytes embedded in the program's data section must match what
        # :class:`pydefi.bridge.Eureka` produces off-chain — locks the two
        # encoders in step.
        kwargs = dict(
            denom=_EUREKA_DENOM,
            amount=10**6,
            receiver="mantra1qq...",
            source_client="07-tendermint-0",
            timeout_timestamp=1_700_000_000,
            memo="hi",
        )
        expected = encode_send_transfer_calldata(**kwargs)
        p = Program()
        send_transfer(p, transfer_addr=_EUREKA_TRANSFER, **kwargs)
        p.builder.stop()
        assert expected in p.build()

    def test_patch_offsets_match_abi_layout(self):
        """``_AMOUNT_OFFSET`` and ``_TIMEOUT_OFFSET`` must point at the actual
        ``amount`` / ``timeoutTimestamp`` slots in the encoded calldata.
        Without this, a runtime amount or timeout could land at the wrong
        offset and the test that just looks for canary bytes in the bytecode
        would still pass.
        """
        from pydefi.vm.eureka import _AMOUNT_OFFSET, _TIMEOUT_OFFSET

        # Encode with sentinel values and check they land at the documented offsets.
        amount_sentinel = 0xAABBCCDDEEFF
        timeout_sentinel = 0x1122334455667788
        calldata = encode_send_transfer_calldata(
            denom=_EUREKA_DENOM,
            amount=amount_sentinel,
            receiver="r",
            source_client="c",
            timeout_timestamp=timeout_sentinel,
        )
        # uint256 slots are right-aligned in 32 bytes.
        assert int.from_bytes(calldata[_AMOUNT_OFFSET : _AMOUNT_OFFSET + 32], "big") == amount_sentinel
        assert int.from_bytes(calldata[_TIMEOUT_OFFSET : _TIMEOUT_OFFSET + 32], "big") == timeout_sentinel


# ---------------------------------------------------------------------------
# EurekaComposer — interface id + sendTransferAndCompose calldata encoder
# ---------------------------------------------------------------------------


_COMPOSE_TYPES = ["(address,uint256,string,string,string,uint64,string)", "bytes"]


class TestEurekaComposer:
    def test_interface_id_matches_spec(self):
        # The Python and Solidity constants must agree — IBCCallbackReceiver
        # ERC-165-probes this exact id; if they drift the composer is silently
        # skipped and the follow-up program never runs.
        assert IIBC_SENDER_CALLBACKS_INTERFACE_ID == bytes.fromhex("d3ce6f1b")

    def test_send_and_compose_calldata_round_trips(self):
        from eth_abi import decode as abi_decode

        denom = Address(b"\x22" * 20)
        program = b"\x60\x00\x60\x00\xf3"  # PUSH1 0 PUSH1 0 RETURN
        data = encode_send_and_compose_calldata(
            denom=denom,
            amount=42,
            receiver="mantra1qq",
            source_client="07-tendermint-0",
            timeout_timestamp=1_700_000_000,
            program=program,
            memo="hi",
        )
        msg_tuple, decoded_program = abi_decode(_COMPOSE_TYPES, data[4:])
        d_denom, d_amount, d_receiver, d_src, d_dest, d_ts, d_memo = msg_tuple
        assert decoded_program == program
        assert d_denom.lower() == "0x" + bytes(denom).hex().lower()
        assert (d_amount, d_receiver, d_src, d_dest, d_ts, d_memo) == (
            42,
            "mantra1qq",
            "07-tendermint-0",
            "transfer",
            1_700_000_000,
            "hi",
        )

    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("amount", 1 << 256, "uint256"),
            ("timeout_timestamp", 1 << 64, "uint64"),
        ],
    )
    def test_send_and_compose_validates_range(self, field, value, match):
        kwargs = dict(
            denom=Address(b"\x22" * 20),
            amount=1,
            receiver="r",
            source_client="c",
            timeout_timestamp=1,
            program=b"",
        )
        kwargs[field] = value
        with pytest.raises(ValueError, match=match):
            encode_send_and_compose_calldata(**kwargs)

    def test_composer_solidity_compiles(self):
        # Skipped when the solidity toolchain isn't available in this env.
        try:
            from pathlib import Path

            from tests.live.sol_utils import compile_sol_file, ensure_solc
        except ImportError:
            pytest.skip("solcx / sol_utils not available in this env")

        ensure_solc("0.8.24")
        out = compile_sol_file(
            Path(__file__).parent.parent / "pydefi" / "bridge" / "EurekaComposer.sol",
            "EurekaComposer",
        )
        assert out["abi"] and out["bin"]
        # The externally-callable surface is what callers expect.
        fn_names = {e["name"] for e in out["abi"] if e.get("type") == "function"}
        assert {
            "sendTransferAndCompose",
            "onAckPacket",
            "onTimeoutPacket",
            "supportsInterface",
            "programs",
            "ics20Transfer",
            "interpreter",
        } <= fn_names
        # The composer must expose UnauthorizedCallback so the auth gate on
        # onAckPacket / onTimeoutPacket can surface a typed revert. Without
        # this check, anyone could call the callbacks and either trigger a
        # premature run or wipe a registered program.
        error_names = {e["name"] for e in out["abi"] if e.get("type") == "error"}
        assert "UnauthorizedCallback" in error_names
