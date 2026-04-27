"""Tests for :class:`pydefi.vm.context.ProgramContext` (typed Venom IR builder)."""

from __future__ import annotations

from typing import Callable

import pytest
from eth_abi.abi import encode as eth_abi_encode
from vyper.compiler.settings import Settings, anchor_settings
from vyper.semantics.types.bytestrings import BytesT, StringT
from vyper.semantics.types.primitives import AddressT, BytesM_T
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.semantics.types.subscriptable import DArrayT, SArrayT, TupleT
from vyper.venom.basicblock import IRLiteral
from vyper.venom.context import IRContext

from pydefi.vm.context import ProgramContext, RuntimeBytes, parse_sig, parse_sigs
from pydefi.vm.stdlib import build_stdlib
from tests.conftest import mini_evm

# Belt-and-suspenders: pin shanghai globally so that any non-anchored sub-call
# in the IR pipeline (ProgramContext() already defaults shanghai) still picks
# up the right opcode table.
_SHANGHAI = Settings(evm_version="shanghai")


@pytest.fixture(autouse=True)
def _pin_shanghai_evm():
    with anchor_settings(_SHANGHAI):
        yield


def _internal_layout_uint_and_bytes(value: int, payload: bytes, maxlen: int) -> bytes:
    """Vyper internal-layout form of ``(uint256, BytesT[maxlen])`` — the shape
    :meth:`abi_decode` writes into its destination buffer."""
    out = value.to_bytes(32, "big")
    out += len(payload).to_bytes(32, "big")
    padded_len = ((maxlen + 31) // 32) * 32
    out += payload + b"\x00" * (padded_len - len(payload))
    return out


def _compile_and_run(build_fn: Callable[[ProgramContext], None], calldata: bytes = b"") -> bytes:
    """Run *build_fn* against a fresh :class:`ProgramContext`, compile, execute via
    :func:`mini_evm`, return the raw output bytes (asserting non-revert)."""
    prog = ProgramContext()
    build_fn(prog)
    bytecode = prog.compile()
    result = mini_evm(bytecode, calldata=calldata) if calldata else mini_evm(bytecode)
    assert not result.is_error, f"unexpected revert: {result.output.hex()}"
    return bytes(result.output)


def _encode_and_return(types: tuple, values: tuple, **enc_kwargs) -> bytes:
    """Build a one-shot ProgramContext: ``abi_encode(types, values) → RETURN(buf, length)``
    and return the resulting EVM output bytes."""

    def build(prog: ProgramContext) -> None:
        enc = prog.abi_encode(types, values, **enc_kwargs)
        prog.builder.return_(enc.buf, enc.length)

    return _compile_and_run(build)


class TestConstruction:
    def test_fresh_context_has_main_entry(self) -> None:
        ctx = ProgramContext()
        assert ctx.ir_ctx.entry_function is not None
        assert str(ctx.ir_ctx.entry_function.name) == "main"
        assert ctx.builder is not None

    def test_inherited_new_variable(self) -> None:
        ctx = ProgramContext()
        v = ctx.new_variable("x", UINT256_T)
        assert v.name == "x"
        assert str(v.typ) == "uint256"
        assert not v.value.is_stack_value

    def test_inherited_lookup(self) -> None:
        ctx = ProgramContext()
        ctx.new_variable("y", UINT256_T)
        v = ctx.lookup("y")
        assert v.name == "y"

    def test_basic_compile_round_trip(self) -> None:
        ctx = ProgramContext()
        x = ctx.builder.literal(42)
        ctx.builder.mstore(ctx.builder.alloca(32), x)
        ctx.builder.stop()
        bytecode = ctx.compile()
        assert isinstance(bytecode, bytes)
        assert len(bytecode) > 0

    def test_stdlib_then_program_compiles(self) -> None:
        ir_ctx = IRContext()
        build_stdlib(ir_ctx)
        ctx = ProgramContext(ir_ctx, "main")
        ctx.builder.stop()
        bytecode = ctx.compile()
        # compile() reorders so the entry function emits first regardless of
        # insertion order (build_stdlib added two funcs before main).
        assert next(iter(ir_ctx.functions)) == ir_ctx.entry_function.name
        assert len(bytecode) > 0


_UINT_BYTES_CASES = [
    pytest.param(0, b"", 64, id="empty-payload"),
    pytest.param(42, b"hello", 64, id="short-payload"),
    pytest.param(1, bytes(range(33)), 64, id="cross-32B-boundary"),
]


class TestEncodeAbi:
    def test_single_uint256(self) -> None:
        out = _encode_and_return((UINT256_T,), (42,))
        assert out == eth_abi_encode(["uint256"], [42])

    def test_multiple_scalars(self) -> None:
        out = _encode_and_return((UINT256_T, UINT256_T), (10, 20))
        assert out == eth_abi_encode(["uint256", "uint256"], [10, 20])

    def test_address_left_pads_to_32_bytes(self) -> None:
        addr_int = 0xABCDEF0000000000000000000000000000001234
        out = _encode_and_return((AddressT(),), (addr_int,))
        assert out[12:32] == addr_int.to_bytes(20, "big")

    def test_bytes32(self) -> None:
        val_int = int.from_bytes(b"hello" + b"\x00" * 27, "big")
        out = _encode_and_return((BytesM_T(32),), (val_int,))
        assert len(out) == 32
        assert out[:5] == b"hello"

    def test_bytesm_accepts_python_bytes(self) -> None:
        payload = b"\x11\x22\x33\x44"
        out = _encode_and_return((BytesM_T(4),), (payload,))
        assert out[:4] == payload
        assert out[4:] == b"\x00" * 28

    def test_bytesm_rejects_wrong_size_python_bytes(self) -> None:
        with pytest.raises(ValueError, match="expects exactly 4 bytes"):
            _encode_and_return((BytesM_T(4),), (b"\x11\x22\x33",))

    def test_signed_int_accepts_negative_python_int(self) -> None:
        int256 = parse_sig("int256")
        out = _encode_and_return((int256,), (-1,))
        assert out == eth_abi_encode(["int256"], [-1])

    def test_signed_int_rejects_out_of_bounds(self) -> None:
        int8 = parse_sig("int8")
        with pytest.raises(ValueError, match="out of bounds"):
            _encode_and_return((int8,), (128,))
        with pytest.raises(ValueError, match="out of bounds"):
            _encode_and_return((int8,), (-129,))

    def test_method_id_prepends_4_byte_selector(self) -> None:
        method_id = bytes.fromhex("a9059cbb")
        addr_int = 0x0000000000000000000000000000000000000001
        out = _encode_and_return(
            (AddressT(), UINT256_T),
            (addr_int, 2),
            method_id=method_id,
        )
        assert len(out) == 68  # 4 (selector) + 64 (two static words)
        assert out[:4] == method_id
        assert out[4:] == eth_abi_encode(["address", "uint256"], [f"0x{addr_int:040x}", 2])

    @pytest.mark.parametrize("value, payload, maxlen", _UINT_BYTES_CASES)
    def test_baked_uint256_and_bytes(self, value: int, payload: bytes, maxlen: int) -> None:
        """Baked Python literals → ABI encoding matches ``eth_abi.encode``."""
        out = _encode_and_return((UINT256_T, BytesT(maxlen)), (value, payload))
        assert out == eth_abi_encode(["uint256", "bytes"], [value, payload])

    def test_string_literal(self) -> None:
        out = _encode_and_return((StringT(32),), ("hello",))
        assert out == eth_abi_encode(["string"], ["hello"])

    @pytest.mark.parametrize("value, payload", [(0, b""), (42, b"hello"), (1, bytes(range(50)))])
    def test_runtime_bytes_match_eth_abi(self, value: int, payload: bytes) -> None:
        """``RuntimeBytes`` from calldata → same encoding as eth_abi."""
        maxlen = 64

        def build(prog: ProgramContext) -> None:
            b = prog.builder
            value_v = b.calldataload(0)
            length_v = b.calldataload(32)
            padded_len = ((maxlen + 31) // 32) * 32
            payload_buf = b.alloca(padded_len)
            b.calldatacopy(payload_buf, 64, padded_len)
            enc = prog.abi_encode(
                (UINT256_T, BytesT(maxlen)),
                (value_v, RuntimeBytes(length=length_v, data_ptr=payload_buf)),
            )
            b.return_(enc.buf, enc.length)

        calldata = _internal_layout_uint_and_bytes(value, payload, maxlen)
        out = _compile_and_run(build, calldata=calldata)
        assert out == eth_abi_encode(["uint256", "bytes"], [value, payload])

    def test_runtime_bytes_reverts_when_length_exceeds_cap(self) -> None:
        """RuntimeBytes whose runtime length exceeds the BytesT cap must revert."""
        maxlen = 64

        def build(prog: ProgramContext) -> None:
            b = prog.builder
            value_v = b.calldataload(0)
            length_v = b.calldataload(32)
            padded_len = ((maxlen + 31) // 32) * 32
            payload_buf = b.alloca(padded_len)
            b.calldatacopy(payload_buf, 64, padded_len)
            enc = prog.abi_encode(
                (UINT256_T, BytesT(maxlen)),
                (value_v, RuntimeBytes(length=length_v, data_ptr=payload_buf)),
            )
            b.return_(enc.buf, enc.length)

        prog = ProgramContext()
        build(prog)
        bytecode = prog.compile()
        # length=65 > maxlen=64 — must revert
        calldata = (1).to_bytes(32, "big") + (65).to_bytes(32, "big") + bytes(range(64))
        result = mini_evm(bytecode, calldata=calldata)
        assert result.is_error, "expected revert when RuntimeBytes length exceeds cap"

    def test_nested_tuple(self) -> None:
        inner = TupleT((AddressT(), UINT256_T))
        outer = TupleT((UINT256_T, inner))
        addr_int = 0xCB00CB00CB00CB00CB00CB00CB00CB00CB00CB00
        out = _encode_and_return((outer,), ((10, (addr_int, 42)),))
        expected = eth_abi_encode(
            ["(uint256,(address,uint256))"],
            [(10, (f"0x{addr_int:040x}", 42))],
        )
        assert out == expected

    def test_static_array(self) -> None:
        out = _encode_and_return((SArrayT(UINT256_T, 3),), ([10, 20, 30],))
        assert out == eth_abi_encode(["uint256[3]"], [[10, 20, 30]])

    def test_dynamic_array(self) -> None:
        out = _encode_and_return((DArrayT(UINT256_T, 3),), ([100, 200],))
        assert out == eth_abi_encode(["uint256[]"], [[100, 200]])

    def test_parse_sigs_and_direct_types_produce_identical_bytecode(self) -> None:
        """Sig-string types and hand-built VyperType objects must compile to
        byte-for-byte identical bytecode."""
        prog_a = ProgramContext()
        enc_a = prog_a.abi_encode(parse_sigs(["uint256", "bytes:64"]), (42, b"hello"))
        prog_a.builder.return_(enc_a.buf, enc_a.length)

        prog_b = ProgramContext()
        enc_b = prog_b.abi_encode((UINT256_T, BytesT(64)), (42, b"hello"))
        prog_b.builder.return_(enc_b.buf, enc_b.length)

        assert prog_a.compile() == prog_b.compile()

    @pytest.mark.parametrize(
        "values, exc, msg_fragment",
        [
            pytest.param(((1,),), ValueError, "arity mismatch", id="too-few-tuple-members"),
            pytest.param(((1, b"hi", 3),), ValueError, "arity mismatch", id="too-many-tuple-members"),
            pytest.param((1,), TypeError, "expects list/tuple", id="non-sequence-tuple-payload"),
        ],
    )
    def test_tuple_payload_validation(self, values: tuple, exc: type, msg_fragment: str) -> None:
        tuple_type = parse_sig("(uint256, bytes:32)")
        with pytest.raises(exc, match=msg_fragment):
            ProgramContext().abi_encode((tuple_type,), values)


class TestDecodeAbi:
    def test_prim_word_round_trip_uint256(self) -> None:
        """Encode in one program, decode in another — workaround for Vyper's
        ``ConcretizeMemLocPass`` allocator crash on prim-word in-program round-trip."""
        encoded = _encode_and_return((UINT256_T,), (12345,))

        ctx = ProgramContext()
        src = ctx.embed_and_load(encoded)
        decoded = ctx.abi_decode(src, IRLiteral(len(encoded)), (UINT256_T,))
        val = decoded.read_word(0)
        out = ctx.allocate_buffer(32)
        ctx.builder.mstore(out.base_ptr().operand, val)
        ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

        result = mini_evm(ctx.compile())
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 12345

    def test_prim_word_round_trip_two_uint256(self) -> None:
        encoded = _encode_and_return((UINT256_T, UINT256_T), (7, 42))

        ctx = ProgramContext()
        src = ctx.embed_and_load(encoded)
        decoded = ctx.abi_decode(src, IRLiteral(len(encoded)), (UINT256_T, UINT256_T))
        val1 = decoded.read_word(1)
        out = ctx.allocate_buffer(32)
        ctx.builder.mstore(out.base_ptr().operand, val1)
        ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

        result = mini_evm(ctx.compile())
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 42

    @pytest.mark.parametrize("value, payload, maxlen", _UINT_BYTES_CASES)
    def test_calldata_round_trips_to_internal_layout(self, value: int, payload: bytes, maxlen: int) -> None:
        """ABI calldata → decoded buffer must match Vyper's internal layout."""
        encoded = eth_abi_encode(["uint256", "bytes"], [value, payload])
        types = (UINT256_T, BytesT(maxlen))
        wrapped = TupleT(types)

        def build(prog: ProgramContext) -> None:
            b = prog.builder
            src_buf = prog.allocate_buffer(wrapped.abi_type.size_bound(), annotation="abi_src")
            b.calldatacopy(src_buf._ptr, 0, len(encoded))
            decoded = prog.abi_decode(src_buf._ptr, IRLiteral(len(encoded)), types)
            b.return_(decoded.buf, wrapped.memory_bytes_required)

        out = _compile_and_run(build, calldata=encoded)
        assert out == _internal_layout_uint_and_bytes(value, payload, maxlen)

    @pytest.mark.parametrize("value, payload, maxlen", _UINT_BYTES_CASES)
    def test_typed_accessors_match_layout(self, value: int, payload: bytes, maxlen: int) -> None:
        """``DecodedBuffer.read_word`` / ``read_bytes`` retrieve fields at runtime."""
        encoded = eth_abi_encode(["uint256", "bytes"], [value, payload])
        types = parse_sigs(["uint256", f"bytes:{maxlen}"])

        def build(prog: ProgramContext) -> None:
            b = prog.builder
            src_buf = b.alloca(len(encoded) + 32)  # +32 for hi-bound slack
            b.calldatacopy(src_buf, 0, len(encoded))
            decoded = prog.abi_decode(src_buf, IRLiteral(len(encoded)), types)
            value_v = decoded.read_word(0)
            payload_len, payload_ptr = decoded.read_bytes(1)

            # Stage [value][len][data] for return.
            padded_max = ((maxlen + 31) // 32) * 32
            out_buf = b.alloca(64 + padded_max)
            b.mstore(out_buf, value_v)
            b.mstore(b.add(out_buf, 32), payload_len)
            prog.copy_memory_dynamic(b.add(out_buf, 64), payload_ptr, payload_len)
            b.return_(out_buf, b.add(payload_len, 64))

        out = _compile_and_run(build, calldata=encoded)
        expected = value.to_bytes(32, "big") + len(payload).to_bytes(32, "big") + payload
        assert out == expected

    def test_nested_tuple_round_trip(self) -> None:
        """Round-trip a (uint256, (address, uint256)) tuple."""
        inner = TupleT((AddressT(), UINT256_T))
        outer = TupleT((UINT256_T, inner))
        addr_int = 0xCB00CB00CB00CB00CB00CB00CB00CB00CB00CB00

        ctx = ProgramContext()
        enc = ctx.abi_encode((outer,), ((10, (addr_int, 42)),))
        decoded = ctx.abi_decode(enc.buf, enc.length, (outer,))
        # Outer wraps the inner tuple at field 0; inner uint256 sits at offset
        # 32 + 32 = 64 from the decoded buffer.
        inner_uint_ptr = ctx.builder.add(decoded.buf, IRLiteral(64))
        val = ctx.builder.mload(inner_uint_ptr)

        out = ctx.allocate_buffer(32)
        ctx.builder.mstore(out.base_ptr().operand, val)
        ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

        result = mini_evm(ctx.compile())
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 42

    def test_static_array_round_trip(self) -> None:
        arr = SArrayT(UINT256_T, 3)
        ctx = ProgramContext()
        enc = ctx.abi_encode((arr,), ([10, 20, 30],))
        decoded = ctx.abi_decode(enc.buf, enc.length, (arr,))
        # Third element at offset 64 in array memory layout.
        elem2_ptr = ctx.builder.add(decoded.buf, IRLiteral(64))
        val = ctx.builder.mload(elem2_ptr)
        out = ctx.allocate_buffer(32)
        ctx.builder.mstore(out.base_ptr().operand, val)
        ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

        result = mini_evm(ctx.compile())
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 30

    def test_dynamic_array_round_trip(self) -> None:
        arr = DArrayT(UINT256_T, 3)
        ctx = ProgramContext()
        enc = ctx.abi_encode((arr,), ([10, 20],))
        decoded = ctx.abi_decode(enc.buf, enc.length, (arr,))
        # Memory layout: [count][elem0][elem1]...; elem 1 at offset 64.
        elem1_ptr = ctx.builder.add(decoded.buf, IRLiteral(64))
        val = ctx.builder.mload(elem1_ptr)
        out = ctx.allocate_buffer(32)
        ctx.builder.mstore(out.base_ptr().operand, val)
        ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))

        result = mini_evm(ctx.compile())
        assert not result.is_error
        assert int.from_bytes(result.output, "big") == 20

    def test_read_word_rejects_bytes_slot(self) -> None:
        decoded = self._make_decoded(("uint256", "bytes:32"))
        with pytest.raises(TypeError, match="not a primitive"):
            decoded.read_word(1)  # field 1 is bytes:32

    def test_read_bytes_rejects_primitive_slot(self) -> None:
        decoded = self._make_decoded(("uint256", "bytes:32"))
        with pytest.raises(TypeError, match="not BytesT"):
            decoded.read_bytes(0)  # field 0 is uint256

    def test_read_word_rejects_out_of_range_index(self) -> None:
        decoded = self._make_decoded(("uint256", "bytes:32"))
        with pytest.raises(IndexError):
            decoded.read_word(99)

    @staticmethod
    def _make_decoded(sigs: tuple[str, ...]):
        """Build a DecodedBuffer for the given sigs.  The IR does not need
        to compile — we only exercise Python-side accessor validation."""
        prog = ProgramContext()
        b = prog.builder
        types = parse_sigs(list(sigs))
        src_buf = b.alloca(128)
        b.calldatacopy(src_buf, 0, 96)
        return prog.abi_decode(src_buf, IRLiteral(96), types)


class TestParseSig:
    @pytest.mark.parametrize("sig", ["uint256", "uint8", "int128", "address", "bool", "bytes32"])
    def test_atoms_are_prim_word(self, sig: str) -> None:
        assert parse_sig(sig)._is_prim_word

    def test_variable_bytes(self) -> None:
        bytes64 = parse_sig("bytes:64")
        assert isinstance(bytes64, BytesT)
        assert bytes64.length == 64

    def test_dynamic_array_atoms(self) -> None:
        addrs = parse_sig("address[10]")
        assert isinstance(addrs, DArrayT)
        assert addrs.length == 10

        bytes_array = parse_sig("bytes:128[4]")
        assert isinstance(bytes_array, DArrayT)
        assert bytes_array.length == 4
        assert isinstance(bytes_array.value_type, BytesT)
        assert bytes_array.value_type.length == 128

    def test_parse_sigs_returns_tuple_of_types(self) -> None:
        types = parse_sigs(["uint256", "bytes:64"])
        assert len(types) == 2

    def test_bare_bytes_is_ambiguous(self) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            parse_sig("bytes")

    def test_flat_tuple(self) -> None:
        t = parse_sig("(uint256, bytes:64)")
        assert isinstance(t, TupleT)
        assert len(t.member_types) == 2

    def test_nested_tuple(self) -> None:
        t = parse_sig("(uint256, (address, bytes:32))")
        assert isinstance(t, TupleT)
        assert len(t.member_types) == 2
        inner = t.member_types[1]
        assert isinstance(inner, TupleT)
        assert len(inner.member_types) == 2

    def test_tuple_with_dyn_array_atoms(self) -> None:
        t = parse_sig("(bytes:64, bytes:128[4])")
        assert isinstance(t, TupleT)
        assert isinstance(t.member_types[0], BytesT)
        assert isinstance(t.member_types[1], DArrayT)
        assert t.member_types[1].length == 4

    @pytest.mark.parametrize("sig", ["(uint256,bytes:64)", "( uint256 ,  bytes:64 )"])
    def test_whitespace_tolerance(self, sig: str) -> None:
        parse_sig(sig)

    def test_one_element_tuple(self) -> None:
        """``(uint256,)`` parses as ``TupleT((UINT256_T,))`` per ABI spec."""
        t = parse_sig("(uint256,)")
        assert isinstance(t, TupleT) and len(t.member_types) == 1

    @pytest.mark.parametrize("bad", ["(", "(uint256", "uint256)", "()", "(uint256,,bytes:64)"])
    def test_malformed_tuples_raise(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_sig(bad)
