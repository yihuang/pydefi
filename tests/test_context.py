"""Tests for the ProgramContext high-level Venom IR builder.

All ABI encode/decode tests cross-validate against :mod:`eth_abi`.
"""

from __future__ import annotations

import pytest
from eth_abi import encode as eth_abi_encode
from eth_abi.grammar import parse as _parse_abi_type
from vyper.compiler.settings import Settings, anchor_settings
from vyper.semantics.types.bytestrings import BytesT, StringT
from vyper.semantics.types.primitives import AddressT, BytesM_T
from vyper.semantics.types.shortcuts import UINT256_T
from vyper.semantics.types.subscriptable import DArrayT, SArrayT
from vyper.semantics.types.utils import type_from_abi
from vyper.venom.basicblock import IRLiteral, IROperand, IRVariable
from vyper.venom.context import IRContext

from pydefi.vm.context import ProgramContext
from pydefi.vm.stdlib import build_stdlib
from tests.conftest import mini_evm

_SHANGHAI_SETTINGS = Settings(evm_version="shanghai")


@pytest.fixture(autouse=True)
def _pin_shanghai_evm():
    with anchor_settings(_SHANGHAI_SETTINGS):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_return_bytes_buffer(ctx: ProgramContext, buf: IROperand) -> None:
    b = ctx.builder
    assert isinstance(buf, IRVariable)
    length = b.mload(buf)
    size = b.add(length, IRLiteral(32))
    b.return_(buf, size)


def _is_dynamic_abi(abi_type: str) -> bool:
    """Check if an ABI type is dynamically-sized."""
    parsed = _parse_abi_type(abi_type)
    if parsed.is_array:
        return True
    return parsed.base in ("bytes", "string")


def _expected_raw(abi_type: str, value) -> bytes:
    """Return raw ABI encoding of *value* (no tuple envelope).

    For static types the eth_abi output is used directly.
    For dynamic types the 32-byte tuple offset is stripped.
    """
    return eth_abi_encode([abi_type], [value])


def _check_encode(abi_type: str, value) -> None:
    """Encode *value* via Venom encoder and cross-check against eth_abi."""
    vyper_type = type_from_abi({"type": abi_type})
    expected = _expected_raw(abi_type, value)

    ctx = ProgramContext()
    if vyper_type._is_prim_word:
        args = [IRLiteral(value)]
    else:
        # Complex type: value is prepackaged bytes, embed in data section
        buf = ctx.embed_and_load(value)
        args = [buf]
    buf = ctx.abi_encode(args, [vyper_type], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, buf.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error

    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected), f"len mismatch: {enc_len} vs {len(expected)}"
    assert result.output[32:32 + enc_len] == expected, "data mismatch"


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------


def test_construct():
    ctx = ProgramContext()
    assert ctx.ir_ctx.entry_function is not None
    assert str(ctx.ir_ctx.entry_function.name) == "main"
    assert ctx.builder is not None


def test_inherited_new_variable():
    ctx = ProgramContext()
    v = ctx.new_variable("x", UINT256_T)
    assert v.name == "x"
    assert str(v.typ) == "uint256"
    assert not v.value.is_stack_value


def test_inherited_lookup():
    ctx = ProgramContext()
    ctx.new_variable("y", UINT256_T)
    v = ctx.lookup("y")
    assert v.name == "y"


def test_basic_compile():
    ctx = ProgramContext()
    x = ctx.builder.literal(42)
    ctx.builder.mstore(ctx.builder.alloca(32), x)
    ctx.builder.stop()
    bytecode = ctx.compile()
    assert isinstance(bytecode, bytes)
    assert len(bytecode) > 0


def test_stdlib_then_program_compiles():
    ir_ctx = IRContext()
    build_stdlib(ir_ctx)
    ctx = ProgramContext(ir_ctx, "main")
    ctx.builder.stop()
    assert next(iter(ir_ctx.functions)) == ir_ctx.entry_function.name
    bytecode = ctx.compile()
    assert len(bytecode) > 0


# ---------------------------------------------------------------------------
# 2. ABI encode — cross-validated against eth_abi
# ---------------------------------------------------------------------------


def test_abi_encode_uint256():
    _check_encode("uint256", 42)


def test_abi_encode_address():
    """Encode an address — eth_abi expects hex string."""
    expected = eth_abi_encode(["address"], ["0xABCDEF0000000000000000000000000000001234"])
    ctx = ProgramContext()
    addr_val = 0xABCDEF0000000000000000000000000000001234
    buf = ctx.abi_encode([IRLiteral(addr_val)], [AddressT()])
    _build_return_bytes_buffer(ctx, buf.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected)
    assert result.output[32:32 + enc_len] == expected


def test_abi_encode_bytes32():
    raw = b"hello" + b"\x00" * 27
    expected = eth_abi_encode(["bytes32"], [raw])
    ctx = ProgramContext()
    val = int.from_bytes(raw, "big")
    buf = ctx.abi_encode([IRLiteral(val)], [BytesM_T(32)])
    _build_return_bytes_buffer(ctx, buf.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert result.output[32:32 + enc_len] == expected


def test_abi_encode_tuple():
    """Encode (uint256, uint256) — no dynamic members, eth_abi matches directly."""
    expected = eth_abi_encode(["uint256", "uint256"], [10, 20])
    ctx = ProgramContext()
    buf = ctx.abi_encode([IRLiteral(10), IRLiteral(20)], [UINT256_T, UINT256_T])
    _build_return_bytes_buffer(ctx, buf.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected)
    assert result.output[32:32 + enc_len] == expected


def test_abi_encode_with_method_id():
    method_id = bytes.fromhex("a9059cbb")
    addr_hex = "0xABCDEF0000000000000000000000000000001234"
    addr_val = int(addr_hex, 16)
    val = 99
    expected = eth_abi_encode(["address", "uint256"], [addr_hex, val])

    ctx = ProgramContext()
    buf = ctx.abi_encode(
        [IRLiteral(addr_val), IRLiteral(val)],
        [AddressT(), UINT256_T],
        method_id=method_id,
    )
    _build_return_bytes_buffer(ctx, buf.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(method_id) + len(expected)
    assert result.output[32:36] == method_id
    assert result.output[36:36 + len(expected)] == expected


# ---------------------------------------------------------------------------
# 3. Dynamic bytes / string — cross-validated with eth_abi
# ---------------------------------------------------------------------------


def test_abi_encode_dynamic_bytes():
    """Encode Bytes[32] — eth_abi raw format: [length][data]."""
    data = b"dead"
    expected = _expected_raw("bytes", data)
    ctx = ProgramContext()
    buf = ctx.embed_and_load((4).to_bytes(32, "big") + b"dead\x00" * 8)
    encoded = ctx.abi_encode([buf], [BytesT(32)], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected)
    assert result.output[32:32 + enc_len] == expected


def test_abi_encode_string():
    s = "hello"
    expected = _expected_raw("string", s)
    ctx = ProgramContext()
    buf = ctx.embed_and_load((5).to_bytes(32, "big") + b"hello\x00" * 8)
    encoded = ctx.abi_encode([buf], [StringT(32)], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected)
    assert result.output[32:32 + enc_len] == expected


# ---------------------------------------------------------------------------
# 4. ABI decode — cross-validated against eth_abi
# ---------------------------------------------------------------------------


def test_abi_decode_uint256():
    expected = eth_abi_encode(["uint256"], [99])
    ctx = ProgramContext()
    buf = ctx.embed_and_load((len(expected)).to_bytes(32, "big") + expected)
    decoded = ctx.abi_decode(buf, UINT256_T, unwrap_tuple=False)
    loaded = ctx.builder.mload(decoded.operand)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 99


def test_abi_decode_reverts_on_short_data():
    ctx = ProgramContext()
    input_data = (16).to_bytes(32, "big") + (42).to_bytes(16, "big") + b"\x00" * 16
    buf = ctx.embed_and_load(input_data)
    ctx.abi_decode(buf, UINT256_T, unwrap_tuple=False)
    ctx.builder.stop()
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert result.is_error


def test_abi_decode_dynamic_bytes():
    """Decode bytes — input from eth_abi, wrapped in tuple offset."""
    data = b"dead"
    raw_abi = _expected_raw("bytes", data)
    # Wrapped in a tuple: [offset=32][raw_abi]
    wrapped = (32).to_bytes(32, "big") + raw_abi
    # Bytes buffer: [total_length][wrapped ABI data]
    buf_data = (len(wrapped)).to_bytes(32, "big") + wrapped
    ctx = ProgramContext()
    buf = ctx.embed_and_load(buf_data)
    decoded = ctx.abi_decode(buf, BytesT(32))  # unwrap_tuple=True wraps in (Bytes,)
    _build_return_bytes_buffer(ctx, decoded.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    dec_len = int.from_bytes(result.output[:32], "big")
    assert dec_len == len(data)
    assert result.output[32:32 + dec_len] == data


# ---------------------------------------------------------------------------
# 5. Round-trip — cross-validated with eth_abi
# ---------------------------------------------------------------------------


def test_abi_roundtrip_uint256():
    ctx_enc = ProgramContext()
    buf = ctx_enc.abi_encode([IRLiteral(12345)], [UINT256_T])
    _build_return_bytes_buffer(ctx_enc, buf.operand)
    bytecode_enc = ctx_enc.compile()
    result_enc = mini_evm(bytecode_enc)
    assert not result_enc.is_error
    encoded = result_enc.output

    expected = eth_abi_encode(["uint256"], [12345])
    enc_len = int.from_bytes(encoded[:32], "big")
    assert encoded[32:32 + enc_len] == expected

    ctx_dec = ProgramContext()
    buf_dec = ctx_dec.embed_and_load(encoded)
    decoded = ctx_dec.abi_decode(buf_dec, UINT256_T)
    loaded_val = ctx_dec.builder.mload(decoded.operand)
    out = ctx_dec.allocate_buffer(32)
    ctx_dec.builder.mstore(out.base_ptr().operand, loaded_val)
    ctx_dec.builder.return_(out.base_ptr().operand, IRLiteral(32))
    bytecode_dec = ctx_dec.compile()
    result_dec = mini_evm(bytecode_dec)
    assert not result_dec.is_error
    assert int.from_bytes(result_dec.output, "big") == 12345


# ---------------------------------------------------------------------------
# 6. Complex types: arrays (cross-validated with eth_abi)
# ---------------------------------------------------------------------------


def test_abi_encode_static_array():
    expected = eth_abi_encode(["uint256[3]"], [[10, 20, 30]])
    arr_type = SArrayT(UINT256_T, 3)
    raw = (10).to_bytes(32, "big") + (20).to_bytes(32, "big") + (30).to_bytes(32, "big")
    ctx = ProgramContext()
    buf = ctx.embed_and_load(raw)
    encoded = ctx.abi_encode([buf], [arr_type], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected)
    assert result.output[32:32 + enc_len] == expected


def test_abi_decode_static_array():
    data = eth_abi_encode(["uint256[3]"], [[10, 20, 30]])
    arr_type = SArrayT(UINT256_T, 3)
    ctx = ProgramContext()
    buf = ctx.embed_and_load((len(data)).to_bytes(32, "big") + data)
    decoded = ctx.abi_decode(buf, arr_type)
    elem2_ptr = ctx.builder.add(decoded.operand, IRLiteral(64))
    loaded = ctx.builder.mload(elem2_ptr)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 30


def test_abi_encode_dynamic_array():
    """Encode DynArray[uint256,3] with ensure_tuple=False (raw array data)."""
    expected = _expected_raw("uint256[]", [100, 200])
    arr_type = DArrayT(UINT256_T, 3)
    raw = (2).to_bytes(32, "big") + (100).to_bytes(32, "big") + (200).to_bytes(32, "big") + (0).to_bytes(32, "big")
    ctx = ProgramContext()
    buf = ctx.embed_and_load(raw)
    encoded = ctx.abi_encode([buf], [arr_type], ensure_tuple=False)
    _build_return_bytes_buffer(ctx, encoded.operand)
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    enc_len = int.from_bytes(result.output[:32], "big")
    assert enc_len == len(expected)
    assert result.output[32:32 + enc_len] == expected


def test_abi_decode_dynamic_array():
    """Decode DynArray[uint256, 3] — input from eth_abi (raw)."""
    raw_abi = _expected_raw("uint256[]", [10, 20])
    arr_type = DArrayT(UINT256_T, 3)
    ctx = ProgramContext()
    buf = ctx.embed_and_load((len(raw_abi)).to_bytes(32, "big") + raw_abi)
    decoded = ctx.abi_decode(buf, arr_type, unwrap_tuple=False)
    elem1_ptr = ctx.builder.add(decoded.operand, IRLiteral(64))
    loaded = ctx.builder.mload(elem1_ptr)
    out = ctx.allocate_buffer(32)
    ctx.builder.mstore(out.base_ptr().operand, loaded)
    ctx.builder.return_(out.base_ptr().operand, IRLiteral(32))
    bytecode = ctx.compile()
    result = mini_evm(bytecode)
    assert not result.is_error
    assert int.from_bytes(result.output, "big") == 20
