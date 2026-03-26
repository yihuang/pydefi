"""DeFiVM program builder — Python DSL for assembling DeFiVM bytecode.

Each function emits the raw bytes for one DeFiVM instruction.  Programs are
built by concatenating instruction bytes::

    from pydefi.vm.program import push_u256, push_addr, push_bytes, call, assert_ge

    program = (
        push_bytes(swap_calldata)
        + push_u256(0)
        + push_addr(SWAP_ADAPTER)
        + push_u256(0)
        + call()
        + push_addr(WHALE)
        + push_addr(WETH)
        + assert_ge("slippage: amount_out too low")
    )

Opcode reference (mirrors DeFiVM.sol)
--------------------------------------

Stack / Register
  0x01  PUSH_U256 <32 bytes>
  0x02  PUSH_ADDR <20 bytes>
  0x03  PUSH_BYTES <2-byte len> <data>
  0x04  DUP
  0x05  SWAP
  0x06  POP
  0x10  LOAD_REG  <reg-idx>
  0x11  STORE_REG <reg-idx>

Control flow
  0x20  JUMP  <2-byte target>
  0x21  JUMPI <2-byte target>    (forward only; pops condition)
  0x22  REVERT_IF <1-byte msgLen> <msg>
  0x23  ASSERT_GE <1-byte msgLen> <msg>  (a >= b: pop a top, pop b below)
  0x24  ASSERT_LE <1-byte msgLen> <msg>  (a <= b)

External / introspection
  0x30  CALL  <1-byte flags>  (bit0=requireSuccess)
  0x31  BALANCE_OF            (pop: token, account → push balance)
  0x32  SELF_ADDR             (push address(this))
  0x33  SUB                   (pop a, b → push a - b, saturates to 0 if a < b)

ABI / data
  0x40  PATCH_U256 <2-byte offset>
  0x41  PATCH_ADDR <2-byte offset>
  0x42  RET_U256   <2-byte offset>
  0x43  RET_SLICE  <2-byte offset> <2-byte len>
"""

from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# Opcode constants (mirrors DeFiVM.sol)
# ---------------------------------------------------------------------------

OP_PUSH_U256: int = 0x01
OP_PUSH_ADDR: int = 0x02
OP_PUSH_BYTES: int = 0x03
OP_DUP: int = 0x04
OP_SWAP: int = 0x05
OP_POP: int = 0x06
OP_LOAD_REG: int = 0x10
OP_STORE_REG: int = 0x11
OP_JUMP: int = 0x20
OP_JUMPI: int = 0x21
OP_REVERT_IF: int = 0x22
OP_ASSERT_GE: int = 0x23
OP_ASSERT_LE: int = 0x24
OP_CALL: int = 0x30
OP_BALANCE_OF: int = 0x31
OP_SELF_ADDR: int = 0x32
OP_SUB: int = 0x33
OP_PATCH_U256: int = 0x40
OP_PATCH_ADDR: int = 0x41
OP_RET_U256: int = 0x42
OP_RET_SLICE: int = 0x43

# ---------------------------------------------------------------------------
# Low-level encoding helpers
# ---------------------------------------------------------------------------


def _u256(n: int) -> bytes:
    """Encode a non-negative integer as a 32-byte big-endian uint256."""
    if n < 0:
        raise ValueError(f"_u256: value must be non-negative, got {n}")
    return n.to_bytes(32, "big")


def _addr(a: str) -> bytes:
    """Decode a hex Ethereum address string to 20 raw bytes."""
    raw = bytes.fromhex(a.removeprefix("0x"))
    if len(raw) != 20:
        raise ValueError(f"bad address length: {a!r}")
    return raw


def _u16(n: int) -> bytes:
    if not 0 <= n <= 0xFFFF:
        raise ValueError(f"_u16: value must be in 0..65535, got {n}")
    return struct.pack(">H", n)


# ---------------------------------------------------------------------------
# Stack / Register instructions
# ---------------------------------------------------------------------------


def push_u256(n: int) -> bytes:
    """Emit PUSH_U256 — push a uint256 literal onto the stack."""
    return bytes([OP_PUSH_U256]) + _u256(n)


def push_addr(a: str) -> bytes:
    """Emit PUSH_ADDR — push a 20-byte Ethereum address onto the stack."""
    return bytes([OP_PUSH_ADDR]) + _addr(a)


def push_bytes(data: bytes) -> bytes:
    """Emit PUSH_BYTES — push a bytes buffer onto the stack.

    The buffer is stored in the VM's internal buffer array; the stack receives
    the buffer index as a ``bytes32`` value.
    """
    if len(data) > 0xFFFF:
        raise ValueError(f"push_bytes: data too large ({len(data)} bytes, max 65535)")
    return bytes([OP_PUSH_BYTES]) + _u16(len(data)) + data


def dup() -> bytes:
    """Emit DUP — duplicate the top stack item."""
    return bytes([OP_DUP])


def swap() -> bytes:
    """Emit SWAP — exchange the top two stack items."""
    return bytes([OP_SWAP])


def pop() -> bytes:
    """Emit POP — discard the top stack item."""
    return bytes([OP_POP])


def load_reg(i: int) -> bytes:
    """Emit LOAD_REG i — push register *i* onto the stack."""
    if not 0 <= i <= 15:
        raise ValueError(f"load_reg: register index must be 0..15, got {i}")
    return bytes([OP_LOAD_REG, i])


def store_reg(i: int) -> bytes:
    """Emit STORE_REG i — pop the top of the stack into register *i*."""
    if not 0 <= i <= 15:
        raise ValueError(f"store_reg: register index must be 0..15, got {i}")
    return bytes([OP_STORE_REG, i])


# ---------------------------------------------------------------------------
# Control flow instructions
# ---------------------------------------------------------------------------


def jump(target: int) -> bytes:
    """Emit JUMP — unconditional forward jump to *target* (byte offset in program)."""
    return bytes([OP_JUMP]) + _u16(target)


def jumpi(target: int) -> bytes:
    """Emit JUMPI — conditional forward jump; pops a boolean condition from the stack."""
    return bytes([OP_JUMPI]) + _u16(target)


def revert_if(msg: str) -> bytes:
    """Emit REVERT_IF — pop condition; if truthy, revert with *msg* as ``Error(string)``."""
    raw = msg.encode()
    if len(raw) > 255:
        raise ValueError(f"revert_if: message too long ({len(raw)} bytes, max 255)")
    return bytes([OP_REVERT_IF, len(raw)]) + raw


def assert_ge(msg: str = "") -> bytes:
    """Emit ASSERT_GE — pop *a* (top), pop *b* (below); revert if ``a < b``.

    Stack effect: ``(a, b → )`` where ``a`` is the value that must be ≥ ``b``.
    """
    raw = msg.encode()
    if len(raw) > 255:
        raise ValueError(f"assert_ge: message too long ({len(raw)} bytes, max 255)")
    return bytes([OP_ASSERT_GE, len(raw)]) + raw


def assert_le(msg: str = "") -> bytes:
    """Emit ASSERT_LE — pop *a* (top), pop *b* (below); revert if ``a > b``."""
    raw = msg.encode()
    if len(raw) > 255:
        raise ValueError(f"assert_le: message too long ({len(raw)} bytes, max 255)")
    return bytes([OP_ASSERT_LE, len(raw)]) + raw


# ---------------------------------------------------------------------------
# External / introspection instructions
# ---------------------------------------------------------------------------


def call(require_success: bool = True) -> bytes:
    """Emit CALL — make a whitelisted external call.

    The caller must have pushed (top to bottom):

    - ``gasLimit`` (uint256)  ← top of stack
    - ``to``       (address)
    - ``value``    (uint256)
    - ``calldataBufIdx`` (buffer index from PUSH_BYTES)

    After the call the stack is unchanged (success/failure is handled via
    ``require_success``); the return data is available via RET_U256/RET_SLICE.
    """
    flags = 0x01 if require_success else 0x00
    return bytes([OP_CALL, flags])


def balance_of() -> bytes:
    """Emit BALANCE_OF — pop token address and account address; push ERC-20 balance."""
    return bytes([OP_BALANCE_OF])


def self_addr() -> bytes:
    """Emit SELF_ADDR — push the VM contract's own address onto the stack.

    Combined with :func:`balance_of` and ``push_u256(0)`` (ETH token), this
    gives the self ETH balance::

        self_addr() + push_u256(0) + balance_of()
    """
    return bytes([OP_SELF_ADDR])


def sub() -> bytes:
    """Emit SUB — pop ``a`` (top) then ``b``; push ``a - b`` (saturates to 0 if a < b).

    Use with two :func:`balance_of` calls to compute a balance delta::

        push_addr(account)
        + push_addr(token)    # or push_u256(0) for ETH
        + balance_of()        # pre-balance on stack
        + <call that changes balance>
        + push_addr(account)
        + push_addr(token)
        + balance_of()        # post-balance on top
        + sub()               # post - pre
    """
    return bytes([OP_SUB])


# ---------------------------------------------------------------------------
# ABI / data instructions
# ---------------------------------------------------------------------------


def patch_u256(offset: int) -> bytes:
    """Emit PATCH_U256 — overwrite a 32-byte word in a buffer at *offset*.

    Stack effect: pop value (uint256), pop bufIdx → push bufIdx.
    """
    return bytes([OP_PATCH_U256]) + _u16(offset)


def patch_addr(offset: int) -> bytes:
    """Emit PATCH_ADDR — overwrite 20 bytes in a buffer starting at *offset*.

    Stack effect: pop addr (address), pop bufIdx → push bufIdx.
    The 20 bytes are written starting exactly at *offset* (raw byte copy).
    """
    return bytes([OP_PATCH_ADDR]) + _u16(offset)


def ret_u256(offset: int) -> bytes:
    """Emit RET_U256 — push a uint256 from the last call's returndata at *offset*."""
    return bytes([OP_RET_U256]) + _u16(offset)


def ret_slice(offset: int, length: int) -> bytes:
    """Emit RET_SLICE — push a bytes slice from the last call's returndata.

    The slice ``returndata[offset : offset + length]`` is stored as a new buffer
    and its index is pushed onto the stack.
    """
    return bytes([OP_RET_SLICE]) + _u16(offset) + _u16(length)
