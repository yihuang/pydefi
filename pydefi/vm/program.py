"""DeFiVM program builder — Python DSL for assembling DeFiVM EVM bytecode.

Programs are raw EVM bytecode executed on the native EVM stack by the
Analog-Labs interpreter.  ``execute()`` in DeFiVM.sol runs a program via
``DELEGATECALL`` into that interpreter; it does not deploy the bytecode via
``CREATE``.  As a result, callers should reason about calldata, memory, and
gas behaviour using the interpreter/delegatecall execution model rather than
the semantics of a freshly deployed contract.

Memory conventions used by the interpreter
------------------------------------------
- Registers:          ``memory[0x80 + i*32]`` for ``i`` in 0..15
- Free memory pointer: ``memory[0x40]`` (initialised to 0x280 on first use)
- Dynamic buffers:    allocated starting at ``memory[0x280]``

Usage example::

    from pydefi.vm.program import push_u256, push_addr, push_bytes, call, assert_ge

    program = (
        push_u256(0) + push_u256(0)          # retLen, retOffset
        + push_bytes(swap_calldata)
        + push_u256(0) + push_addr(SWAP_ADAPTER) + gas_opcode()
        + call()
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Opcode constants — single EVM opcode identifiers
# ---------------------------------------------------------------------------

OP_PUSH_U256: int = 0x7F  # PUSH32 — emitted by push_u256()
OP_PUSH_ADDR: int = 0x73  # PUSH20 — emitted by push_addr()
OP_DUP: int = 0x80  # DUP1 — emitted by dup()
OP_SWAP: int = 0x90  # SWAP1 — emitted by swap()
OP_POP: int = 0x50  # POP — emitted by pop()
# NOTE: OP_JUMP and OP_JUMPI share the PUSH2 prefix (0x61) with load_reg/store_reg.
# They identify the first byte of the 4-byte PUSH2 target + JUMP/JUMPI sequence.
OP_JUMP: int = 0x61  # first byte of jump() sequence (PUSH2 target JUMP)
OP_JUMPI: int = 0x61  # first byte of jumpi() sequence (PUSH2 target JUMPI)
OP_ADD: int = 0x01  # ADD — emitted by add()
OP_MUL: int = 0x02  # MUL — emitted by mul()
OP_SUB: int = 0x81  # DUP2 — first byte of saturating sub() sequence
OP_DIV: int = 0x04  # DIV — emitted by div()
OP_MOD: int = 0x06  # MOD — emitted by mod()
OP_LT: int = 0x10  # LT — emitted by lt()
OP_GT: int = 0x11  # GT — emitted by gt()
OP_EQ: int = 0x14  # EQ — emitted by eq()
OP_ISZERO: int = 0x15  # ISZERO — emitted by iszero()
OP_AND: int = 0x16  # AND — emitted by bitwise_and()
OP_OR: int = 0x17  # OR — emitted by bitwise_or()
OP_XOR: int = 0x18  # XOR — emitted by bitwise_xor()
OP_NOT: int = 0x19  # NOT — emitted by bitwise_not()
OP_SHL: int = 0x1B  # SHL — emitted by shl()
OP_SHR: int = 0x1C  # SHR — emitted by shr()
OP_GAS: int = 0x5A  # GAS — emitted by gas_opcode()
OP_CALL: int = 0xF1  # CALL — core opcode in the call() sequence
OP_SELF_ADDR: int = 0x30  # ADDRESS — emitted by self_addr()
OP_BALANCE: int = 0x31  # BALANCE — used in balance_of() ETH path
OP_MLOAD: int = 0x51  # MLOAD — load word from memory
OP_MSTORE: int = 0x52  # MSTORE — store word to memory
OP_JUMPDEST: int = 0x5B  # JUMPDEST — marks a valid jump target
OP_RETURNDATACOPY: int = 0x3E  # RETURNDATACOPY — copy returndata into memory
OP_STATICCALL: int = 0xFA  # STATICCALL — used by balance_of() ERC-20 path
OP_REVERT: int = 0xFD  # REVERT

# ---------------------------------------------------------------------------
# Private module-level opcode aliases (used only inside multi-opcode helpers)
# ---------------------------------------------------------------------------

_PUSH1: int = 0x60  # PUSH1 — followed by one immediate byte
_PUSH2: int = 0x61  # PUSH2 — followed by two immediate bytes (= OP_JUMP first byte)
_PC: int = 0x58  # PC — push current program counter
_JUMP: int = 0x56  # JUMP raw opcode (inside multi-opcode sequences)
_JUMPI: int = 0x57  # JUMPI raw opcode
_SUB_OP: int = 0x03  # SUB — integer subtraction
_DUP2: int = 0x81  # DUP2
_DUP3: int = 0x82  # DUP3
_DUP4: int = 0x83  # DUP4
_DUP6: int = 0x85  # DUP6
_SWAP2: int = 0x91  # SWAP2
_SWAP3: int = 0x92  # SWAP3

# ---------------------------------------------------------------------------
# Stack instructions
# ---------------------------------------------------------------------------


def push_u256(n: int) -> bytes:
    """Emit PUSH32 — push a uint256 literal onto the native EVM stack."""
    if n < 0:
        raise ValueError(f"push_u256: value must be non-negative, got {n}")
    return bytes([OP_PUSH_U256]) + n.to_bytes(32, "big")


def push_addr(a: str) -> bytes:
    """Emit PUSH20 — push a 20-byte Ethereum address onto the native EVM stack."""
    raw = bytes.fromhex(a.removeprefix("0x"))
    if len(raw) != 20:
        raise ValueError(f"push_addr: bad address length: {a!r}")
    return bytes([OP_PUSH_ADDR]) + raw


def gas_opcode() -> bytes:
    """Emit GAS — push the remaining gas onto the stack."""
    return bytes([OP_GAS])


def push_bytes(data: bytes) -> bytes:
    """Copy *data* into free memory and leave ``[argsOffset(TOS), argsLen(2nd)]``.

    Each 32-byte chunk of *data* is embedded as a ``PUSH32`` immediate and
    stored with ``MSTORE``.  The free memory pointer at ``mem[0x40]`` is
    initialised to ``0x280`` minimum and advanced after the allocation.

    This implementation avoids ``CALLDATACOPY`` (opcode ``0x37``) because the
    Analog-Labs EVM interpreter's CALLDATACOPY handler ignores the user-supplied
    source offset and always reads from ``calldatasize`` (past the end of the
    program calldata), copying zeros instead of the intended bytes.
    """
    if len(data) > 0xFFFF:
        raise ValueError(f"push_bytes: data too large ({len(data)} bytes, max 65535)")
    blen = len(data)
    blen_padded = (blen + 31) & ~31

    def _push_imm(v: int) -> bytes:
        """Smallest PUSH opcode for non-negative integer *v* (1–3 bytes)."""
        if v <= 0xFF:
            return bytes([_PUSH1, v])  # PUSH1
        elif v <= 0xFFFF:
            return bytes([_PUSH2, v >> 8, v & 0xFF])  # PUSH2
        else:
            n = (v.bit_length() + 7) // 8
            return bytes([0x5F + n]) + v.to_bytes(n, "big")

    # Build the output in a list of chunks to avoid O(n²) bytes concatenation.
    parts: list[bytes] = []

    # Compute max_fp = fp | (0x280 * (fp == 0))
    parts.append(
        bytes(
            [
                _PUSH1,
                0x40,  # PUSH1 0x40
                OP_MLOAD,  # MLOAD           → [fp]
                _PUSH2,
                0x02,
                0x80,  # PUSH2 0x0280    → [0x280, fp]
                _DUP2,  # DUP2            → [fp, 0x280, fp]
                OP_ISZERO,  # ISZERO          → [fp==0, 0x280, fp]
                OP_MUL,  # MUL             → [0x280*(fp==0), fp]
                OP_OR,  # OR              → [max_fp]
            ]
        )
    )

    # Store each 32-byte chunk; max_fp stays at TOS between iterations.
    for i in range(0, blen_padded, 32):
        chunk = data[i : i + 32].ljust(32, b"\x00")
        chunk_val = int.from_bytes(chunk, "big")

        # Duplicate max_fp, then compute store address = max_fp + i.
        parts.append(bytes([OP_DUP]))  # DUP1 → [max_fp, max_fp, ...]
        if i > 0:
            parts.append(_push_imm(i))  # [i, max_fp, max_fp]
            parts.append(bytes([OP_ADD]))  # ADD → [max_fp+i, max_fp]

        # Push the 32-byte chunk (use PUSH1 0 for all-zero chunks).
        if chunk_val == 0:
            parts.append(bytes([_PUSH1, 0x00]))  # PUSH1 0
        else:
            parts.append(bytes([OP_PUSH_U256]) + chunk)  # PUSH32 chunk

        parts.append(bytes([OP_SWAP, OP_MSTORE]))  # SWAP1, MSTORE → mem[max_fp+i]=chunk; [max_fp]

    # Update free-memory pointer: mem[0x40] = max_fp + blen_padded.
    parts.append(bytes([OP_DUP]))  # DUP1 → [max_fp, max_fp]
    parts.append(_push_imm(blen_padded))  # [blen_padded, max_fp, max_fp]
    parts.append(bytes([OP_ADD]))  # ADD → [new_fp, max_fp]
    parts.append(bytes([_PUSH1, 0x40]))  # PUSH1 0x40
    parts.append(bytes([OP_MSTORE]))  # MSTORE → mem[0x40] = new_fp; [max_fp]

    # Leave [argsOffset(TOS), argsLen(2nd)].
    parts.append(_push_imm(blen))  # [blen, max_fp]
    parts.append(bytes([OP_SWAP]))  # SWAP1 → [max_fp=argsOffset, blen=argsLen]
    return b"".join(parts)


def dup() -> bytes:
    """Emit DUP1 — duplicate the top stack item."""
    return bytes([OP_DUP])


def swap() -> bytes:
    """Emit SWAP1 — exchange the top two stack items."""
    return bytes([OP_SWAP])


def pop() -> bytes:
    """Emit POP — discard the top stack item."""
    return bytes([OP_POP])


def load_reg(i: int) -> bytes:
    """Emit PUSH2 addr MLOAD — push register *i* onto the stack."""
    if not 0 <= i <= 15:
        raise ValueError(f"load_reg: register index must be 0..15, got {i}")
    addr = 0x80 + i * 32
    return bytes([_PUSH2, addr >> 8, addr & 0xFF, OP_MLOAD])


def store_reg(i: int) -> bytes:
    """Emit PUSH2 addr MSTORE — pop TOS into register *i*."""
    if not 0 <= i <= 15:
        raise ValueError(f"store_reg: register index must be 0..15, got {i}")
    addr = 0x80 + i * 32
    return bytes([_PUSH2, addr >> 8, addr & 0xFF, OP_MSTORE])


# ---------------------------------------------------------------------------
# Control flow instructions
# ---------------------------------------------------------------------------


def jump(target: int) -> bytes:
    """Emit PUSH2 target JUMP — unconditional jump."""
    return bytes([_PUSH2, target >> 8, target & 0xFF, _JUMP])


def jumpi(target: int) -> bytes:
    """Emit PUSH2 target JUMPI — conditional jump; pops condition from stack."""
    return bytes([_PUSH2, target >> 8, target & 0xFF, _JUMPI])


def revert_if(msg: str) -> bytes:
    """Pop condition; if non-zero, revert with ``Error(string)`` *msg* (≤32 bytes).

    Self-contained 101-byte PC-relative sequence.
    """
    raw = msg.encode()
    if len(raw) > 32:
        raise ValueError(f"revert_if: message too long ({len(raw)} bytes, max 32)")
    msglen = len(raw)
    msg_word = int.from_bytes(raw.ljust(32, b"\x00"), "big")
    selector_word = 0x08C379A000000000000000000000000000000000000000000000000000000000

    # 94-byte revert block: builds Error(string) and reverts
    revert_block = (
        bytes(
            [
                _PUSH1,
                0x40,  # PUSH1 0x40
                OP_MLOAD,  # MLOAD                   → [scratch]
                OP_PUSH_U256,  # PUSH32 selector
            ]
        )
        + selector_word.to_bytes(32, "big")
        + bytes(
            [
                _DUP2,  # DUP2                    → [scratch, sel, scratch]
                OP_MSTORE,  # MSTORE                  → [scratch]
                _PUSH1,
                0x20,  # PUSH1 0x20
                _DUP2,  # DUP2                    → [scratch, 32, scratch]
                _PUSH1,
                0x04,  # PUSH1 4
                OP_ADD,  # ADD                     → [scratch+4, 32, scratch]
                OP_MSTORE,  # MSTORE                  → [scratch]
                _PUSH1,
                msglen,  # PUSH1 msglen
                _DUP2,  # DUP2                    → [scratch, msglen, scratch]
                _PUSH1,
                0x24,  # PUSH1 0x24
                OP_ADD,  # ADD                     → [scratch+0x24, msglen, scratch]
                OP_MSTORE,  # MSTORE                  → [scratch]
                OP_PUSH_U256,  # PUSH32 msg_word
            ]
        )
        + msg_word.to_bytes(32, "big")
        + bytes(
            [
                _DUP2,  # DUP2                    → [scratch, msg_word, scratch]
                _PUSH1,
                0x44,  # PUSH1 0x44
                OP_ADD,  # ADD                     → [scratch+0x44, msg_word, scratch]
                OP_MSTORE,  # MSTORE                  → [scratch]
                _PUSH1,
                0x64,  # PUSH1 0x64 (100)
                OP_SWAP,  # SWAP1                   → [scratch, 100]
                OP_REVERT,  # REVERT
            ]
        )
    )
    assert len(revert_block) == 94

    # Full sequence: ISZERO PC PUSH1 99 ADD JUMPI <revert_block> JUMPDEST
    # PC at byte 1; JUMPDEST at byte 100; distance = 99
    return (
        bytes(
            [
                OP_ISZERO,  # ISZERO       byte 0
                _PC,  # PC           byte 1  (= instr_start + 1)
                _PUSH1,
                99,  # PUSH1 99     bytes 2-3
                OP_ADD,  # ADD          byte 4
                _JUMPI,  # JUMPI        byte 5
            ]
        )
        + revert_block
        + bytes([OP_JUMPDEST])
    )  # JUMPDEST  byte 100


def assert_ge(msg: str = "") -> bytes:
    """Pop *a* (TOS), *b* (2nd); revert if ``a < b``.  Stack effect: ``(a, b → )``."""
    raw = msg.encode()
    if len(raw) > 32:
        raise ValueError(f"assert_ge: message too long ({len(raw)} bytes, max 32)")
    # DUP2 DUP2 LT produces [a<b, a, b]; revert_if consumes [a<b]; then POP POP
    return bytes([_DUP2, _DUP2, OP_LT]) + revert_if(msg) + bytes([OP_POP, OP_POP])


def assert_le(msg: str = "") -> bytes:
    """Pop *a* (TOS), *b* (2nd); revert if ``a > b``.  Stack effect: ``(a, b → )``."""
    raw = msg.encode()
    if len(raw) > 32:
        raise ValueError(f"assert_le: message too long ({len(raw)} bytes, max 32)")
    return bytes([_DUP2, _DUP2, OP_GT]) + revert_if(msg) + bytes([OP_POP, OP_POP])


# ---------------------------------------------------------------------------
# Arithmetic / bitwise instructions (direct EVM opcodes)
# ---------------------------------------------------------------------------


def add() -> bytes:
    """Emit ADD."""
    return bytes([OP_ADD])


def mul() -> bytes:
    """Emit MUL."""
    return bytes([OP_MUL])


def sub() -> bytes:
    """Emit saturating SUB: ``max(a - b, 0)`` where *a* is TOS, *b* is 2nd.

    8-byte sequence: DUP2 DUP2 LT ISZERO SWAP2 SWAP1 SUB MUL
    """
    return bytes([_DUP2, _DUP2, OP_LT, OP_ISZERO, _SWAP2, OP_SWAP, _SUB_OP, OP_MUL])


def div() -> bytes:
    """Emit DIV."""
    return bytes([OP_DIV])


def mod() -> bytes:
    """Emit MOD."""
    return bytes([OP_MOD])


def lt() -> bytes:
    """Emit LT."""
    return bytes([OP_LT])


def gt() -> bytes:
    """Emit GT."""
    return bytes([OP_GT])


def eq() -> bytes:
    """Emit EQ."""
    return bytes([OP_EQ])


def iszero() -> bytes:
    """Emit ISZERO."""
    return bytes([OP_ISZERO])


def bitwise_and() -> bytes:
    """Emit AND."""
    return bytes([OP_AND])


def bitwise_or() -> bytes:
    """Emit OR."""
    return bytes([OP_OR])


def bitwise_xor() -> bytes:
    """Emit XOR."""
    return bytes([OP_XOR])


def bitwise_not() -> bytes:
    """Emit NOT."""
    return bytes([OP_NOT])


def shl() -> bytes:
    """Emit SHL."""
    return bytes([OP_SHL])


def shr() -> bytes:
    """Emit SHR."""
    return bytes([OP_SHR])


def self_addr() -> bytes:
    """Emit ADDRESS — push the deployed program's own address."""
    return bytes([OP_SELF_ADDR])


# ---------------------------------------------------------------------------
# External call
# ---------------------------------------------------------------------------


def call(require_success: bool = True) -> bytes:
    """Emit EVM CALL with optional PC-relative revert on failure.

    Stack before (TOS first): gas, addr, value, argsOffset, argsLen, retOffset, retLen.
    Stack after (require_success=False): [success].
    Stack after (require_success=True): [success] (reverts on failure; on success the CALL
    success flag remains on the stack).
    """
    if not require_success:
        return bytes([OP_CALL])
    # CALL DUP1 PC PUSH1 9 ADD JUMPI PUSH1 0 DUP1 REVERT JUMPDEST
    # PC at byte 2; JUMPDEST at byte 11; distance = 9
    return bytes(
        [
            OP_CALL,  # CALL          byte 0
            OP_DUP,  # DUP1          byte 1
            _PC,  # PC            byte 2  (= instr_start + 2)
            _PUSH1,
            9,  # PUSH1 9       bytes 3-4
            OP_ADD,  # ADD           byte 5
            _JUMPI,  # JUMPI         byte 6
            _PUSH1,
            0x00,  # PUSH1 0       bytes 7-8
            OP_DUP,  # DUP1          byte 9
            OP_REVERT,  # REVERT        byte 10
            OP_JUMPDEST,  # JUMPDEST      byte 11
        ]
    )


# ---------------------------------------------------------------------------
# Balance query
# ---------------------------------------------------------------------------


def balance_of() -> bytes:
    """Pop token (TOS) and account (2nd); push balance.

    If token == 0: EVM BALANCE(account).
    If token != 0: STATICCALL token.balanceOf(account).

    75-byte PC-relative sequence.
    """
    SELECTOR = 0x70A0823100000000000000000000000000000000000000000000000000000000

    # Preamble (7 bytes): if token==0 jump to ETH path at byte 71
    # PC at byte 2; ETH_PATH_JUMPDEST at byte 71; distance = 69
    preamble = bytes(
        [
            OP_DUP,  # DUP1         byte 0
            OP_ISZERO,  # ISZERO       byte 1
            _PC,  # PC           byte 2
            _PUSH1,
            69,  # PUSH1 69     bytes 3-4
            OP_ADD,  # ADD          byte 5
            _JUMPI,  # JUMPI        byte 6
        ]
    )

    # ERC-20 path (64 bytes, bytes 7-70)
    # PC at relative byte 59 → absolute byte 66; END_JUMPDEST at byte 74; distance = 8
    erc20_path = (
        bytes(
            [
                _PUSH1,
                0x40,  # PUSH1 0x40
                OP_MLOAD,  # MLOAD                         → [fp, token, account]
                OP_PUSH_U256,  # PUSH32 selector
            ]
        )
        + SELECTOR.to_bytes(32, "big")
        + bytes(
            [
                _DUP2,  # DUP2                          → [fp, sel, fp, token, account]
                OP_MSTORE,  # MSTORE   mem[fp]=sel          → [fp, token, account]
                _DUP3,  # DUP3                          → [account, fp, token, account]
                _DUP2,  # DUP2                          → [fp, account, fp, token, account]
                _PUSH1,
                0x04,  # PUSH1 4
                OP_ADD,  # ADD                           → [fp+4, account, fp, token, account]
                OP_MSTORE,  # MSTORE   mem[fp+4]=account    → [fp, token, account]
                # STATICCALL(gas, token, fp, 36, fp, 32)
                _PUSH1,
                0x20,  # PUSH1 0x20  retLen=32
                _DUP2,  # DUP2        retOff=fp
                _PUSH1,
                0x24,  # PUSH1 0x24  argsLen=36
                _DUP4,  # DUP4        argsOff=fp
                _DUP6,  # DUP6        addr=token
                OP_GAS,  # GAS
                OP_STATICCALL,  # STATICCALL  → [success, fp, token, account]
                OP_SWAP,  # SWAP1                         → [fp, success, token, account]
                OP_MLOAD,  # MLOAD                         → [balance, success, token, account]
                _SWAP3,  # SWAP3                         → [account, success, token, balance]
                OP_POP,  # POP                           → [success, token, balance]
                OP_POP,  # POP                           → [token, balance]
                OP_POP,  # POP                           → [balance]
                # Jump to END at byte 74
                _PC,  # PC    byte 66 (= 7 + 59)
                _PUSH1,
                8,  # PUSH1 8
                OP_ADD,  # ADD
                _JUMP,  # JUMP
            ]
        )
    )
    assert len(erc20_path) == 64

    # ETH path (bytes 71-73)
    eth_path = bytes(
        [
            OP_JUMPDEST,  # JUMPDEST  byte 71
            OP_POP,  # POP       byte 72  (remove token=0)
            OP_BALANCE,  # BALANCE   byte 73
        ]
    )

    # END (byte 74)
    end = bytes([OP_JUMPDEST])

    result = preamble + erc20_path + eth_path + end
    assert len(result) == 75
    return result


# ---------------------------------------------------------------------------
# Calldata patching
# ---------------------------------------------------------------------------


def patch_value(offset: int, size: int) -> bytes:
    """Overwrite a ``size``-byte value in the calldata buffer at *offset*.

    ABI right-aligns values shorter than 32 bytes within a 32-byte word, so the
    MSTORE target is ``offset + size - 32``.

    Args:
        offset: Byte offset of the value's first byte inside the calldata buffer.
        size:   Number of bytes occupied by the value.  Must satisfy
                ``0 < size <= 32``.

    Stack before: [value(TOS), argsOffset(2nd), argsLen(3rd), ...]
    Stack after:  [argsOffset(TOS), argsLen(2nd), ...]
    """
    if not (0 < size <= 32):
        raise ValueError(f"patch_value: size must be in (0, 32], got {size}")
    mstore_off = offset + size - 32
    if mstore_off < 0:
        raise ValueError(
            f"patch_value: offset {offset} is too small for size {size}; MSTORE target {mstore_off} would be negative"
        )
    if mstore_off > 0xFFFF:
        raise ValueError(f"patch_value: mstore offset {mstore_off} exceeds 16-bit PUSH2 range")
    return bytes([_DUP2, _PUSH2, mstore_off >> 8, mstore_off & 0xFF, OP_ADD, OP_MSTORE])


def ret_u256(offset: int) -> bytes:
    """Copy 32 bytes from returndata at *offset* into free memory; push the value.

    11-byte sequence.
    """
    return bytes(
        [
            _PUSH1,
            0x40,  # PUSH1 0x40
            OP_MLOAD,  # MLOAD         → [fp]
            _PUSH1,
            0x20,  # PUSH1 0x20    → [32, fp]
            _PUSH2,
            offset >> 8,
            offset & 0xFF,  # PUSH2 offset  → [offset, 32, fp]
            _DUP3,  # DUP3          → [fp, offset, 32, fp]
            OP_RETURNDATACOPY,  # RETURNDATACOPY → [fp]
            OP_MLOAD,  # MLOAD         → [value]
        ]
    )


def ret_slice(offset: int, length: int) -> bytes:
    """Copy a slice from returndata into free memory; push ``[argsOffset, argsLen]``.

    23-byte sequence.
    """
    length_padded = (length + 31) & ~31
    return bytes(
        [
            _PUSH1,
            0x40,  # PUSH1 0x40
            OP_MLOAD,  # MLOAD              → [fp]
            OP_DUP,  # DUP1               → [fp, fp]
            _PUSH2,
            length >> 8,
            length & 0xFF,  # PUSH2 length
            _PUSH2,
            offset >> 8,
            offset & 0xFF,  # PUSH2 offset
            _DUP4,  # DUP4               → [fp, offset, length, fp, fp]
            OP_RETURNDATACOPY,  # RETURNDATACOPY     → [fp, fp]
            _PUSH2,
            length_padded >> 8,
            length_padded & 0xFF,
            OP_ADD,  # ADD                → [new_fp, fp]
            _PUSH1,
            0x40,  # PUSH1 0x40
            OP_MSTORE,  # MSTORE             → [fp]
            _PUSH2,
            length >> 8,
            length & 0xFF,  # PUSH2 length
            OP_SWAP,  # SWAP1              → [fp=argsOffset, length=argsLen]
        ]
    )
