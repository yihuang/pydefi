"""DeFiVM program builder — Python DSL for assembling DeFiVM EVM bytecode.

Programs are raw EVM bytecode that execute on the native EVM stack.
``execute()`` in DeFiVM.sol deploys the bytecode as a contract via CREATE
and then calls it forwarding all ETH.  The native EVM stack IS the VM stack.

Memory conventions in deployed programs
-----------------------------------------
- Registers:          ``memory[0x80 + i*32]`` for ``i`` in 0..15
- Free memory pointer: ``memory[0x40]`` (0 in a fresh deployed contract)
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
OP_CALL: int = 0xF1  # CALL — core opcode in the call() sequence
OP_SELF_ADDR: int = 0x30  # ADDRESS — emitted by self_addr()

# ---------------------------------------------------------------------------
# Stack instructions
# ---------------------------------------------------------------------------


def push_u256(n: int) -> bytes:
    """Emit PUSH32 — push a uint256 literal onto the native EVM stack."""
    if n < 0:
        raise ValueError(f"push_u256: value must be non-negative, got {n}")
    return bytes([0x7F]) + n.to_bytes(32, "big")


def push_addr(a: str) -> bytes:
    """Emit PUSH20 — push a 20-byte Ethereum address onto the native EVM stack."""
    raw = bytes.fromhex(a.removeprefix("0x"))
    if len(raw) != 20:
        raise ValueError(f"push_addr: bad address length: {a!r}")
    return bytes([0x73]) + raw


def gas_opcode() -> bytes:
    """Emit GAS — push the remaining gas onto the stack."""
    return bytes([0x5A])


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
            return bytes([0x60, v])  # PUSH1
        elif v <= 0xFFFF:
            return bytes([0x61, v >> 8, v & 0xFF])  # PUSH2
        else:
            n = (v.bit_length() + 7) // 8
            return bytes([0x5F + n]) + v.to_bytes(n, "big")

    # Compute max_fp = fp | (0x280 * (fp == 0))
    code = bytes(
        [
            0x60,
            0x40,  # PUSH1 0x40
            0x51,  # MLOAD           → [fp]
            0x61,
            0x02,
            0x80,  # PUSH2 0x0280    → [0x280, fp]
            0x81,  # DUP2            → [fp, 0x280, fp]
            0x15,  # ISZERO          → [fp==0, 0x280, fp]
            0x02,  # MUL             → [0x280*(fp==0), fp]
            0x17,  # OR              → [max_fp]
        ]
    )

    # Store each 32-byte chunk; max_fp stays at TOS between iterations.
    for i in range(0, blen_padded, 32):
        chunk = data[i : i + 32].ljust(32, b"\x00")
        chunk_val = int.from_bytes(chunk, "big")

        # Duplicate max_fp, then compute store address = max_fp + i.
        code += bytes([0x80])  # DUP1 → [max_fp, max_fp, ...]
        if i > 0:
            code += _push_imm(i)  # [i, max_fp, max_fp]
            code += bytes([0x01])  # ADD → [max_fp+i, max_fp]

        # Push the 32-byte chunk (use PUSH1 0 for all-zero chunks).
        if chunk_val == 0:
            code += bytes([0x60, 0x00])  # PUSH1 0
        else:
            code += bytes([0x7F]) + chunk  # PUSH32 chunk

        code += bytes([0x90, 0x52])  # SWAP1, MSTORE → mem[max_fp+i]=chunk; [max_fp]

    # Update free-memory pointer: mem[0x40] = max_fp + blen_padded.
    code += bytes([0x80])  # DUP1 → [max_fp, max_fp]
    code += _push_imm(blen_padded)  # [blen_padded, max_fp, max_fp]
    code += bytes([0x01])  # ADD → [new_fp, max_fp]
    code += bytes([0x60, 0x40])  # PUSH1 0x40
    code += bytes([0x52])  # MSTORE → mem[0x40] = new_fp; [max_fp]

    # Leave [argsOffset(TOS), argsLen(2nd)].
    code += _push_imm(blen)  # [blen, max_fp]
    code += bytes([0x90])  # SWAP1 → [max_fp=argsOffset, blen=argsLen]
    return code


def dup() -> bytes:
    """Emit DUP1 — duplicate the top stack item."""
    return bytes([0x80])


def swap() -> bytes:
    """Emit SWAP1 — exchange the top two stack items."""
    return bytes([0x90])


def pop() -> bytes:
    """Emit POP — discard the top stack item."""
    return bytes([0x50])


def load_reg(i: int) -> bytes:
    """Emit PUSH2 addr MLOAD — push register *i* onto the stack."""
    if not 0 <= i <= 15:
        raise ValueError(f"load_reg: register index must be 0..15, got {i}")
    addr = 0x80 + i * 32
    return bytes([0x61, addr >> 8, addr & 0xFF, 0x51])


def store_reg(i: int) -> bytes:
    """Emit PUSH2 addr MSTORE — pop TOS into register *i*."""
    if not 0 <= i <= 15:
        raise ValueError(f"store_reg: register index must be 0..15, got {i}")
    addr = 0x80 + i * 32
    return bytes([0x61, addr >> 8, addr & 0xFF, 0x52])


# ---------------------------------------------------------------------------
# Control flow instructions
# ---------------------------------------------------------------------------


def jump(target: int) -> bytes:
    """Emit PUSH2 target JUMP — unconditional jump."""
    return bytes([0x61, target >> 8, target & 0xFF, 0x56])


def jumpi(target: int) -> bytes:
    """Emit PUSH2 target JUMPI — conditional jump; pops condition from stack."""
    return bytes([0x61, target >> 8, target & 0xFF, 0x57])


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
                0x60,
                0x40,  # PUSH1 0x40
                0x51,  # MLOAD                   → [scratch]
                0x7F,  # PUSH32 selector
            ]
        )
        + selector_word.to_bytes(32, "big")
        + bytes(
            [
                0x81,  # DUP2                    → [scratch, sel, scratch]
                0x52,  # MSTORE                  → [scratch]
                0x60,
                0x20,  # PUSH1 0x20
                0x81,  # DUP2                    → [scratch, 32, scratch]
                0x60,
                0x04,  # PUSH1 4
                0x01,  # ADD                     → [scratch+4, 32, scratch]
                0x52,  # MSTORE                  → [scratch]
                0x60,
                msglen,  # PUSH1 msglen
                0x81,  # DUP2                    → [scratch, msglen, scratch]
                0x60,
                0x24,  # PUSH1 0x24
                0x01,  # ADD                     → [scratch+0x24, msglen, scratch]
                0x52,  # MSTORE                  → [scratch]
                0x7F,  # PUSH32 msg_word
            ]
        )
        + msg_word.to_bytes(32, "big")
        + bytes(
            [
                0x81,  # DUP2                    → [scratch, msg_word, scratch]
                0x60,
                0x44,  # PUSH1 0x44
                0x01,  # ADD                     → [scratch+0x44, msg_word, scratch]
                0x52,  # MSTORE                  → [scratch]
                0x60,
                0x64,  # PUSH1 0x64 (100)
                0x90,  # SWAP1                   → [scratch, 100]
                0xFD,  # REVERT
            ]
        )
    )
    assert len(revert_block) == 94

    # Full sequence: ISZERO PC PUSH1 99 ADD JUMPI <revert_block> JUMPDEST
    # PC at byte 1; JUMPDEST at byte 100; distance = 99
    return (
        bytes(
            [
                0x15,  # ISZERO       byte 0
                0x58,  # PC           byte 1  (= instr_start + 1)
                0x60,
                99,  # PUSH1 99     bytes 2-3
                0x01,  # ADD          byte 4
                0x57,  # JUMPI        byte 5
            ]
        )
        + revert_block
        + bytes([0x5B])
    )  # JUMPDEST  byte 100


def assert_ge(msg: str = "") -> bytes:
    """Pop *a* (TOS), *b* (2nd); revert if ``a < b``.  Stack effect: ``(a, b → )``."""
    raw = msg.encode()
    if len(raw) > 32:
        raise ValueError(f"assert_ge: message too long ({len(raw)} bytes, max 32)")
    # DUP2 DUP2 LT produces [a<b, a, b]; revert_if consumes [a<b]; then POP POP
    return bytes([0x81, 0x81, 0x10]) + revert_if(msg) + bytes([0x50, 0x50])


def assert_le(msg: str = "") -> bytes:
    """Pop *a* (TOS), *b* (2nd); revert if ``a > b``.  Stack effect: ``(a, b → )``."""
    raw = msg.encode()
    if len(raw) > 32:
        raise ValueError(f"assert_le: message too long ({len(raw)} bytes, max 32)")
    return bytes([0x81, 0x81, 0x11]) + revert_if(msg) + bytes([0x50, 0x50])


# ---------------------------------------------------------------------------
# Arithmetic / bitwise instructions (direct EVM opcodes)
# ---------------------------------------------------------------------------


def add() -> bytes:
    """Emit ADD."""
    return bytes([0x01])


def mul() -> bytes:
    """Emit MUL."""
    return bytes([0x02])


def sub() -> bytes:
    """Emit saturating SUB: ``max(a - b, 0)`` where *a* is TOS, *b* is 2nd.

    8-byte sequence: DUP2 DUP2 LT ISZERO SWAP2 SWAP1 SUB MUL
    """
    return bytes([0x81, 0x81, 0x10, 0x15, 0x91, 0x90, 0x03, 0x02])


def div() -> bytes:
    """Emit DIV."""
    return bytes([0x04])


def mod() -> bytes:
    """Emit MOD."""
    return bytes([0x06])


def lt() -> bytes:
    """Emit LT."""
    return bytes([0x10])


def gt() -> bytes:
    """Emit GT."""
    return bytes([0x11])


def eq() -> bytes:
    """Emit EQ."""
    return bytes([0x14])


def iszero() -> bytes:
    """Emit ISZERO."""
    return bytes([0x15])


def bitwise_and() -> bytes:
    """Emit AND."""
    return bytes([0x16])


def bitwise_or() -> bytes:
    """Emit OR."""
    return bytes([0x17])


def bitwise_xor() -> bytes:
    """Emit XOR."""
    return bytes([0x18])


def bitwise_not() -> bytes:
    """Emit NOT."""
    return bytes([0x19])


def shl() -> bytes:
    """Emit SHL."""
    return bytes([0x1B])


def shr() -> bytes:
    """Emit SHR."""
    return bytes([0x1C])


def self_addr() -> bytes:
    """Emit ADDRESS — push the deployed program's own address."""
    return bytes([0x30])


# ---------------------------------------------------------------------------
# External call
# ---------------------------------------------------------------------------


def call(require_success: bool = True) -> bytes:
    """Emit EVM CALL with optional PC-relative revert on failure.

    Stack before (TOS first): gas, addr, value, argsOffset, argsLen, retOffset, retLen.
    Stack after (require_success=False): [success].
    Stack after (require_success=True): [] (reverts on failure, otherwise no return value).
    """
    if not require_success:
        return bytes([0xF1])
    # CALL DUP1 PC PUSH1 9 ADD JUMPI PUSH1 0 DUP1 REVERT JUMPDEST
    # PC at byte 2; JUMPDEST at byte 11; distance = 9
    return bytes(
        [
            0xF1,  # CALL          byte 0
            0x80,  # DUP1          byte 1
            0x58,  # PC            byte 2  (= instr_start + 2)
            0x60,
            9,  # PUSH1 9       bytes 3-4
            0x01,  # ADD           byte 5
            0x57,  # JUMPI         byte 6
            0x60,
            0x00,  # PUSH1 0       bytes 7-8
            0x80,  # DUP1          byte 9
            0xFD,  # REVERT        byte 10
            0x5B,  # JUMPDEST      byte 11
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
            0x80,  # DUP1         byte 0
            0x15,  # ISZERO       byte 1
            0x58,  # PC           byte 2
            0x60,
            69,  # PUSH1 69     bytes 3-4
            0x01,  # ADD          byte 5
            0x57,  # JUMPI        byte 6
        ]
    )

    # ERC-20 path (64 bytes, bytes 7-70)
    # PC at relative byte 59 → absolute byte 66; END_JUMPDEST at byte 74; distance = 8
    erc20_path = (
        bytes(
            [
                0x60,
                0x40,  # PUSH1 0x40
                0x51,  # MLOAD                         → [fp, token, account]
                0x7F,  # PUSH32 selector
            ]
        )
        + SELECTOR.to_bytes(32, "big")
        + bytes(
            [
                0x81,  # DUP2                          → [fp, sel, fp, token, account]
                0x52,  # MSTORE   mem[fp]=sel          → [fp, token, account]
                0x82,  # DUP3                          → [account, fp, token, account]
                0x81,  # DUP2                          → [fp, account, fp, token, account]
                0x60,
                0x04,  # PUSH1 4
                0x01,  # ADD                           → [fp+4, account, fp, token, account]
                0x52,  # MSTORE   mem[fp+4]=account    → [fp, token, account]
                # STATICCALL(gas, token, fp, 36, fp, 32)
                0x60,
                0x20,  # PUSH1 0x20  retLen=32
                0x81,  # DUP2        retOff=fp
                0x60,
                0x24,  # PUSH1 0x24  argsLen=36
                0x83,  # DUP4        argsOff=fp
                0x85,  # DUP6        addr=token
                0x5A,  # GAS
                0xFA,  # STATICCALL  → [success, fp, token, account]
                0x90,  # SWAP1                         → [fp, success, token, account]
                0x51,  # MLOAD                         → [balance, success, token, account]
                0x92,  # SWAP3                         → [account, success, token, balance]
                0x50,  # POP                           → [success, token, balance]
                0x50,  # POP                           → [token, balance]
                0x50,  # POP                           → [balance]
                # Jump to END at byte 74
                0x58,  # PC    byte 66 (= 7 + 59)
                0x60,
                8,  # PUSH1 8
                0x01,  # ADD
                0x56,  # JUMP
            ]
        )
    )
    assert len(erc20_path) == 64

    # ETH path (bytes 71-73)
    eth_path = bytes(
        [
            0x5B,  # JUMPDEST  byte 71
            0x50,  # POP       byte 72  (remove token=0)
            0x31,  # BALANCE   byte 73
        ]
    )

    # END (byte 74)
    end = bytes([0x5B])

    result = preamble + erc20_path + eth_path + end
    assert len(result) == 75
    return result


# ---------------------------------------------------------------------------
# Calldata patching
# ---------------------------------------------------------------------------


def patch_u256(offset: int) -> bytes:
    """Overwrite a 32-byte word in the calldata buffer at *offset*.

    Stack before: [value(TOS), argsOffset(2nd), argsLen(3rd), ...]
    Stack after:  [argsOffset(TOS), argsLen(2nd), ...]
    """
    return bytes([0x81, 0x61, offset >> 8, offset & 0xFF, 0x01, 0x52])


def patch_addr(offset: int) -> bytes:
    """Overwrite a 20-byte address in the calldata buffer at *offset*.

    ABI places the 20-byte address at ``[offset..offset+19]``, so the
    32-byte MSTORE target is at ``offset - 12``.

    Stack before: [addr(TOS), argsOffset(2nd), argsLen(3rd), ...]
    Stack after:  [argsOffset(TOS), argsLen(2nd), ...]
    """
    mstore_off = offset - 12
    return bytes([0x81, 0x61, mstore_off >> 8, mstore_off & 0xFF, 0x01, 0x52])


# ---------------------------------------------------------------------------
# Returndata helpers
# ---------------------------------------------------------------------------


def ret_u256(offset: int) -> bytes:
    """Copy 32 bytes from returndata at *offset* into free memory; push the value.

    11-byte sequence.
    """
    return bytes(
        [
            0x60,
            0x40,  # PUSH1 0x40
            0x51,  # MLOAD         → [fp]
            0x60,
            0x20,  # PUSH1 0x20    → [32, fp]
            0x61,
            offset >> 8,
            offset & 0xFF,  # PUSH2 offset  → [offset, 32, fp]
            0x82,  # DUP3          → [fp, offset, 32, fp]
            0x3E,  # RETURNDATACOPY → [fp]
            0x51,  # MLOAD         → [value]
        ]
    )


def ret_slice(offset: int, length: int) -> bytes:
    """Copy a slice from returndata into free memory; push ``[argsOffset, argsLen]``.

    23-byte sequence.
    """
    length_padded = (length + 31) & ~31
    return bytes(
        [
            0x60,
            0x40,  # PUSH1 0x40
            0x51,  # MLOAD              → [fp]
            0x80,  # DUP1               → [fp, fp]
            0x61,
            length >> 8,
            length & 0xFF,  # PUSH2 length
            0x61,
            offset >> 8,
            offset & 0xFF,  # PUSH2 offset
            0x83,  # DUP4               → [fp, offset, length, fp, fp]
            0x3E,  # RETURNDATACOPY     → [fp, fp]
            0x61,
            length_padded >> 8,
            length_padded & 0xFF,
            0x01,  # ADD                → [new_fp, fp]
            0x60,
            0x40,  # PUSH1 0x40
            0x52,  # MSTORE             → [fp]
            0x61,
            length >> 8,
            length & 0xFF,  # PUSH2 length
            0x90,  # SWAP1              → [fp=argsOffset, length=argsLen]
        ]
    )
