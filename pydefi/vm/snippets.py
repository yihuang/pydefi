from eth_abi import encode

from .program import FREE_MEMORY_POINTER, Program

# Error(string) selector
ERROR_SELECTOR = b'\x08\xc3y\xa0'


def push_scratch(data: bytes) -> Program:
    '''
    copy bytes to scatch space, and leave [argsOffset(TOS), argsLen(2nd)] on stack.
    '''
    assert len(data) <= 64, f"push_scratch: data too long ({len(data)} bytes, max 64)"

    p = Program()
    p.push(int.from_bytes(data[:32], "big")).push(0x00).mstore()
    if len(data) > 32:
        p.push(int.from_bytes(data[32:], "big")).push(0x20).mstore()

    p.push(0).push(len(data))  # [0, len(data)]
    return p


def push_bytes(data: bytes, update_fp: bool = False) -> Program:
    '''
    copy bytes to free memory, optionally updating the free memory pointer,
    and leave [argsOffset(TOS), argsLen(2nd)] on stack.
    '''
    assert len(data) <= 0xFFFF, f"push_bytes_noupdate: data too large ({len(data)} bytes, max 65535)"

    blen = len(data)
    blen_padded = (blen + 31) & ~31  # round up to nearest multiple of 32

    p = Program()
    p.push(blen) # [blen]
    p.push(FREE_MEMORY_POINTER).mload()  # [max_fp, blen]

    # Store each 32-byte chunk; max_fp stays at TOS between iterations.
    for i in range(0, blen_padded, 32):
        val = int.from_bytes(data[i : i + 32], "big")

        p.push(val).dup2()  # [max_fp, chunk, max_fp, blen]
        if i > 0:
            p.push(i).add()  # [max_fp+i, chunk, max_fp, blen]
        p.mstore()  # mem[max_fp+i]=chunk; [max_fp, blen]

    if update_fp:
        # update pointer, stack unchanged
        p.dup().push(blen_padded).add().push(FREE_MEMORY_POINTER).mstore()

    return p


def push_tmp(data: bytes) -> Program:
    if len(data) <= 64:
        return push_scratch(data)
    else:
        return push_bytes(data, update_fp=False)


def revert_if(msg: str) -> Program:
    '''
    if TOS is zero, revert Error(msg); otherwise continue execution.  Stack effect: (flag → ).
    '''
    data = ERROR_SELECTOR + encode(['string'], [msg])
    err_section = push_tmp(data).revert()
    jump_offset = len(err_section.assemble())+4

    p = Program()
    p.iszero()
    p.pc().push(jump_offset).add()
    p.jumpi()  # if TOS != 0 jump to end; otherwise continue with revert sequence
    p.extend(err_section)  # [argsOffset, argsLen]
    p.jumpdest()
    return p


def assert_ge(n: int, msg: str) -> bytes:
    """
    assert TOS >= n, where n is a non-negative integer fitting in 32 bytes; revert with *msg* if not.
    pop the TOS after the call.
    """
    p = Program()
    p.push(n).lt()  # if TOS < n, flag=1; else flag=0
    p.extend(revert_if(msg))  # if flag=1 revert with msg; else continue
    return p


def assert_le(msg: str = "") -> bytes:
    """Pop *a* (TOS), *b* (2nd); revert if ``a > b``.  Stack effect: ``(a, b → )``."""
    raw = msg.encode()
    if len(raw) > 32:
        raise ValueError(f"assert_le: message too long ({len(raw)} bytes, max 32)")
    return bytes([_DUP2, _DUP2, OP_GT]) + revert_if(msg) + bytes([OP_POP, OP_POP])


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
