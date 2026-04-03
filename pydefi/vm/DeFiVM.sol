// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title DeFiVM
 * @notice A minimal, register-based macro-assembler / interpreter for composable DeFi flows.
 *
 * Design principles
 * -----------------
 *  * Atomic execution  - a "program" runs all-at-once; any revert undoes everything.
 *  * Register-based    - 16 named registers (R0-R15) plus a temporary 32-element stack.
 *  * Fully stateless   - no owner, no whitelist; the CALL opcode can reach any address.
 *
 * Implementation notes
 * --------------------
 *  The interpreter is written entirely in Yul (inline assembly).  All execution
 *  state (stack, registers, buffer table) lives in raw memory regions managed
 *  through direct MSTORE/MLOAD.  Every opcode delegates to the corresponding
 *  native EVM opcode (add, mul, lt, call, ...) so there is no emulation overhead.
 *
 * Memory layout (allocated once at the start of execute())
 * ---------------------------------------------------------
 *  [stackBase .. stackBase + 0x400)  - VM stack: 32 x 32-byte slots
 *  [regsBase  .. regsBase  + 0x200)  - VM registers: 16 x 32-byte slots
 *  [bufsBase  .. bufsBase  + 0x200)  - Buffer pointer table: 16 x 32-byte slots
 *                                      Each slot holds a memory pointer to a
 *                                      length-prefixed byte buffer: [len][data...]
 *
 * Security assumptions
 * --------------------
 *  1. Never approve tokens directly to this contract.  Approvals can be drained by
 *     any caller because `execute` is permissionless.  Use permit signatures instead
 *     (approve and spend atomically inside the program).
 *  2. Do not leave token or ETH balances in this contract between transactions.
 *     Any residual balance is accessible to arbitrary programs.
 *  3. Users must verify every adapter address they include in a program and simulate
 *     the full transaction off-chain before broadcasting.
 *
 * Instruction set
 * ---------------
 * Stack / Register
 *   0x01  PUSH_U256  <32 bytes>          push uint256
 *   0x02  PUSH_ADDR  <20 bytes>          push address
 *   0x03  PUSH_BYTES <2-byte len> <data> push bytes blob (stored in buffer array)
 *   0x04  DUP                            duplicate top of stack
 *   0x05  SWAP                           swap top two items
 *   0x06  POP                            discard top
 *   0x10  LOAD_REG   <1-byte i>          push register[i]
 *   0x11  STORE_REG  <1-byte i>          pop -> register[i]
 *
 * Control flow
 *   0x20  JUMP       <2-byte target>     unconditional jump
 *   0x21  JUMPI      <2-byte target>     jump if top-of-stack != 0
 *   0x22  REVERT_IF  <1-byte msgLen>     revert with msg if top != 0
 *   0x23  ASSERT_GE  <1-byte msgLen>     pop a, b -> revert if a < b  (a >= b required)
 *   0x24  ASSERT_LE  <1-byte msgLen>     pop a, b -> revert if a > b  (a <= b required)
 *
 * External / introspection
 *   0x30  CALL       <1-byte flags>      pop: gasLimit, to, value, calldataBufIdx -> push success
 *                                        flags bit-0: require success
 *   0x31  BALANCE_OF                     pop: token (0x0=ETH), account -> push balance
 *   0x32  SELF_ADDR                      push address(this)
 *   0x33  SUB                            pop a, b -> push a - b  (saturates to 0 if a < b)
 *   0x34  ADD                            pop a, b -> push a + b  (wrapping uint256)
 *   0x35  MUL                            pop a, b -> push a * b  (wrapping uint256)
 *   0x36  DIV                            pop a, b -> push a / b  (0 if b == 0)
 *   0x37  MOD                            pop a, b -> push a % b  (0 if b == 0)
 *   0x38  LT                             pop a (TOS), b -> push 1 if a < b else 0
 *   0x39  GT                             pop a (TOS), b -> push 1 if a > b else 0
 *   0x3a  EQ                             pop a (TOS), b -> push 1 if a == b else 0
 *   0x3b  ISZERO                         pop a -> push 1 if a == 0 else 0
 *   0x3c  AND                            pop a (TOS), b -> push a & b
 *   0x3d  OR                             pop a (TOS), b -> push a | b
 *   0x3e  XOR                            pop a (TOS), b -> push a ^ b
 *   0x3f  NOT                            pop a -> push ~a
 *   0x44  SHL                            pop shift (TOS), value -> push value << shift (EVM SHL)
 *   0x45  SHR                            pop shift (TOS), value -> push value >> shift (EVM SHR)
 *
 * ABI / data
 *   0x40  PATCH_U256 <2-byte offset>     pop: value, bufIdx -> patch 32-byte word in buffer
 *   0x41  PATCH_ADDR <2-byte offset>     pop: addr,  bufIdx -> patch 20-byte word in buffer
 *   0x42  RET_U256   <2-byte offset>     push uint256 from last returndata at offset
 *   0x43  RET_SLICE  <2-byte off> <2-byte len>  push bytes slice from last returndata
 */
contract DeFiVM {
    /// @notice Allow the VM to receive ETH (needed for value-bearing calls).
    receive() external payable {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Execute a DeFiVM program atomically.
     * @param program  Bytecode stream (packed instructions).
     *
     * The entire interpreter runs in a single Yul assembly block.
     * The VM stack, register file, and buffer pointer table are contiguous
     * memory regions addressed directly with MSTORE/MLOAD.  Every arithmetic,
     * comparison, and logic opcode delegates straight to the corresponding EVM
     * opcode - no emulation overhead.  External calls use the native EVM CALL
     * instruction.  Any revert undoes all side-effects.
     */
    function execute(bytes calldata program) external payable {
        // Copy calldata to memory once so every read is a cheap MLOAD.
        bytes memory prog = program;

        assembly {
            // ----------------------------------------------------------------
            // Memory layout
            // ----------------------------------------------------------------
            //  stackBase : 32 slots x 32 bytes  (0x400 bytes)
            //  regsBase  : 16 slots x 32 bytes  (0x200 bytes)
            //  bufsBase  : 16 slots x 32 bytes  (0x200 bytes)
            //              each slot = pointer to [len (32B)][data...]
            let stackBase := mload(0x40)
            let regsBase  := add(stackBase, 0x400)
            let bufsBase  := add(regsBase,  0x200)
            mstore(0x40,  add(bufsBase, 0x200))

            let progData  := add(prog, 32)   // pointer to first program byte
            let plen      := mload(prog)      // total program length

            // Interpreter state kept on the native EVM stack as Yul variables
            let pc       := 0
            let sp       := 0   // VM stack depth (items)
            let numBufs  := 0   // allocated byte buffer count
            let retPtr   := 0   // pointer to last returndata: [len (32B)][data...]

            // ----------------------------------------------------------------
            // Helper: encode Error(string) and revert
            // msgSrc: pointer to raw bytes; msgLen: length in bytes
            // ----------------------------------------------------------------
            function revertWithMsg(msgSrc, msgLen) {
                let base := mload(0x40)
                // Error(string) selector = 0x08c379a0
                mstore(base,
                    0x08c379a000000000000000000000000000000000000000000000000000000000)
                mstore(add(base, 4),  32)
                mstore(add(base, 36), msgLen)
                let dst := add(base, 68)
                for { let i := 0 } lt(i, msgLen) { i := add(i, 32) } {
                    mstore(add(dst, i), mload(add(msgSrc, i)))
                }
                revert(base, add(68, msgLen))
            }

            // ----------------------------------------------------------------
            // Helper: revert with a static string literal (<=32 bytes)
            // word: right-zero-padded 32-byte value; len: byte count
            // ----------------------------------------------------------------
            function revertStatic(word, len) {
                let base := mload(0x40)
                mstore(base,
                    0x08c379a000000000000000000000000000000000000000000000000000000000)
                mstore(add(base, 4),  32)
                mstore(add(base, 36), len)
                mstore(add(base, 68), word)
                revert(base, 100)
            }

            // ----------------------------------------------------------------
            // Helper: allocate a length-prefixed buffer [len][data] in free memory
            // Returns: pointer to the start of the buffer.
            // The allocation is padded to a 32-byte boundary so word-copy loops
            // (used in PUSH_BYTES, RET_SLICE) can safely over-read the last
            // partial word without writing into unrelated allocations.
            //
            // Alignment math:
            //   raw end = bufPtr + 32 (length word) + dataLen
            //   round up to 32 = (raw end + 31) & ~31
            // ----------------------------------------------------------------
            function allocBuf(dataLen) -> bufPtr {
                bufPtr := mload(0x40)
                // (bufPtr + 32 + dataLen + 31) & ~31  → next 32-byte boundary
                mstore(0x40, and(add(add(bufPtr, add(dataLen, 32)), 31), not(31)))
                mstore(bufPtr, dataLen)
            }

            // ----------------------------------------------------------------
            // Main dispatch loop
            // ----------------------------------------------------------------
            for {} lt(pc, plen) {} {
                let op := byte(0, mload(add(progData, pc)))
                pc     := add(pc, 1)

                switch op

                // ── 0x01  PUSH_U256 ─────────────────────────────────────
                case 0x01 {
                    let v := mload(add(progData, pc))
                    pc    := add(pc, 32)
                    mstore(add(stackBase, shl(5, sp)), v)
                    sp    := add(sp, 1)
                }

                // ── 0x02  PUSH_ADDR ─────────────────────────────────────
                // Address is stored big-endian in 20 bytes.  Reading 32 bytes
                // puts those 20 bytes in the HIGH 20 bytes of the word; shifting
                // right by 96 bits (12 bytes) leaves the address in the low 20.
                case 0x02 {
                    let raw := mload(add(progData, pc))
                    pc      := add(pc, 20)
                    mstore(add(stackBase, shl(5, sp)), shr(96, raw))
                    sp      := add(sp, 1)
                }

                // ── 0x03  PUSH_BYTES ────────────────────────────────────
                case 0x03 {
                    let blen   := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    let bufPtr := allocBuf(blen)
                    let src    := add(progData, pc)
                    let dst    := add(bufPtr, 32)
                    // Word-by-word copy; allocBuf padded the region so the
                    // final over-read word writes into reserved space only.
                    for { let i := 0 } lt(i, blen) { i := add(i, 32) } {
                        mstore(add(dst, i), mload(add(src, i)))
                    }
                    pc := add(pc, blen)
                    mstore(add(bufsBase, shl(5, numBufs)), bufPtr)
                    mstore(add(stackBase, shl(5, sp)), numBufs)
                    sp      := add(sp, 1)
                    numBufs := add(numBufs, 1)
                }

                // ── 0x04  DUP ───────────────────────────────────────────
                case 0x04 {
                    let v := mload(add(stackBase, shl(5, sub(sp, 1))))
                    mstore(add(stackBase, shl(5, sp)), v)
                    sp := add(sp, 1)
                }

                // ── 0x05  SWAP ──────────────────────────────────────────
                case 0x05 {
                    let ai := shl(5, sub(sp, 1))
                    let bi := shl(5, sub(sp, 2))
                    let av := mload(add(stackBase, ai))
                    let bv := mload(add(stackBase, bi))
                    mstore(add(stackBase, ai), bv)
                    mstore(add(stackBase, bi), av)
                }

                // ── 0x06  POP ───────────────────────────────────────────
                case 0x06 {
                    sp := sub(sp, 1)
                }

                // ── 0x10  LOAD_REG ──────────────────────────────────────
                case 0x10 {
                    let i := byte(0, mload(add(progData, pc)))
                    pc    := add(pc, 1)
                    let v := mload(add(regsBase, shl(5, i)))
                    mstore(add(stackBase, shl(5, sp)), v)
                    sp    := add(sp, 1)
                }

                // ── 0x11  STORE_REG ─────────────────────────────────────
                case 0x11 {
                    let i := byte(0, mload(add(progData, pc)))
                    pc    := add(pc, 1)
                    sp    := sub(sp, 1)
                    let v := mload(add(stackBase, shl(5, sp)))
                    mstore(add(regsBase, shl(5, i)), v)
                }

                // ── 0x20  JUMP ──────────────────────────────────────────
                case 0x20 {
                    pc := shr(240, mload(add(progData, pc)))
                }

                // ── 0x21  JUMPI ─────────────────────────────────────────
                case 0x21 {
                    let target := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    sp         := sub(sp, 1)
                    let cond   := mload(add(stackBase, shl(5, sp)))
                    if cond { pc := target }
                }

                // ── 0x22  REVERT_IF ─────────────────────────────────────
                case 0x22 {
                    let msgLen := byte(0, mload(add(progData, pc)))
                    pc         := add(pc, 1)
                    sp         := sub(sp, 1)
                    if mload(add(stackBase, shl(5, sp))) {
                        revertWithMsg(add(progData, pc), msgLen)
                    }
                    pc := add(pc, msgLen)
                }

                // ── 0x23  ASSERT_GE ─────────────────────────────────────
                // pop a (TOS), pop b; revert if a < b  (require a >= b)
                case 0x23 {
                    let msgLen := byte(0, mload(add(progData, pc)))
                    pc         := add(pc, 1)
                    sp         := sub(sp, 2)
                    let a      := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b      := mload(add(stackBase, shl(5, sp)))
                    if lt(a, b) { revertWithMsg(add(progData, pc), msgLen) }
                    pc := add(pc, msgLen)
                }

                // ── 0x24  ASSERT_LE ─────────────────────────────────────
                // pop a (TOS), pop b; revert if a > b  (require a <= b)
                case 0x24 {
                    let msgLen := byte(0, mload(add(progData, pc)))
                    pc         := add(pc, 1)
                    sp         := sub(sp, 2)
                    let a      := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b      := mload(add(stackBase, shl(5, sp)))
                    if gt(a, b) { revertWithMsg(add(progData, pc), msgLen) }
                    pc := add(pc, msgLen)
                }

                // ── 0x30  CALL ──────────────────────────────────────────
                // pop: gasLimit (TOS), to, value, calldataBufIdx
                // push: 1 on success, 0 on failure
                // Uses the native EVM CALL opcode directly.
                case 0x30 {
                    let flags    := byte(0, mload(add(progData, pc)))
                    pc           := add(pc, 1)
                    sp           := sub(sp, 4)
                    let gasLimit := mload(add(stackBase, shl(5, add(sp, 3))))
                    let toAddr   := mload(add(stackBase, shl(5, add(sp, 2))))
                    let callVal  := mload(add(stackBase, shl(5, add(sp, 1))))
                    let bufIdx   := mload(add(stackBase, shl(5, sp)))
                    let bufPtr   := mload(add(bufsBase, shl(5, bufIdx)))
                    let bufLen   := mload(bufPtr)
                    let bufData  := add(bufPtr, 32)

                    // Native EVM CALL: no calldata copy, returndata via RETURNDATACOPY
                    let callOk := 0
                    switch iszero(gasLimit)
                    case 1  { callOk := call(gas(),    toAddr, callVal, bufData, bufLen, 0, 0) }
                    default { callOk := call(gasLimit, toAddr, callVal, bufData, bufLen, 0, 0) }

                    // Capture returndata into a fresh buffer
                    let rdLen := returndatasize()
                    let rdBuf := allocBuf(rdLen)
                    returndatacopy(add(rdBuf, 32), 0, rdLen)
                    retPtr := rdBuf

                    if and(flags, 1) {
                        if iszero(callOk) {
                            revertStatic(
                                "DeFiVM: adapter call failed",
                                27
                            )
                        }
                    }

                    mstore(add(stackBase, shl(5, sp)), callOk)
                    sp := add(sp, 1)
                }

                // ── 0x31  BALANCE_OF ────────────────────────────────────
                // pop: token (TOS, 0x0 = ETH), account; push balance
                case 0x31 {
                    sp          := sub(sp, 2)
                    let token   := mload(add(stackBase, shl(5, add(sp, 1))))
                    let account := mload(add(stackBase, shl(5, sp)))

                    let bal := 0
                    switch iszero(token)
                    case 1 { bal := balance(account) }
                    default {
                        // balanceOf(address) selector: 0x70a08231
                        let tmp := mload(0x40)
                        mstore(tmp,
                            0x70a0823100000000000000000000000000000000000000000000000000000000)
                        mstore(add(tmp, 4), account)
                        let ok := staticcall(gas(), token, tmp, 36, tmp, 32)
                        if iszero(ok) {
                            revertStatic("DeFiVM: balanceOf failed", 24)
                        }
                        bal := mload(tmp)
                    }
                    mstore(add(stackBase, shl(5, sp)), bal)
                    sp := add(sp, 1)
                }

                // ── 0x32  SELF_ADDR ─────────────────────────────────────
                case 0x32 {
                    mstore(add(stackBase, shl(5, sp)), address())
                    sp := add(sp, 1)
                }

                // ── 0x33  SUB (saturating) ──────────────────────────────
                case 0x33 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    let r := 0
                    if iszero(lt(a, b)) { r := sub(a, b) }
                    mstore(add(stackBase, shl(5, sp)), r)
                    sp := add(sp, 1)
                }

                // ── 0x34  ADD ───────────────────────────────────────────
                case 0x34 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), add(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x35  MUL ───────────────────────────────────────────
                case 0x35 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), mul(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x36  DIV ───────────────────────────────────────────
                case 0x36 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), div(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x37  MOD ───────────────────────────────────────────
                case 0x37 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), mod(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x38  LT ────────────────────────────────────────────
                case 0x38 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), lt(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x39  GT ────────────────────────────────────────────
                case 0x39 {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), gt(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x3a  EQ ────────────────────────────────────────────
                case 0x3a {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), eq(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x3b  ISZERO ────────────────────────────────────────
                case 0x3b {
                    sp    := sub(sp, 1)
                    let a := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), iszero(a))
                    sp    := add(sp, 1)
                }

                // ── 0x3c  AND ───────────────────────────────────────────
                case 0x3c {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), and(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x3d  OR ────────────────────────────────────────────
                case 0x3d {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), or(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x3e  XOR ───────────────────────────────────────────
                case 0x3e {
                    sp    := sub(sp, 2)
                    let a := mload(add(stackBase, shl(5, add(sp, 1))))
                    let b := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), xor(a, b))
                    sp    := add(sp, 1)
                }

                // ── 0x3f  NOT ───────────────────────────────────────────
                case 0x3f {
                    sp    := sub(sp, 1)
                    let a := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), not(a))
                    sp    := add(sp, 1)
                }

                // ── 0x40  PATCH_U256 ────────────────────────────────────
                // pop: value (TOS), bufIdx; overwrite 32-byte word at offset; push bufIdx
                case 0x40 {
                    let offset := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    sp         := sub(sp, 2)
                    let pval   := mload(add(stackBase, shl(5, add(sp, 1))))
                    let bufIdx := mload(add(stackBase, shl(5, sp)))
                    let bufPtr := mload(add(bufsBase, shl(5, bufIdx)))
                    mstore(add(add(bufPtr, 32), offset), pval)
                    mstore(add(stackBase, shl(5, sp)), bufIdx)
                    sp := add(sp, 1)
                }

                // ── 0x41  PATCH_ADDR ────────────────────────────────────
                // pop: addr (TOS), bufIdx; overwrite 20-byte address at offset; push bufIdx
                // The address value lives in the LOW 20 bytes of the 256-bit stack word.
                // Left-shift by 96 bits moves those 20 bytes to the HIGH positions so
                // BYTE(k, shifted) for k=0..19 yields the raw address bytes in order.
                //
                // We use MSTORE8 to write exactly 20 bytes because the offset may not be
                // 32-byte aligned, and using MSTORE would overwrite adjacent buffer bytes
                // (12 bytes before or after the address field) which could corrupt
                // the calldata template.  The 20-iteration loop is cheap relative to the
                // surrounding CALL cost.
                case 0x41 {
                    let offset := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    sp         := sub(sp, 2)
                    let paddr  := mload(add(stackBase, shl(5, add(sp, 1))))
                    let bufIdx := mload(add(stackBase, shl(5, sp)))
                    let bufPtr := mload(add(bufsBase, shl(5, bufIdx)))
                    let dst    := add(add(bufPtr, 32), offset)
                    let hi     := shl(96, paddr)
                    for { let k := 0 } lt(k, 20) { k := add(k, 1) } {
                        mstore8(add(dst, k), byte(k, hi))
                    }
                    mstore(add(stackBase, shl(5, sp)), bufIdx)
                    sp := add(sp, 1)
                }

                // ── 0x42  RET_U256 ──────────────────────────────────────
                case 0x42 {
                    let offset := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    let v      := mload(add(add(retPtr, 32), offset))
                    mstore(add(stackBase, shl(5, sp)), v)
                    sp := add(sp, 1)
                }

                // ── 0x43  RET_SLICE ─────────────────────────────────────
                case 0x43 {
                    let offset := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    let slen   := shr(240, mload(add(progData, pc)))
                    pc         := add(pc, 2)
                    let bufPtr := allocBuf(slen)
                    let src    := add(add(retPtr, 32), offset)
                    let dst    := add(bufPtr, 32)
                    for { let i := 0 } lt(i, slen) { i := add(i, 32) } {
                        mstore(add(dst, i), mload(add(src, i)))
                    }
                    mstore(add(bufsBase, shl(5, numBufs)), bufPtr)
                    mstore(add(stackBase, shl(5, sp)), numBufs)
                    sp      := add(sp, 1)
                    numBufs := add(numBufs, 1)
                }

                // ── 0x44  SHL ───────────────────────────────────────────
                case 0x44 {
                    sp        := sub(sp, 2)
                    let shift := mload(add(stackBase, shl(5, add(sp, 1))))
                    let value := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), shl(shift, value))
                    sp := add(sp, 1)
                }

                // ── 0x45  SHR ───────────────────────────────────────────
                case 0x45 {
                    sp        := sub(sp, 2)
                    let shift := mload(add(stackBase, shl(5, add(sp, 1))))
                    let value := mload(add(stackBase, shl(5, sp)))
                    mstore(add(stackBase, shl(5, sp)), shr(shift, value))
                    sp := add(sp, 1)
                }

                // ── Unknown opcode ──────────────────────────────────────
                default {
                    revertStatic("DeFiVM: unknown opcode", 22)
                }
            }
        }
    }
}
