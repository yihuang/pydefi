// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title DeFiVM
 * @notice A minimal, register-based macro-assembler / interpreter for composable DeFi flows.
 *
 * Design principles
 * -----------------
 *  • Atomic execution – a "program" runs all-at-once; any revert undoes everything.
 *  • Register-based   – 16 named registers (R0-R15) plus a temporary 32-element stack.
 *  • Fully stateless  – no owner, no whitelist; the CALL opcode can reach any address.
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
 *
 * ABI / data
 *   0x40  PATCH_U256 <2-byte offset>     pop: value, bufIdx -> patch 32-byte word in buffer
 *   0x41  PATCH_ADDR <2-byte offset>     pop: addr,  bufIdx -> patch 20-byte word in buffer
 *   0x42  RET_U256   <2-byte offset>     push uint256 from last returndata at offset
 *   0x43  RET_SLICE  <2-byte off> <2-byte len>  push bytes slice from last returndata
 */
contract DeFiVM {
    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    uint8  private constant MAX_STACK   = 32;
    uint8  private constant MAX_REGS    = 16;
    uint8  private constant MAX_BUFFERS = 16;

    // Opcodes
    uint8 private constant OP_PUSH_U256   = 0x01;
    uint8 private constant OP_PUSH_ADDR   = 0x02;
    uint8 private constant OP_PUSH_BYTES  = 0x03;
    uint8 private constant OP_DUP         = 0x04;
    uint8 private constant OP_SWAP        = 0x05;
    uint8 private constant OP_POP         = 0x06;
    uint8 private constant OP_LOAD_REG    = 0x10;
    uint8 private constant OP_STORE_REG   = 0x11;
    uint8 private constant OP_JUMP        = 0x20;
    uint8 private constant OP_JUMPI       = 0x21;
    uint8 private constant OP_REVERT_IF   = 0x22;
    uint8 private constant OP_ASSERT_GE   = 0x23;
    uint8 private constant OP_ASSERT_LE   = 0x24;
    uint8 private constant OP_CALL        = 0x30;
    uint8 private constant OP_BALANCE_OF  = 0x31;
    uint8 private constant OP_SELF_ADDR   = 0x32;
    uint8 private constant OP_SUB         = 0x33;
    uint8 private constant OP_ADD         = 0x34;
    uint8 private constant OP_MUL         = 0x35;
    uint8 private constant OP_DIV         = 0x36;
    uint8 private constant OP_MOD         = 0x37;
    uint8 private constant OP_PATCH_U256  = 0x40;
    uint8 private constant OP_PATCH_ADDR  = 0x41;
    uint8 private constant OP_RET_U256    = 0x42;
    uint8 private constant OP_RET_SLICE   = 0x43;

    /// @notice Allow the VM to receive ETH (needed for value-bearing calls).
    receive() external payable {}

    // -------------------------------------------------------------------------
    // Execution state (kept entirely in memory)
    // -------------------------------------------------------------------------

    struct VMState {
        // Stack (up to MAX_STACK = 32 entries)
        bytes32[32] stack;
        uint8       sp;           // sp == 0 means empty

        // Registers (16)
        bytes32[16] regs;

        // Byte buffer store (for calldata templates and return-data slices)
        bytes[16]   buffers;
        uint8       numBufs;

        // Last external call returndata
        bytes       retdata;
    }

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Execute a DeFiVM program atomically.
     * @param program  Bytecode stream (packed instructions).
     *
     * Any revert undoes all side-effects.
     */
    function execute(bytes calldata program) external payable {
        // Copy calldata to memory once; all subsequent reads use memory ops.
        bytes memory prog = program;
        VMState memory s;

        uint256 pc = 0;
        uint256 plen = prog.length;

        while (pc < plen) {
            uint8 op = uint8(prog[pc]);
            pc++;

            // ------------------------------------------------------------------
            // Stack / data
            // ------------------------------------------------------------------
            if (op == OP_PUSH_U256) {
                require(pc + 32 <= plen, "DeFiVM: truncated PUSH_U256");
                bytes32 v;
                uint256 moff = pc;
                assembly {
                    v := mload(add(add(prog, 32), moff))
                }
                _push(s, v);
                pc += 32;

            } else if (op == OP_PUSH_ADDR) {
                require(pc + 20 <= plen, "DeFiVM: truncated PUSH_ADDR");
                // Addresses are stored big-endian, 20 bytes.
                // Read into the high 20 bytes of a bytes32 then shift right 12 bytes.
                bytes32 raw;
                uint256 moff = pc;
                assembly {
                    raw := mload(add(add(prog, 32), moff))
                }
                address a = address(uint160(uint256(raw >> 96)));
                _push(s, bytes32(uint256(uint160(a))));
                pc += 20;

            } else if (op == OP_PUSH_BYTES) {
                require(pc + 2 <= plen, "DeFiVM: truncated PUSH_BYTES length");
                uint16 blen = _read_u16(prog, pc);
                pc += 2;
                require(pc + blen <= plen, "DeFiVM: truncated PUSH_BYTES data");
                require(s.numBufs < MAX_BUFFERS, "DeFiVM: buffer limit");
                bytes memory buf = new bytes(blen);
                for (uint256 i = 0; i < blen; i++) {
                    buf[i] = prog[pc + i];
                }
                uint8 idx = s.numBufs;
                s.buffers[idx] = buf;
                s.numBufs++;
                _push(s, bytes32(uint256(idx)));
                pc += blen;

            } else if (op == OP_DUP) {
                require(s.sp > 0, "DeFiVM: DUP on empty stack");
                uint8 top = s.sp - 1;
                _push(s, s.stack[top]);

            } else if (op == OP_SWAP) {
                require(s.sp >= 2, "DeFiVM: SWAP needs 2 items");
                uint8 a = s.sp - 1;
                uint8 b = s.sp - 2;
                (s.stack[a], s.stack[b]) = (s.stack[b], s.stack[a]);

            } else if (op == OP_POP) {
                require(s.sp > 0, "DeFiVM: POP on empty stack");
                s.sp--;

            // ------------------------------------------------------------------
            // Register load / store
            // ------------------------------------------------------------------
            } else if (op == OP_LOAD_REG) {
                require(pc + 1 <= plen, "DeFiVM: truncated LOAD_REG");
                uint8 i = uint8(prog[pc]);
                pc++;
                require(i < MAX_REGS, "DeFiVM: bad register");
                _push(s, s.regs[i]);

            } else if (op == OP_STORE_REG) {
                require(pc + 1 <= plen, "DeFiVM: truncated STORE_REG");
                uint8 i = uint8(prog[pc]);
                pc++;
                require(i < MAX_REGS, "DeFiVM: bad register");
                require(s.sp > 0, "DeFiVM: STORE_REG empty stack");
                s.sp--;
                s.regs[i] = s.stack[s.sp];

            // ------------------------------------------------------------------
            // Control flow
            // ------------------------------------------------------------------
            } else if (op == OP_JUMP) {
                require(pc + 2 <= plen, "DeFiVM: truncated JUMP");
                uint16 target = _read_u16(prog, pc);
                pc += 2;
                require(target <= plen, "DeFiVM: JUMP out of bounds");
                pc = target;

            } else if (op == OP_JUMPI) {
                require(pc + 2 <= plen, "DeFiVM: truncated JUMPI");
                uint16 target = _read_u16(prog, pc);
                pc += 2;
                require(s.sp > 0, "DeFiVM: JUMPI empty stack");
                s.sp--;
                bytes32 cond = s.stack[s.sp];
                if (cond != bytes32(0)) {
                    require(target <= plen, "DeFiVM: JUMPI out of bounds");
                    pc = target;
                }

            } else if (op == OP_REVERT_IF) {
                require(pc + 1 <= plen, "DeFiVM: truncated REVERT_IF");
                uint8 msgLen = uint8(prog[pc]);
                pc++;
                require(pc + msgLen <= plen, "DeFiVM: truncated REVERT_IF msg");
                require(s.sp > 0, "DeFiVM: REVERT_IF empty stack");
                s.sp--;
                if (s.stack[s.sp] != bytes32(0)) {
                    bytes memory msg_ = new bytes(msgLen);
                    for (uint8 k = 0; k < msgLen; k++) {
                        msg_[k] = prog[pc + k];
                    }
                    revert(string(msg_));
                }
                pc += msgLen;

            } else if (op == OP_ASSERT_GE) {
                // pop a (top), pop b (below) -> require a >= b
                require(pc + 1 <= plen, "DeFiVM: truncated ASSERT_GE");
                uint8 msgLen = uint8(prog[pc]);
                pc++;
                require(pc + msgLen <= plen, "DeFiVM: truncated ASSERT_GE msg");
                require(s.sp >= 2, "DeFiVM: ASSERT_GE needs 2 items");
                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);
                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);
                if (a < b) {
                    bytes memory msg_ = new bytes(msgLen);
                    for (uint8 k = 0; k < msgLen; k++) {
                        msg_[k] = prog[pc + k];
                    }
                    revert(string(msg_));
                }
                pc += msgLen;

            } else if (op == OP_ASSERT_LE) {
                // pop a (top), pop b (below) -> require a <= b
                require(pc + 1 <= plen, "DeFiVM: truncated ASSERT_LE");
                uint8 msgLen = uint8(prog[pc]);
                pc++;
                require(pc + msgLen <= plen, "DeFiVM: truncated ASSERT_LE msg");
                require(s.sp >= 2, "DeFiVM: ASSERT_LE needs 2 items");
                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);
                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);
                if (a > b) {
                    bytes memory msg_ = new bytes(msgLen);
                    for (uint8 k = 0; k < msgLen; k++) {
                        msg_[k] = prog[pc + k];
                    }
                    revert(string(msg_));
                }
                pc += msgLen;

            // ------------------------------------------------------------------
            // External / introspection
            // ------------------------------------------------------------------
            } else if (op == OP_CALL) {
                require(pc + 1 <= plen, "DeFiVM: truncated CALL flags");
                uint8 flags = uint8(prog[pc]);
                pc++;
                bool requireSuccess = (flags & 0x01) != 0;

                // Stack order (top to bottom): gasLimit, to, value, calldataBufIdx
                require(s.sp >= 4, "DeFiVM: CALL needs 4 items");

                s.sp--;
                uint256 gasLimit = uint256(s.stack[s.sp]);

                s.sp--;
                address to = address(uint160(uint256(s.stack[s.sp])));

                s.sp--;
                uint256 callValue = uint256(s.stack[s.sp]);

                s.sp--;
                uint8 bufIdx = uint8(uint256(s.stack[s.sp]));
                require(bufIdx < s.numBufs, "DeFiVM: CALL invalid buffer");
                bytes memory calldata_ = s.buffers[bufIdx];

                bool ok;
                bytes memory ret;
                if (gasLimit == 0) {
                    (ok, ret) = to.call{value: callValue}(calldata_);
                } else {
                    (ok, ret) = to.call{value: callValue, gas: gasLimit}(calldata_);
                }
                s.retdata = ret;

                if (requireSuccess) {
                    require(ok, "DeFiVM: adapter call failed");
                }
                _push(s, ok ? bytes32(uint256(1)) : bytes32(0));

            } else if (op == OP_BALANCE_OF) {
                // pop: token (address, 0x0=ETH), account (address) -> push balance (uint256)
                require(s.sp >= 2, "DeFiVM: BALANCE_OF needs 2 items");

                s.sp--;
                address token = address(uint160(uint256(s.stack[s.sp])));

                s.sp--;
                address account = address(uint160(uint256(s.stack[s.sp])));

                uint256 bal;
                if (token == address(0)) {
                    bal = account.balance;
                } else {
                    // balanceOf(address) selector: 0x70a08231
                    (bool ok, bytes memory res) = token.staticcall(
                        abi.encodeWithSelector(0x70a08231, account)
                    );
                    require(ok, "DeFiVM: balanceOf failed");
                    bal = abi.decode(res, (uint256));
                }
                _push(s, bytes32(bal));

            } else if (op == OP_SELF_ADDR) {
                _push(s, bytes32(uint256(uint160(address(this)))));

            } else if (op == OP_SUB) {
                // pop a (top), pop b -> push (a - b), saturates to 0
                require(s.sp >= 2, "DeFiVM: SUB needs 2 items");

                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);

                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);

                _push(s, bytes32(a >= b ? a - b : 0));

            } else if (op == OP_ADD) {
                // pop a (top), pop b -> push a + b (wrapping uint256)
                require(s.sp >= 2, "DeFiVM: ADD needs 2 items");

                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);

                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);

                unchecked { _push(s, bytes32(a + b)); }

            } else if (op == OP_MUL) {
                // pop a (top), pop b -> push a * b (wrapping uint256)
                require(s.sp >= 2, "DeFiVM: MUL needs 2 items");

                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);

                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);

                unchecked { _push(s, bytes32(a * b)); }

            } else if (op == OP_DIV) {
                // pop a (top), pop b -> push a / b (0 if b == 0)
                require(s.sp >= 2, "DeFiVM: DIV needs 2 items");

                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);

                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);

                _push(s, bytes32(b == 0 ? 0 : a / b));

            } else if (op == OP_MOD) {
                // pop a (top), pop b -> push a % b (0 if b == 0)
                require(s.sp >= 2, "DeFiVM: MOD needs 2 items");

                s.sp--;
                uint256 a = uint256(s.stack[s.sp]);

                s.sp--;
                uint256 b = uint256(s.stack[s.sp]);

                _push(s, bytes32(b == 0 ? 0 : a % b));

            // ------------------------------------------------------------------
            // ABI / data patching
            // ------------------------------------------------------------------
            } else if (op == OP_PATCH_U256) {
                // <2-byte offset>  |  pop: value (uint256), bufIdx (bytes)
                require(pc + 2 <= plen, "DeFiVM: truncated PATCH_U256");
                uint16 offset = _read_u16(prog, pc);
                pc += 2;
                require(s.sp >= 2, "DeFiVM: PATCH_U256 needs 2 items");

                s.sp--;
                uint256 pval = uint256(s.stack[s.sp]);

                s.sp--;
                uint8 bidx = uint8(uint256(s.stack[s.sp]));
                require(bidx < s.numBufs, "DeFiVM: PATCH_U256 invalid buffer");
                require(uint256(offset) + 32 <= s.buffers[bidx].length, "DeFiVM: PATCH_U256 out of bounds");
                bytes memory b = s.buffers[bidx];
                uint256 moff = offset;
                assembly {
                    mstore(add(add(b, 32), moff), pval)
                }
                // Push the (now-modified) buffer index back
                _push(s, bytes32(uint256(bidx)));

            } else if (op == OP_PATCH_ADDR) {
                // <2-byte offset>  |  pop: addr (address), bufIdx (bytes)
                require(pc + 2 <= plen, "DeFiVM: truncated PATCH_ADDR");
                uint16 offset = _read_u16(prog, pc);
                pc += 2;
                require(s.sp >= 2, "DeFiVM: PATCH_ADDR needs 2 items");

                s.sp--;
                address paddr = address(uint160(uint256(s.stack[s.sp])));

                s.sp--;
                uint8 bidx = uint8(uint256(s.stack[s.sp]));
                require(bidx < s.numBufs, "DeFiVM: PATCH_ADDR invalid buffer");
                require(uint256(offset) + 20 <= s.buffers[bidx].length, "DeFiVM: PATCH_ADDR out of bounds");
                // Patch 20 bytes starting exactly at `offset` (raw byte-wise copy).
                bytes memory b = s.buffers[bidx];
                bytes memory addrBytes = abi.encodePacked(paddr);
                for (uint256 i = 0; i < 20; i++) {
                    b[uint256(offset) + i] = addrBytes[i];
                }
                _push(s, bytes32(uint256(bidx)));

            } else if (op == OP_RET_U256) {
                // <2-byte offset>  -> push uint256 from last returndata
                require(pc + 2 <= plen, "DeFiVM: truncated RET_U256");
                uint16 offset = _read_u16(prog, pc);
                pc += 2;
                require(uint256(offset) + 32 <= s.retdata.length, "DeFiVM: RET_U256 out of bounds");
                bytes32 word;
                bytes memory rd = s.retdata;
                uint256 moff = offset;
                assembly {
                    word := mload(add(add(rd, 32), moff))
                }
                _push(s, word);

            } else if (op == OP_RET_SLICE) {
                // <2-byte offset> <2-byte len>  -> push bytes slice from last returndata
                require(pc + 4 <= plen, "DeFiVM: truncated RET_SLICE");
                uint16 offset = _read_u16(prog, pc);
                pc += 2;
                uint16 slen = _read_u16(prog, pc);
                pc += 2;
                require(uint256(offset) + slen <= s.retdata.length, "DeFiVM: RET_SLICE out of bounds");
                require(s.numBufs < MAX_BUFFERS, "DeFiVM: buffer limit (RET_SLICE)");
                bytes memory slice = new bytes(slen);
                for (uint256 k = 0; k < slen; k++) {
                    slice[k] = s.retdata[offset + k];
                }
                uint8 idx = s.numBufs;
                s.buffers[idx] = slice;
                s.numBufs++;
                _push(s, bytes32(uint256(idx)));

            } else {
                revert("DeFiVM: unknown opcode");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// @dev Push a value onto the VM stack.
    function _push(VMState memory s, bytes32 val) private pure {
        require(s.sp < MAX_STACK, "DeFiVM: stack overflow");
        s.stack[s.sp] = val;
        s.sp++;
    }

    /// @dev Read a big-endian uint16 from a memory bytes array at position `pos`.
    function _read_u16(bytes memory b, uint256 pos) private pure returns (uint16 v) {
        assembly {
            v := shr(240, mload(add(add(b, 32), pos)))
        }
    }
}
