"""DeFiVM program builder — Python DSL for assembling DeFiVM EVM bytecode.

Programs are raw EVM bytecode executed on the native EVM stack by the
Analog-Labs interpreter.  ``execute()`` in DeFiVM.sol runs a program via
``DELEGATECALL`` into that interpreter.

Memory conventions used by the interpreter
------------------------------------------

Adopt the same memory layout as the Solidity compiler:

- Scratch space:       ``memory[0x00:0x3f]``
- Free memory pointer: ``memory[0x40:0x5f]``
- Zero slot:           ``memory[0x60:0x7f]``

Free memory pointer is initialized to 0x80, zero slot is never written to.

* Store slice less than 64 bytes temporarily, use scratch space.
* Store slice greater than 64 bytes temporarily, use free memory and not update the free memory pointer.
* Store slice and reuse it later, write to free memory and update the free memory pointer.

When assemble the bytecode, the program builder automatically prepends the bytecode to initialize the free memory
pointer to 0x80.
"""

from __future__ import annotations

from evmdasm import EvmInstructions, Instruction
from evmdasm import argtypes as T
from evmdasm.registry import INSTRUCTIONS, InstructionRegistry

registry = InstructionRegistry(
    instructions=INSTRUCTIONS
    + [
        Instruction(
            opcode=0x5f,
            name="PUSH0",
            category="stack",
            gas=2,
            length_of_operand=0x0,
            description="Place constant 0 on stack.",
            returns=[T.Value("item")],
        ),
    ]
)

FREE_MEMORY_POINTER = 0x40

INIT_FP_PROGRAM = b"\x60\x80\x60\x40\x52"  # PUSH1 0x80 PUSH1 0x40 MSTORE

def int2bytes(n: int) -> bytes:
    """
    convert int to bytes in big endian format, minimal 1 byte, to be used PUSH opcodes
    """
    if n == 0:
        return b""
    else:
        return n.to_bytes((n.bit_length() + 7) // 8, "big")


class Program(EvmInstructions):
    """
    p = Program()
    p.push(b"\x01\x02").push(100)
    c.dup1().swap1().pop()
    c.op("JUMPDEST")

    # allow chaining
    c.op("OR").op("OR")
    """

    def __getattr__(self, name: str) -> callable[[bytes | int | None], Program]:
        "catch all the undefined calls, try to match instructions"
        instr = registry.by_name.get(name.upper())
        if not instr:
            raise AttributeError("Instruction %s does not exist" % name)

        def op(arg: bytes | int | None = None) -> Program:
            """
            EVM opcode expects at most one operand
            """
            new_instr = instr.clone()
            if new_instr.operand_length > 0:
                if arg is None:
                    raise TypeError("Instruction %s requires an operand" % name)

                if isinstance(arg, int):
                    arg = int2bytes(arg)

                if len(arg) != new_instr.operand_length:
                    raise TypeError(
                        "Instruction %s requires an operand of length %d, but got %d"
                        % (name, new_instr.operand_length, len(arg))
                    )

                new_instr.operand_bytes = arg
            else:
                if arg is not None:
                    raise TypeError("Instruction %s does not take an operand" % name)
            self.append(new_instr)
            return self

        return op

    def push(self, data: int | bytes) -> Program:
        """
        Pick the right PUSH opcode based on the length of the data
        """
        if isinstance(data, int):
            data = int2bytes(data)

        if not data:
            return self.push0()

        name = "PUSH%d" % len(data)
        return getattr(self, name)(data)

    def op(self, name, arg: int | bytes | None = None) -> Program:
        return getattr(self, name)(arg)

    def pop(self) -> Program:
        """
        Override the pop method inherited from list
        """
        return getattr(self, "POP")()

    def extend(self, p: Program) -> Program:
        """
        Override to allow chaining
        """
        self.extend(p)
        return self

    def assemble(self, init_fp: bool = False) -> bytes:
        bz = super(Program, self).assemble().as_bytes
        if init_fp:
            return INIT_FP_PROGRAM + bz
        return bz

    def __bytes__(self) -> bytes:
        return self.assemble()
