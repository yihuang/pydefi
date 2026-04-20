from __future__ import annotations

from evmdasm import EvmInstructions
from evmdasm.registry import registry


def int2bytes(n: int) -> bytes:
    """
    convert int to bytes in big endian format, minimal 1 byte, to be used PUSH opcodes
    """
    if n == 0:
        return b"\x00"
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
        '''
        pick the right PUSH opcode based on the length of the data
        '''
        if isinstance(data, int):
            data = int2bytes(data)

        name = "PUSH%d" % len(data)
        return getattr(self, name)(data)

    def op(self, name, arg: int | bytes | None = None) -> Program:
        return getattr(self, name)(arg)

    def pop(self) -> Program:
        '''
        override the pop method inherited from list
        '''
        return getattr(self, "POP")()

    def assemble(self) -> bytes:
        return super(Program, self).assemble().as_bytes

    def __bytes__(self) -> bytes:
        return self.assemble()
