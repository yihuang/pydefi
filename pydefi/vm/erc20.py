from __future__ import annotations

from eth_contract.erc20 import ERC20

from pydefi.types import Address
from pydefi.vm.context import Operand, Program, ValueLike


def emit_approve(prog: Program, token: Address, spender: Address, amount: ValueLike, *, reset: bool = False) -> Operand:
    """Emit ``token.approve(spender, amount)`` into *prog*, asserting success and
    returning the CALL-success operand.  With *reset*, prepend ``approve(0)`` for
    USDT-style tokens that forbid a nonzero→nonzero allowance change."""
    if reset:
        prog.assert_(prog.call_contract(token, ERC20.fns.approve, spender, 0), "approve reset failed")
    ok = prog.call_contract(token, ERC20.fns.approve, spender, amount)
    prog.assert_(ok, "approve failed")
    return ok
