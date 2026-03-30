"""DeFiVM ABI helpers — calldata encoding for contract calls.

Uses the `eth-contract` library to encode calldata from human-readable ABI
signatures.  For well-known interfaces such as ERC-20, use the pre-built
contract objects directly::

    from eth_contract.erc20 import ERC20

    transfer_cd  = bytes(ERC20.fns.transfer(RECIPIENT, amount).data)
    approve_cd   = bytes(ERC20.fns.approve(ROUTER, amount).data)
    balance_cd   = bytes(ERC20.fns.balanceOf(ACCOUNT).data)

For arbitrary function signatures use :func:`encode_calldata` or the fluent
builder's :meth:`~pydefi.vm.builder.Program.call_contract_abi` helper::

    from pydefi.vm import Program
    from pydefi.vm.abi import encode_calldata

    # Build calldata from a human-readable signature and args
    calldata = encode_calldata(
        "function transfer(address to, uint256 amount)",
        [RECIPIENT, 10**18],
    )

    # Or use the fluent builder helper directly
    bytecode = (
        Program()
        .call_contract_abi(TOKEN, "transfer(address,uint256)", RECIPIENT, 10**18)
        .pop()
        .build()
    )
"""

from __future__ import annotations


def encode_calldata(abi_sig: str, args: list | tuple = ()) -> bytes:
    """Encode EVM calldata from a human-readable ABI function signature and arguments.

    Uses :class:`eth_contract.contract.ContractFunction` for parsing and
    encoding, so all Solidity primitive types and arbitrarily-nested
    tuple/array types are supported.

    Args:
        abi_sig: Human-readable function signature.  The ``function`` keyword
            is optional — both ``"transfer(address,uint256)"`` and
            ``"function transfer(address to, uint256 amount) external"`` are
            accepted.  Parameter names are also optional.
        args: Positional arguments matching the signature's input parameters.
            Pass as a list or tuple; the order must match the parameter order.

    Returns:
        4-byte Keccak-256 selector + ABI-encoded arguments.

    Example::

        # Simple ERC-20 transfer
        calldata = encode_calldata("transfer(address,uint256)", [RECIPIENT, 10**18])

        # Uniswap V3 exactInputSingle with a struct arg
        calldata = encode_calldata(
            "function exactInputSingle("
            "  (address tokenIn, address tokenOut, uint24 fee, address recipient,"
            "   uint256 deadline, uint256 amountIn, uint256 amountOutMinimum,"
            "   uint160 sqrtPriceLimitX96) params"
            ")",
            [(TOKEN_IN, TOKEN_OUT, 3000, RECIPIENT, deadline, amount_in, 0, 0)],
        )
    """
    from eth_contract.contract import ContractFunction

    normalised = abi_sig if abi_sig.lstrip().startswith("function ") else "function " + abi_sig
    fn = ContractFunction.from_abi(normalised)
    return bytes(fn(*args).data)
