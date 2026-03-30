"""DeFiVM ABI helpers — calldata builders for common ERC-20 operations.

These helpers produce raw ``bytes`` calldata suitable for use with
:func:`pydefi.vm.program.push_bytes` or :meth:`pydefi.vm.builder.Program.call_contract`.

Example — approve then swap::

    from pydefi.vm import Program
    from pydefi.vm.abi import erc20_approve, erc20_transfer

    bytecode = (
        Program()
        .call_contract(TOKEN, erc20_approve(ROUTER, amount_in))
        .call_contract(ROUTER, swap_calldata)
        .call_contract(TOKEN, erc20_transfer(RECIPIENT, 0))
        .build()
    )

Human-readable ABI example::

    from pydefi.vm import Program
    from pydefi.vm.abi import encode_calldata

    # Build calldata from a human-readable signature and args
    calldata = encode_calldata(
        "function transfer(address to, uint256 amount)",
        ["0xRecipient...", 10 ** 18],
    )

    # Or use the fluent builder helper directly
    bytecode = (
        Program()
        .call_contract_abi(TOKEN, "transfer(address,uint256)", RECIPIENT, 10 ** 18)
        .pop()
        .build()
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _u256(n: int) -> bytes:
    if n < 0:
        raise ValueError(f"_u256: value must be non-negative, got {n}")
    return n.to_bytes(32, "big")


def _addr_word(a: str) -> bytes:
    """Encode an Ethereum address as a 32-byte ABI word (12 zero bytes + 20 address bytes)."""
    raw = bytes.fromhex(a.removeprefix("0x"))
    if len(raw) != 20:
        raise ValueError(f"bad address length: expected 20 bytes, got {len(raw)} from {a!r}")
    return b"\x00" * 12 + raw


def _abi_type_to_str(inp: dict) -> str:
    """Convert an ABI JSON input/output entry to an ``eth_abi``-compatible type string.

    Recursively handles ``tuple`` types by building ``(T1,T2,...)`` from the
    ``components`` field, preserving any array suffix (e.g. ``tuple[]`` → ``(T1,T2,...)[]``).
    """
    type_ = inp["type"]
    if type_.startswith("tuple"):
        suffix = type_[len("tuple"):]          # e.g. "" | "[]" | "[5]"
        components = inp.get("components", [])
        inner = ",".join(_abi_type_to_str(c) for c in components)
        return f"({inner}){suffix}"
    return type_


# ---------------------------------------------------------------------------
# Human-readable ABI calldata encoder
# ---------------------------------------------------------------------------


def encode_calldata(abi_sig: str, args: list | tuple = ()) -> bytes:
    """Encode EVM calldata from a human-readable ABI function signature and arguments.

    Uses :func:`eth_contract.human.parse_function_signature` to parse the
    signature and :func:`eth_abi.encode` for argument encoding, so all Solidity
    primitive types and arbitrarily-nested tuple/array types are supported.

    Args:
        abi_sig: Human-readable function signature.  The ``function`` keyword is
            optional — both ``"transfer(address,uint256)"`` and
            ``"function transfer(address to, uint256 amount) external"`` are
            accepted.
        args: Positional arguments matching the signature's input parameters.
            Pass as a list or tuple; the order must match the parameter order.

    Returns:
        4-byte Keccak-256 selector + ABI-encoded arguments.

    Raises:
        :exc:`ValueError`: If ``abi_sig`` cannot be parsed as a function signature.

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
    from eth_abi import encode as _encode
    from eth_contract.human import parse_function_signature
    from eth_utils import keccak

    # Normalise: eth_contract requires the 'function' keyword
    normalised = abi_sig if abi_sig.lstrip().startswith("function ") else "function " + abi_sig
    fn = parse_function_signature(normalised)

    inputs = fn.get("inputs", [])
    types = [_abi_type_to_str(inp) for inp in inputs]

    # Canonical signature for selector — type names only, no spaces
    canonical = fn["name"] + "(" + ",".join(types) + ")"
    selector = keccak(text=canonical)[:4]

    encoded_args = _encode(types, list(args)) if types else b""
    return selector + encoded_args


# ---------------------------------------------------------------------------
# ERC-20 calldata builders
# ---------------------------------------------------------------------------


def erc20_transfer(to: str, amount: int) -> bytes:
    """Build ``transfer(address,uint256)`` calldata.

    Selector: ``0xa9059cbb``
    """
    return bytes.fromhex("a9059cbb") + _addr_word(to) + _u256(amount)


def erc20_approve(spender: str, amount: int) -> bytes:
    """Build ``approve(address,uint256)`` calldata.

    Selector: ``0x095ea7b3``
    """
    return bytes.fromhex("095ea7b3") + _addr_word(spender) + _u256(amount)


def erc20_transfer_from(from_addr: str, to: str, amount: int) -> bytes:
    """Build ``transferFrom(address,address,uint256)`` calldata.

    Selector: ``0x23b872dd``
    """
    return bytes.fromhex("23b872dd") + _addr_word(from_addr) + _addr_word(to) + _u256(amount)


def erc20_balance_of(account: str) -> bytes:
    """Build ``balanceOf(address)`` calldata.

    Selector: ``0x70a08231``
    """
    return bytes.fromhex("70a08231") + _addr_word(account)
