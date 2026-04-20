"""Multi-hop swap composer for DeFiVM — direct pool calls.

This module provides helpers to compose multi-hop DEX swaps as atomic DeFiVM
programs that call pool contracts **directly**, without relying on router
contracts (e.g. Uniswap's Universal Router).

Each "hop" is implemented as:

* **Uniswap V3-style pools** — call ``pool.swap()`` directly.  The pool fires
  ``uniswapV3SwapCallback`` (or an equivalent variant); ``DeFiVM.fallback()``
  handles the repayment automatically.

* **Uniswap V2-style pairs** — pre-transfer the input tokens from the VM to the
  pair, compute ``amountOut`` from on-chain reserves at runtime, then call
  ``pair.swap()`` directly.

Callback data encoding
----------------------
When the pool calls back into DeFiVM the ``data`` parameter must be encoded so
that ``DeFiVM.fallback()`` knows which token to repay:

* **V3-style callbacks** (Uniswap V3, Algebra/QuickSwap, PancakeSwap V3,
  Solidly V3)::

      data = encode_v3_callback_data(token_in)
      # = abi.encode(address tokenIn)

* **V2-style callbacks** (Uniswap V2 and forks, Aerodrome/Velodrome hook,
  Ramses V2)::

      data = encode_v2_callback_data(token_in, amount_owed)
      # = abi.encode(address tokenIn, uint256 amountOwed)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eth_abi import encode
from eth_contract.contract import ContractFunction

from pydefi.types import Address, RouteSwap, SwapProtocol
from pydefi.vm.builder import Patch, Program
from pydefi.vm.program import (
    _SWAP2,
    add,
    bitwise_not,
    div,
    dup,
    mul,
    push_u256,
    ret_u256,
    swap,
)

# ---------------------------------------------------------------------------
# Pool function ABI signatures — selectors and argument encoding are computed
# automatically by ContractFunction; no hardcoded hex values needed.
# ---------------------------------------------------------------------------

# pool.swap(address recipient, bool zeroForOne, int256 amountSpecified,
#           uint160 sqrtPriceLimitX96, bytes data) — Uniswap V3 pool
_V3_POOL_SWAP_FN = ContractFunction.from_abi(
    "function swap(address recipient, bool zeroForOne, int256 amountSpecified, uint160 sqrtPriceLimitX96, bytes data)"
)

# V3 sqrtPriceLimitX96 boundaries (TickMath.MIN/MAX_SQRT_RATIO ± 1)
_SQRT_PRICE_MIN: int = 4295128740
_SQRT_PRICE_MAX: int = 1461446703485210103287273052203988822378723970341

# ---------------------------------------------------------------------------
# Return-data offsets (kept for backwards compatibility)
# ---------------------------------------------------------------------------

#: Byte offset of ``amountOut`` in the returndata of a V3 ``exactInputSingle``
#: router call.  Not used by the direct pool-call composer but kept for users
#: who still call router contracts.
V3_AMOUNT_OUT_OFFSET: int = 0

#: Byte offset of the final element of the ``amounts[]`` array returned by
#: ``swapExactTokensForTokens`` (Uniswap V2 router).  Not used by the direct
#: pool-call composer but kept for users who still call router contracts.
V2_AMOUNT_OUT_OFFSET: int = 96

# ---------------------------------------------------------------------------
# Callback data encoding helpers
# ---------------------------------------------------------------------------


def encode_v3_callback_data(token_in: Address) -> bytes:
    """Encode the ``data`` field for a V3-style flash-swap callback.

    The DeFiVM fallback handler expects ``abi.encode(address tokenIn)`` in the
    ``data`` parameter of ``uniswapV3SwapCallback``, ``algebraSwapCallback``,
    ``pancakeV3SwapCallback``, and ``solidlyV3SwapCallback``.

    Args:
        token_in: Address of the token the pool expects to receive back.

    Returns:
        32-byte ABI-encoded ``(address)``.
    """
    return encode(["address"], [token_in])


def encode_v2_callback_data(token_in: Address, amount_owed: int) -> bytes:
    """Encode the ``data`` field for a V2-style flash-swap callback.

    The DeFiVM fallback handler expects ``abi.encode(address tokenIn,
    uint256 amountOwed)`` in the ``data`` parameter of ``uniswapV2Call``,
    Aerodrome ``hook``, and ``ramsesV2FlashCallback``.

    Args:
        token_in: Address of the token the pool expects to receive back.
        amount_owed: Exact repayment amount (borrowed amount + fee).

    Returns:
        64-byte ABI-encoded ``(address, uint256)``.
    """
    return encode(["address", "uint256"], [token_in, amount_owed])


# ---------------------------------------------------------------------------
# Calldata builder helpers
# ---------------------------------------------------------------------------


def v3_pool_swap_calldata(
    recipient: Address,
    zero_for_one: bool,
    amount_in: int,
    sqrt_price_limit_x96: int,
    token_in: Address,
) -> bytes:
    """Build calldata for a direct ``pool.swap()`` call (Uniswap V3 pool).

    The callback ``data`` encodes ``token_in`` so that ``DeFiVM.fallback()``
    can repay the pool automatically.

    Args:
        recipient: Address that receives the output tokens.
        zero_for_one: ``True`` if swapping token0 → token1.
        amount_in: Exact input amount (use ``0`` as placeholder when patching
            at runtime).
        sqrt_price_limit_x96: Price limit.  Pass ``0`` to use the safe default
            (``MIN_SQRT_RATIO + 1`` for zero-for-one, ``MAX_SQRT_RATIO - 1``
            otherwise).
        token_in: Input token address (encoded in callback data).

    Returns:
        ABI-encoded calldata including the 4-byte selector.
    """
    if sqrt_price_limit_x96 == 0:
        sqrt_price_limit_x96 = _SQRT_PRICE_MIN if zero_for_one else _SQRT_PRICE_MAX
    callback_data = encode_v3_callback_data(token_in)
    return _V3_POOL_SWAP_FN(recipient, zero_for_one, amount_in, sqrt_price_limit_x96, callback_data).data


def encode_v3_path(tokens: list[Address], fees: list[int]) -> bytes:
    """Encode a V3 multi-hop path as ABI-packed bytes.

    Args:
        tokens: Ordered list of token addresses (at least 2).
        fees: Fee tier for each hop (``len(fees) == len(tokens) - 1``).

    Returns:
        ABI-packed bytes: ``token0 + fee0 + token1 + fee1 + token2 + …``

    Raises:
        ValueError: If ``len(fees) != len(tokens) - 1``.
    """
    if len(fees) != len(tokens) - 1:
        raise ValueError(f"encode_v3_path: len(fees) ({len(fees)}) must equal len(tokens)-1 ({len(tokens) - 1})")
    result = tokens[0]
    for fee, token in zip(fees, tokens[1:]):
        result += fee.to_bytes(3, "big")
        result += token
    return result


# ---------------------------------------------------------------------------
# Swap hop descriptor
# ---------------------------------------------------------------------------


@dataclass
class SwapHop:
    """Descriptor for one swap hop in a multi-hop DeFiVM program.

    Both V2 and V3 hops call the pool/pair contract **directly** — no router
    is needed.

    Attributes:
        protocol: DEX protocol to use for this hop.
        pool: Pool or pair contract address (not a router).
        token_in: Input token address.
        token_out: Output token address.
        fee: Pool fee in **basis points** (e.g. ``30`` for 0.30 %).  For V3
            pools the fee is encoded in the pool itself and is not passed to
            ``pool.swap()``; it is kept here for documentation only.  For V2
            pairs it is used to compute ``amountOut`` from reserves on-chain.
        recipient: Address to receive the output tokens.  For intermediate
            hops this must be the DeFiVM contract address so that tokens are
            available for subsequent hops.
        zero_for_one: ``True`` if ``token_in`` is ``token0`` in the pool/pair.
            Required to determine the swap direction for V3 pools and the
            reserve ordering for V2 on-chain amountOut computation.
        sqrt_price_limit_x96: V3 only — price limit passed to ``pool.swap()``.
            Pass ``0`` to use the safe default (``MIN_SQRT_RATIO + 1`` or
            ``MAX_SQRT_RATIO - 1`` depending on direction).
    """

    protocol: SwapProtocol
    pool: Address
    token_in: Address
    token_out: Address
    fee_bps: int
    recipient: Address
    zero_for_one: bool
    sqrt_price_limit_x96: int = field(default=0)


# ---------------------------------------------------------------------------
# Internal program-segment builders
# ---------------------------------------------------------------------------


def _build_v3_pool_swap_segment(hop: SwapHop) -> Program:
    """V3 pool direct swap (stack ABI).

    Stack contract:
    - input:  ``[... , amount_in]``
    - output: ``[... , amount_out]``
    """
    sqrt_price_limit_x96 = hop.sqrt_price_limit_x96
    if sqrt_price_limit_x96 == 0:
        sqrt_price_limit_x96 = _SQRT_PRICE_MIN if hop.zero_for_one else _SQRT_PRICE_MAX
    callback_data = encode_v3_callback_data(hop.token_in)

    prog = Program()
    prog.call_contract_abi(
        hop.pool,
        "function swap(address recipient, bool zeroForOne,"
        " int256 amountSpecified, uint160 sqrtPriceLimitX96, bytes data)",
        hop.recipient,
        hop.zero_for_one,
        Patch(),
        sqrt_price_limit_x96,
        callback_data,
    ).pop()

    if hop.zero_for_one:
        prog._emit(ret_u256(32))
    else:
        prog._emit(ret_u256(0))
    prog._emit(bitwise_not())
    prog._emit(push_u256(1))
    prog._emit(add())
    return prog


def _build_v2_direct_swap_segment(hop: SwapHop) -> Program:
    """V2 pair direct swap (stack ABI).

    Stack contract:
    - input:  ``[... , amount_in]``
    - output: ``[... , amount_out]``
    """
    if not 0 <= hop.fee_bps < 10000:
        raise ValueError(f"hop.fee must be in basis points within [0, 10000), got {hop.fee_bps}")
    fee_num = 10000 - hop.fee_bps

    prog = Program()
    prog.call_contract_abi(hop.pool, "getReserves()").pop()
    if hop.zero_for_one:
        prog._emit(ret_u256(0))
        prog._emit(ret_u256(32))
    else:
        prog._emit(ret_u256(32))
        prog._emit(ret_u256(0))

    # [amount_in, rIn, rOut] -> compute amount_out while keeping amount_in.
    prog._emit(bytes([0x82]))  # DUP3: amount_in
    prog._emit(push_u256(fee_num))
    prog._emit(mul())
    prog._emit(dup())
    prog._emit(bytes([_SWAP2]))
    prog._emit(mul())
    prog._emit(bytes([_SWAP2]))
    prog._emit(push_u256(10000))
    prog._emit(mul())
    prog._emit(add())
    prog._emit(swap())
    prog._emit(div())  # [amount_in, amount_out]
    prog._emit(swap())  # [amount_out, amount_in]

    prog.call_contract_abi(
        hop.token_in,
        "function transfer(address to, uint256 amount)",
        hop.pool,
        Patch(),
    ).pop()

    prog._emit(dup())  # [amount_out, amount_out]
    if hop.zero_for_one:
        prog.call_contract_abi(
            hop.pool,
            "function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes data)",
            0,
            Patch(),
            hop.recipient,
            b"",
        ).pop()
    else:
        prog.call_contract_abi(
            hop.pool,
            "function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes data)",
            Patch(),
            0,
            hop.recipient,
            b"",
        ).pop()

    return prog


_PROTOCOL_LOOKUP: dict[str, SwapProtocol] = {
    "uniswapv2": SwapProtocol.UNISWAP_V2,
    "uniswap_v2": SwapProtocol.UNISWAP_V2,
    "uniswap v2": SwapProtocol.UNISWAP_V2,
    "uniswapv3": SwapProtocol.UNISWAP_V3,
    "uniswap_v3": SwapProtocol.UNISWAP_V3,
    "uniswap v3": SwapProtocol.UNISWAP_V3,
}


def _pool_to_swap_protocol(protocol_name: str) -> SwapProtocol:
    """Convert a human-readable protocol name to a :class:`SwapProtocol` enum value."""
    result = _PROTOCOL_LOOKUP.get(protocol_name.lower())
    if result is None:
        raise ValueError(f"unsupported pool protocol {protocol_name!r}")
    return result


def _swap_hop_from_route_swap(swap_action: RouteSwap, *, recipient: Address) -> SwapHop:
    pool = swap_action.pool
    return SwapHop(
        protocol=_pool_to_swap_protocol(pool.protocol),
        pool=pool.pool_address,
        token_in=pool.token_in.address,
        token_out=pool.token_out.address,
        fee_bps=pool.fee_bps,
        recipient=recipient,
        zero_for_one=swap_action.zero_for_one(),
    )


def _build_route_swap_segment(
    action: RouteSwap,
    *,
    recipient: Address,
) -> Program:
    hop = _swap_hop_from_route_swap(action, recipient=recipient)
    if hop.protocol == SwapProtocol.UNISWAP_V3:
        return _build_v3_pool_swap_segment(hop)
    return _build_v2_direct_swap_segment(hop)
