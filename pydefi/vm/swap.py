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

Quick-start — two-hop swap (WETH → USDC → DAI) via pool contracts
-----------------------------------------------------------------
::

    from pydefi.vm.swap import SwapHop, SwapProtocol, build_multi_hop_program

    hops = [
        SwapHop(
            protocol=SwapProtocol.UNISWAP_V3,
            pool=WETH_USDC_V3_POOL,   # pool address, NOT a router
            token_in=WETH,
            token_out=USDC,
            fee=500,                  # informational; not used in call args
            amount_in=10**18,         # 1 WETH
            amount_out_min=0,
            recipient=VM_ADDRESS,     # keep in VM for next hop
            zero_for_one=True,        # WETH is token0 in this pool
        ),
        SwapHop(
            protocol=SwapProtocol.UNISWAP_V2,
            pool=USDC_DAI_V2_PAIR,    # pair address, NOT a router
            token_in=USDC,
            token_out=DAI,
            fee=30,                   # 0.30 % fee in basis points
            amount_in=0,              # 0 = use previous hop's output at runtime
            amount_out_min=0,
            recipient=USER_ADDRESS,
            zero_for_one=True,        # USDC is token0 in this pair
        ),
    ]

    bytecode = build_multi_hop_program(hops, min_final_out=900 * 10**18).build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from eth_abi import encode
from eth_contract.contract import ContractFunction

from pydefi.vm.builder import Patch, Program
from pydefi.vm.program import (
    _SWAP2,
    add,
    assert_ge,
    balance_of,
    bitwise_not,
    div,
    dup,
    load_reg,
    mul,
    push_addr,
    push_u256,
    ret_u256,
    store_reg,
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


def encode_v3_callback_data(token_in: str) -> bytes:
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


def encode_v2_callback_data(token_in: str, amount_owed: int) -> bytes:
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
    recipient: str,
    zero_for_one: bool,
    amount_in: int,
    sqrt_price_limit_x96: int,
    token_in: str,
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


def encode_v3_path(tokens: list[str], fees: list[int]) -> bytes:
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
    result = bytes.fromhex(tokens[0].removeprefix("0x").zfill(40))
    for fee, token in zip(fees, tokens[1:]):
        result += fee.to_bytes(3, "big")
        result += bytes.fromhex(token.removeprefix("0x").zfill(40))
    return result


# ---------------------------------------------------------------------------
# Swap hop descriptor
# ---------------------------------------------------------------------------


class SwapProtocol(str, Enum):
    """Supported DEX protocols for :class:`SwapHop`.

    Both values use **direct pool/pair calls** — no router contract is involved.
    """

    UNISWAP_V2 = "uniswap_v2"
    """Uniswap V2-compatible pair: pre-transfer tokenIn, then call ``pair.swap()``.

    On-chain amountOut is computed from ``pair.getReserves()`` using the
    constant-product formula, so no off-chain quote is required.
    """

    UNISWAP_V3 = "uniswap_v3"
    """Uniswap V3-compatible pool: call ``pool.swap()`` directly.

    The pool fires a flash-swap callback (``uniswapV3SwapCallback`` or a
    compatible variant) which ``DeFiVM.fallback()`` handles automatically.
    """


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
        amount_in: Static input amount for the **first** hop.  Set to ``0``
            for subsequent hops — the amount is read at runtime from the
            register that holds the previous hop's output.
        amount_out_min: Minimum acceptable output amount for this hop.
            Currently unused in the generated program (pass ``0``); rely on
            the global ``min_final_out`` parameter of
            :func:`build_multi_hop_program` instead.
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
    pool: str
    token_in: str
    token_out: str
    fee: int
    amount_in: int
    amount_out_min: int
    recipient: str
    zero_for_one: bool
    sqrt_price_limit_x96: int = field(default=0)


# ---------------------------------------------------------------------------
# Internal program-segment builders
# ---------------------------------------------------------------------------

_MAX_U256 = 2**256 - 1

#: Default register index used to carry amounts between hops.
_AMOUNT_REG: int = 0

#: Default register index for V2 hops to store the computed amountOut temp.
_AMOUNT_OUT_REG: int = 1


def _build_v3_pool_swap_segment(hop: SwapHop, *, amount_reg: int) -> Program:
    """V3 pool direct swap.

    Sequence:
    1. ``pool.swap(recipient, zeroForOne, amountIn, sqrtPriceLimit,
       abi.encode(tokenIn))`` — amountIn patched from *amount_reg* via
       :class:`~pydefi.vm.builder.Patch` (offset detected automatically).
    2. Extract ``amountOut`` from return values (negate the negative delta).
    3. Store ``amountOut`` back in *amount_reg*.
    """
    sqrt_price_limit_x96 = hop.sqrt_price_limit_x96
    if sqrt_price_limit_x96 == 0:
        sqrt_price_limit_x96 = _SQRT_PRICE_MIN if hop.zero_for_one else _SQRT_PRICE_MAX
    callback_data = encode_v3_callback_data(hop.token_in)

    prog = Program()
    # Use call_contract_abi so the ABI library locates the amountSpecified
    # offset automatically — no hardcoded byte offset.
    prog.call_contract_abi(
        hop.pool,
        "function swap(address recipient, bool zeroForOne,"
        " int256 amountSpecified, uint160 sqrtPriceLimitX96, bytes data)",
        hop.recipient,
        hop.zero_for_one,
        Patch(load_reg(amount_reg)),  # amountSpecified — patched at runtime
        sqrt_price_limit_x96,
        callback_data,
    ).pop()

    # Extract amountOut from returndata.
    # pool.swap() returns (int256 amount0, int256 amount1):
    #   zeroForOne → amount0 > 0 (owed by caller), amount1 < 0 (sent to recipient)
    #   !zeroForOne → amount0 < 0 (sent to recipient), amount1 > 0 (owed by caller)
    # amountOut = |negative value| = two's-complement negation = NOT(v) + 1
    if hop.zero_for_one:
        prog._emit(ret_u256(32))  # amount1 (negative → negate)
    else:
        prog._emit(ret_u256(0))  # amount0 (negative → negate)
    prog._emit(bitwise_not())
    prog._emit(push_u256(1))
    prog._emit(add())

    prog._emit(store_reg(amount_reg))
    return prog


def _build_v2_direct_swap_segment(hop: SwapHop, *, amount_reg: int, amount_out_reg: int) -> Program:
    """V2 pair direct swap: compute amountOut from reserves on-chain.

    Sequence:
    1. ``pair.getReserves()`` — determine reserveIn / reserveOut.
    2. Compute ``amountOut`` using the constant-product formula on the EVM
       stack (no temporary registers for reserves):
       ``amountIn * fee_num * reserveOut / (reserveIn * 10000 + amountIn * fee_num)``
    3. Store computed ``amountOut`` in *amount_out_reg* for calldata patching.
    4. ``tokenIn.transfer(pair, amountIn)`` — transfer input from VM.
    5. ``pair.swap(amount0Out, amount1Out, recipient, "")`` — amountOut patched
       from *amount_out_reg* via :class:`~pydefi.vm.builder.Patch`.
    6. Copy ``amountOut`` from *amount_out_reg* back to *amount_reg*.

    Args:
        hop: The V2 swap hop descriptor.
        amount_reg: Register holding ``amountIn`` on entry; updated with
            ``amountOut`` on exit.
        amount_out_reg: Scratch register for the intermediate ``amountOut``
            value (must differ from *amount_reg*).
    """

    # hop.fee is in basis points (e.g. 30 for 0.30 %)
    # Uniswap V2 standard: amountInWithFee = amountIn * 997 / 1000 (for 0.30 % fee)
    # Generalised:         amountInWithFee = amountIn * (10000 - fee_bps)
    #                      denominator     = reserveIn * 10000 + amountInWithFee
    if not 0 <= hop.fee < 10000:
        raise ValueError(f"hop.fee must be in basis points within [0, 10000), got {hop.fee}")
    fee_num = 10000 - hop.fee

    prog = Program()

    # --- Step 1: Get reserves --------------------------------------------------
    prog.call_contract_abi(hop.pool, "getReserves()").pop()

    # --- Step 2: Compute amountOut on the EVM stack (no reserve registers) ----
    # Push reserveIn and reserveOut from returndata.
    # getReserves() → (reserve0, reserve1, timestamp); each field is 32 bytes.
    if hop.zero_for_one:
        # tokenIn = token0 → reserveIn = reserve0 (ret[0]), reserveOut = reserve1 (ret[32])
        prog._emit(ret_u256(0))  # [rIn]
        prog._emit(ret_u256(32))  # [rIn, rOut]
    else:
        # tokenIn = token1 → reserveIn = reserve1 (ret[32]), reserveOut = reserve0 (ret[0])
        prog._emit(ret_u256(32))  # [rIn]
        prog._emit(ret_u256(0))  # [rIn, rOut]

    # amountInWithFee = amountIn * fee_num
    prog._emit(load_reg(amount_reg))  # [rIn, rOut, amountIn]
    prog._emit(push_u256(fee_num))  # [rIn, rOut, amountIn, fee_num]
    prog._emit(mul())  # [rIn, rOut, aif]
    prog._emit(dup())  # [rIn, rOut, aif, aif_dup]

    # SWAP2 exchanges TOS(aif_dup) with the item 2 below TOS (rOut):
    #   [rIn, rOut, aif, aif_dup]  →  [rIn, aif_dup, aif, rOut]
    prog._emit(bytes([_SWAP2]))
    prog._emit(mul())  # [rIn, aif_dup, numerator=aif*rOut]

    # SWAP2 exchanges TOS(numerator) with the item 2 below TOS (rIn):
    #   [rIn, aif_dup, numerator]  →  [numerator, aif_dup, rIn]
    prog._emit(bytes([_SWAP2]))
    prog._emit(push_u256(10000))  # [numerator, aif_dup, rIn, 10000]
    prog._emit(mul())  # [numerator, aif_dup, rIn*10000]
    prog._emit(add())  # [numerator, denominator=rIn*10000+aif_dup]
    prog._emit(swap())  # [denominator, numerator]   (SWAP1: put numerator at TOS)
    prog._emit(div())  # [amountOut = numerator/denominator]
    prog._emit(store_reg(amount_out_reg))

    # --- Step 3: Transfer amountIn to pair ------------------------------------
    prog.call_contract_abi(
        hop.token_in,
        "function transfer(address to, uint256 amount)",
        hop.pool,
        Patch(load_reg(amount_reg)),
    ).pop()

    # --- Step 4: Call pair.swap with amountOut --------------------------------
    # pair.swap(uint amount0Out, uint amount1Out, address to, bytes data)
    #   zero_for_one → amount0Out=0, amount1Out=amountOut (tokenOut is token1)
    #   !zero_for_one → amount0Out=amountOut, amount1Out=0 (tokenOut is token0)
    if hop.zero_for_one:
        prog.call_contract_abi(
            hop.pool,
            "function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes data)",
            0,
            Patch(load_reg(amount_out_reg)),
            hop.recipient,
            b"",
        ).pop()
    else:
        prog.call_contract_abi(
            hop.pool,
            "function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes data)",
            Patch(load_reg(amount_out_reg)),
            0,
            hop.recipient,
            b"",
        ).pop()

    # --- Step 5: Update amount_reg for the next hop ---------------------------
    prog._emit(load_reg(amount_out_reg))
    prog._emit(store_reg(amount_reg))

    return prog


# ---------------------------------------------------------------------------
# High-level multi-hop composer
# ---------------------------------------------------------------------------


def build_multi_hop_program(
    hops: list[SwapHop],
    min_final_out: int = 0,
    amount_reg: int = _AMOUNT_REG,
    amount_out_reg: int = _AMOUNT_OUT_REG,
) -> Program:
    """Compose a list of swap hops into a single atomic DeFiVM program.

    All hops call pool/pair contracts **directly** — no router is involved.

    For each hop the generated program:

    * **V3 (``UNISWAP_V3``)** — calls ``pool.swap()`` with the input amount
      from *amount_reg*; the pool fires a flash-swap callback that
      ``DeFiVM.fallback()`` handles automatically; then extracts and stores the
      output amount via two's-complement negation of the return value.

    * **V2 (``UNISWAP_V2``)** — reads reserves via ``pair.getReserves()``,
      computes ``amountOut`` on-chain with the constant-product formula, calls
      ``tokenIn.transfer(pair, amountIn)``, and finally calls
      ``pair.swap(amount0Out, amount1Out, recipient, "")``.

    The **first hop** uses ``hop.amount_in`` as the initial amount (pushed into
    *amount_reg*).  Subsequent hops read their input amount directly from
    *amount_reg*, which holds the previous hop's output.

    Args:
        hops: Ordered list of :class:`SwapHop` descriptors.  At least one is
            required.
        min_final_out: If ``> 0``, the program reverts when the last hop's
            output is below this value (slippage guard).  Pass ``0`` to skip.
        amount_reg: DeFiVM register index (0–15) used to pass amounts between
            hops.  Holds ``amountIn`` on entry to each hop and ``amountOut``
            on exit.
        amount_out_reg: DeFiVM register index (0–15) used by V2 hops as a
            scratch register for the intermediate ``amountOut`` value before
            it is patched into the ``pair.swap()`` calldata.  Must differ from
            *amount_reg*.

    Returns:
        A :class:`~pydefi.vm.builder.Program` ready for ``.build()``.

    Raises:
        ValueError: If *hops* is empty, a hop has an unsupported protocol, or
            *amount_reg* equals *amount_out_reg*.
    """
    if not hops:
        raise ValueError("build_multi_hop_program: hops list must not be empty")
    if amount_reg == amount_out_reg:
        raise ValueError(
            f"build_multi_hop_program: amount_reg ({amount_reg}) and "
            f"amount_out_reg ({amount_out_reg}) must be different registers"
        )

    segments: list[Program] = []

    for i, hop in enumerate(hops):
        # For the first hop, initialise amount_reg with the static input amount.
        if i == 0:
            segments.append(Program()._emit(push_u256(hop.amount_in))._emit(store_reg(amount_reg)))

        if hop.protocol == SwapProtocol.UNISWAP_V3:
            swap_seg = _build_v3_pool_swap_segment(hop, amount_reg=amount_reg)
        elif hop.protocol == SwapProtocol.UNISWAP_V2:
            swap_seg = _build_v2_direct_swap_segment(hop, amount_reg=amount_reg, amount_out_reg=amount_out_reg)
        else:
            raise ValueError(f"build_multi_hop_program: unsupported protocol {hop.protocol!r}")

        segments.append(swap_seg)

    # Optional final slippage guard
    if min_final_out > 0:
        final_check = (
            Program()
            ._emit(push_u256(min_final_out))
            ._emit(load_reg(amount_reg))
            ._emit(assert_ge("slippage: out too low"))
        )
        segments.append(final_check)

    return Program.compose(segments)


# ---------------------------------------------------------------------------
# Balance-check helper
# ---------------------------------------------------------------------------


def check_min_balance(token: str, account: str, min_amount: int) -> Program:
    """Return a Program snippet that reverts if ``balanceOf(token, account) < min_amount``.

    Useful as a post-swap safety guard to verify the output landed in the
    expected account.

    Args:
        token: ERC-20 token address.
        account: Account whose balance to check.
        min_amount: Minimum required balance.

    Returns:
        A :class:`~pydefi.vm.builder.Program` snippet.
    """
    return (
        Program()
        ._emit(push_addr(account))
        ._emit(push_addr(token))
        ._emit(balance_of())
        ._emit(push_u256(min_amount))
        ._emit(swap())
        ._emit(assert_ge("balance below minimum"))
    )
