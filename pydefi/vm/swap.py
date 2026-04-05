"""Multi-hop swap composer for DeFiVM.

This module provides helpers to compose multi-hop DEX swaps as atomic DeFiVM
programs.  Each "hop" is an ERC-20 approve followed by a swap call on either a
Uniswap V2-compatible or Uniswap V3-compatible router.  Hops are chained via
DeFiVM registers: the output amount of each hop is stored in a register and
patched into the next hop's ``amountIn`` argument at runtime.

Callback data encoding
----------------------
When initiating flash swaps via DeFiVM, the pool will call back into the VM
contract.  The ``data`` parameter passed to the pool must be encoded as
described below so that ``DeFiVM.fallback()`` can repay the pool:

* **V3-style callbacks** (Uniswap V3, Algebra/QuickSwap, PancakeSwap V3,
  Solidly V3)::

      data = encode_v3_callback_data(token_in)
      # = abi.encode(address tokenIn)

* **V2-style callbacks** (Uniswap V2 and forks, Aerodrome/Velodrome hook,
  Ramses V2)::

      data = encode_v2_callback_data(token_in, amount_owed)
      # = abi.encode(address tokenIn, uint256 amountOwed)

Quick-start — two-hop swap (WETH → USDC → DAI)
------------------------------------------------
::

    import time
    from pydefi.vm.swap import (
        SwapHop,
        SwapProtocol,
        build_multi_hop_program,
        V3_AMOUNT_OUT_OFFSET,
        V2_AMOUNT_OUT_OFFSET,
    )

    WETH      = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC      = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    DAI       = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
    V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

    hops = [
        SwapHop(
            protocol=SwapProtocol.UNISWAP_V3,
            router=V3_ROUTER,
            token_in=WETH,
            token_out=USDC,
            fee=500,
            amount_in=10**18,        # 1 WETH (first hop: static amount)
            amount_out_min=0,
            recipient=VM_ADDRESS,    # keep in VM for next hop
            deadline=int(time.time()) + 600,
            out_offset=V3_AMOUNT_OUT_OFFSET,
        ),
        SwapHop(
            protocol=SwapProtocol.UNISWAP_V2,
            router=V2_ROUTER,
            token_in=USDC,
            token_out=DAI,
            fee=0,                   # not used for V2
            amount_in=0,             # 0 = read from previous hop register at runtime
            amount_out_min=0,
            recipient=USER_ADDRESS,
            deadline=int(time.time()) + 600,
            out_offset=V2_AMOUNT_OUT_OFFSET,
        ),
    ]

    bytecode = build_multi_hop_program(hops, min_final_out=900 * 10**18).build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from eth_abi import encode
from eth_contract import Contract
from eth_contract.erc20 import ERC20

from pydefi.vm.builder import Program
from pydefi.vm.program import (
    assert_ge,
    balance_of,
    load_reg,
    push_addr,
    push_u256,
    ret_u256,
    store_reg,
    swap,
)

# ---------------------------------------------------------------------------
# ABI definitions
# ---------------------------------------------------------------------------

_V2_ROUTER_ABI = [
    "function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)",
]

_V3_ROUTER_ABI = [
    "function exactInputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96) params) external payable returns (uint256 amountOut)",
    "function exactInput((bytes path, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum) params) external payable returns (uint256 amountOut)",
]

# ---------------------------------------------------------------------------
# Return-data offsets
# ---------------------------------------------------------------------------

#: Byte offset of ``amountOut`` in the returndata of a successful
#: ``exactInputSingle`` or ``exactInput`` call (Uniswap V3-style).
#: The function returns a single ``uint256`` so the value is at offset 0.
V3_AMOUNT_OUT_OFFSET: int = 0

#: Byte offset of the final element of the ``amounts[]`` array returned by
#: ``swapExactTokensForTokens`` (Uniswap V2-style) for a **two-token path**.
#:
#: ABI layout of ``uint256[] memory``:
#:   [0..32)   offset pointer (= 32)
#:   [32..64)  array length   (= 2 for a direct single-hop swap)
#:   [64..96)  amounts[0]     = amountIn
#:   [96..128) amounts[1]     = amountOut   ← this offset
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
# Calldata builders
# ---------------------------------------------------------------------------


def v2_swap_calldata(
    token_in: str,
    token_out: str,
    amount_in: int,
    amount_out_min: int,
    recipient: str,
    deadline: int,
) -> bytes:
    """Build calldata for ``swapExactTokensForTokens`` (Uniswap V2-style).

    Args:
        token_in: Input token address.
        token_out: Output token address.
        amount_in: Exact input amount (use ``0`` as placeholder when patching
            at runtime with :func:`build_multi_hop_program`).
        amount_out_min: Minimum acceptable output amount.
        recipient: Address that receives the output tokens.
        deadline: Unix timestamp after which the call reverts.

    Returns:
        ABI-encoded calldata bytes (selector + params).
    """
    router = Contract.from_abi(_V2_ROUTER_ABI)
    return router.fns.swapExactTokensForTokens(
        amount_in, amount_out_min, [token_in, token_out], recipient, deadline
    ).data


def v3_exact_input_single_calldata(
    token_in: str,
    token_out: str,
    fee: int,
    recipient: str,
    deadline: int,
    amount_in: int,
    amount_out_minimum: int,
) -> bytes:
    """Build calldata for ``exactInputSingle`` (Uniswap V3-style).

    Args:
        token_in: Input token address.
        token_out: Output token address.
        fee: Pool fee tier in hundredths of a basis point (e.g. 500, 3000, 10000).
        recipient: Address that receives the output tokens.
        deadline: Unix timestamp after which the call reverts.
        amount_in: Exact input amount (use ``0`` as placeholder when patching
            at runtime with :func:`build_multi_hop_program`).
        amount_out_minimum: Minimum acceptable output amount.

    Returns:
        ABI-encoded calldata bytes (selector + params).
    """
    router = Contract.from_abi(_V3_ROUTER_ABI)
    params = (token_in, token_out, fee, recipient, deadline, amount_in, amount_out_minimum, 0)
    return router.fns.exactInputSingle(params).data


def v3_exact_input_calldata(
    encoded_path: bytes,
    recipient: str,
    deadline: int,
    amount_in: int,
    amount_out_minimum: int,
) -> bytes:
    """Build calldata for ``exactInput`` (Uniswap V3 multi-hop path).

    Args:
        encoded_path: ABI-packed path bytes: ``tokenA + fee(3 bytes) + tokenB + …``
            (use :func:`encode_v3_path` to build this).
        recipient: Address that receives the output tokens.
        deadline: Unix timestamp.
        amount_in: Exact input amount.
        amount_out_minimum: Minimum acceptable output amount.

    Returns:
        ABI-encoded calldata bytes.
    """
    router = Contract.from_abi(_V3_ROUTER_ABI)
    params = (encoded_path, recipient, deadline, amount_in, amount_out_minimum)
    return router.fns.exactInput(params).data


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
    """Supported DEX protocols for :class:`SwapHop`."""

    UNISWAP_V2 = "uniswap_v2"
    """Uniswap V2-compatible: ``swapExactTokensForTokens``."""
    UNISWAP_V3 = "uniswap_v3"
    """Uniswap V3 single-hop: ``exactInputSingle``."""
    UNISWAP_V3_MULTIHOP = "uniswap_v3_multihop"
    """Uniswap V3 multi-hop: ``exactInput`` with an encoded path."""


@dataclass
class SwapHop:
    """Descriptor for one swap hop in a multi-hop program.

    Attributes:
        protocol: DEX protocol to use for this hop.
        router: Router contract address.
        token_in: Input token address.
        token_out: Output token address.
        fee: Pool fee tier in hundredths of a basis point (V3 only; ignored
            for V2).
        amount_in: Static input amount for the first hop.  Set to ``0`` for
            subsequent hops — the composer will patch the amount from the
            previous hop's output register at runtime.
        amount_out_min: Minimum acceptable output amount passed to the router
            for this hop.  Use ``0`` to disable per-hop slippage checking
            (rely on the final ``min_final_out`` check instead).
        recipient: Address to receive the output tokens.  For intermediate
            hops, use the DeFiVM contract address so tokens remain available
            for subsequent hops.
        deadline: Unix timestamp after which the router call reverts.
        out_offset: Byte offset of the output amount in this hop's returndata.
            Use :data:`V3_AMOUNT_OUT_OFFSET` for V3 routers and
            :data:`V2_AMOUNT_OUT_OFFSET` for V2 routers.
        encoded_path: Pre-encoded V3 multi-hop path bytes (required only for
            ``SwapProtocol.UNISWAP_V3_MULTIHOP``; use :func:`encode_v3_path`
            to build it).
        approve_max: When ``True`` (default), approve ``2**256 - 1`` tokens
            to the router before the swap.  When ``False``, approve the
            exact ``amount_in`` (only useful for the first hop when the amount
            is known at program-build time).
    """

    protocol: SwapProtocol
    router: str
    token_in: str
    token_out: str
    fee: int
    amount_in: int
    amount_out_min: int
    recipient: str
    deadline: int
    out_offset: int = field(default=V3_AMOUNT_OUT_OFFSET)
    encoded_path: bytes = field(default=b"")
    approve_max: bool = field(default=True)


# ---------------------------------------------------------------------------
# Internal program-segment builders
# ---------------------------------------------------------------------------

_MAX_U256 = 2**256 - 1

#: Default register used to pass amounts between hops.
_AMOUNT_REG = 0


def _build_approve_segment(token: str, spender: str, amount: int) -> Program:
    """Return a Program that calls ``token.approve(spender, amount)``."""
    return Program().call_contract(token, ERC20.fns.approve(spender, amount).data).pop()


def _build_v2_swap_segment(hop: SwapHop, *, patch_amount: bool, amount_reg: int) -> Program:
    """Return a Program segment for a V2-style swap."""
    calldata = v2_swap_calldata(
        hop.token_in,
        hop.token_out,
        hop.amount_in,
        hop.amount_out_min,
        hop.recipient,
        hop.deadline,
    )
    prog = Program()
    if patch_amount:
        # amountIn is the first word after the 4-byte selector (offset 4).
        prog.call_with_patches(hop.router, calldata, patches=[(4, 32, load_reg(amount_reg))]).pop()
    else:
        prog.call_contract(hop.router, calldata).pop()
    return prog


def _build_v3_single_swap_segment(hop: SwapHop, *, patch_amount: bool, amount_reg: int) -> Program:
    """Return a Program segment for a V3 ``exactInputSingle`` swap."""
    calldata = v3_exact_input_single_calldata(
        hop.token_in,
        hop.token_out,
        hop.fee,
        hop.recipient,
        hop.deadline,
        hop.amount_in,
        hop.amount_out_min,
    )
    prog = Program()
    if patch_amount:
        # ExactInputSingleParams struct layout (after 4-byte selector):
        #   [4..36)   tokenIn            (address)
        #   [36..68)  tokenOut           (address)
        #   [68..100) fee                (uint24, padded)
        #   [100..132) recipient         (address)
        #   [132..164) deadline          (uint256)
        #   [164..196) amountIn          (uint256)  ← patch here
        #   [196..228) amountOutMinimum  (uint256)
        #   [228..260) sqrtPriceLimitX96 (uint160, padded)
        prog.call_with_patches(hop.router, calldata, patches=[(164, 32, load_reg(amount_reg))]).pop()
    else:
        prog.call_contract(hop.router, calldata).pop()
    return prog


def _build_v3_multihop_swap_segment(hop: SwapHop, *, patch_amount: bool, amount_reg: int) -> Program:
    """Return a Program segment for a V3 ``exactInput`` multi-hop swap."""
    if not hop.encoded_path:
        raise ValueError("SwapHop.encoded_path must be non-empty for UNISWAP_V3_MULTIHOP")
    calldata = v3_exact_input_calldata(
        hop.encoded_path,
        hop.recipient,
        hop.deadline,
        hop.amount_in,
        hop.amount_out_min,
    )
    prog = Program()
    if patch_amount:
        # ExactInputParams layout (bytes path is dynamic, the other fields are static):
        #   [4..36)   path offset        (points to the bytes data)
        #   [36..68)  recipient          (address)
        #   [68..100) deadline           (uint256)
        #   [100..132) amountIn          (uint256)  ← patch here
        #   [132..164) amountOutMinimum  (uint256)
        prog.call_with_patches(hop.router, calldata, patches=[(100, 32, load_reg(amount_reg))]).pop()
    else:
        prog.call_contract(hop.router, calldata).pop()
    return prog


# ---------------------------------------------------------------------------
# High-level multi-hop composer
# ---------------------------------------------------------------------------


def build_multi_hop_program(
    hops: list[SwapHop],
    min_final_out: int = 0,
    amount_reg: int = _AMOUNT_REG,
) -> Program:
    """Compose a list of swap hops into a single atomic DeFiVM program.

    The resulting program executes the following steps for each hop in order:

    1. ``token_in.approve(router, MAX_U256)``  (or exact amount when
       ``approve_max=False``).
    2. Call the router's swap function.
    3. Read the output amount from returndata and store it in *amount_reg*.

    The first hop uses ``hop.amount_in`` as a static value encoded in the
    calldata template.  All subsequent hops patch ``amountIn`` at runtime
    from *amount_reg*.

    Optionally, a final ``assert_ge`` check is appended that reverts if the
    last hop's output is below *min_final_out*.

    Args:
        hops: Ordered list of :class:`SwapHop` descriptors.  At least one
            hop is required.
        min_final_out: If ``> 0``, the program reverts when the last hop
            produces fewer tokens than this value.  Pass ``0`` to skip.
        amount_reg: DeFiVM register index (0–15) used to pass the output
            amount between hops.

    Returns:
        A :class:`~pydefi.vm.builder.Program` ready for ``.build()``.

    Raises:
        ValueError: If *hops* is empty or a hop has an unsupported protocol.
    """
    if not hops:
        raise ValueError("build_multi_hop_program: hops list must not be empty")

    segments: list[Program] = []

    for i, hop in enumerate(hops):
        is_first = i == 0
        patch_amount = not is_first

        # ── Approve ──────────────────────────────────────────────────────────
        approve_amount = _MAX_U256 if hop.approve_max else hop.amount_in
        segments.append(_build_approve_segment(hop.token_in, hop.router, approve_amount))

        # ── Swap ─────────────────────────────────────────────────────────────
        if hop.protocol == SwapProtocol.UNISWAP_V2:
            swap_seg = _build_v2_swap_segment(hop, patch_amount=patch_amount, amount_reg=amount_reg)
        elif hop.protocol == SwapProtocol.UNISWAP_V3:
            swap_seg = _build_v3_single_swap_segment(hop, patch_amount=patch_amount, amount_reg=amount_reg)
        elif hop.protocol == SwapProtocol.UNISWAP_V3_MULTIHOP:
            swap_seg = _build_v3_multihop_swap_segment(hop, patch_amount=patch_amount, amount_reg=amount_reg)
        else:
            raise ValueError(f"build_multi_hop_program: unsupported protocol {hop.protocol!r}")

        segments.append(swap_seg)

        # ── Store output in register for next hop ─────────────────────────────
        store_seg = Program()._emit(ret_u256(hop.out_offset))._emit(store_reg(amount_reg))
        segments.append(store_seg)

    # ── Final slippage check ──────────────────────────────────────────────────
    if min_final_out > 0:
        # Stack before assert_ge: [a=amount_out (TOS), b=min_final_out]
        # assert_ge reverts if a < b, i.e. if amount_out < min_final_out
        final_check = (
            Program()
            ._emit(push_u256(min_final_out))  # push b = min_final_out
            ._emit(load_reg(amount_reg))  # push a = amount_out
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
        token: ERC-20 token address (use ``address(0)`` for native ETH).
        account: Account whose balance to check.
        min_amount: Minimum required balance.

    Returns:
        A :class:`~pydefi.vm.builder.Program` snippet.
    """
    # balance_of pops [account(TOS), token(2nd)], pushes [balance]
    # assert_ge pops [a(TOS)=balance, b(2nd)=min_amount], reverts if a < b
    return (
        Program()
        ._emit(push_addr(account))
        ._emit(push_addr(token))
        ._emit(balance_of())
        ._emit(push_u256(min_amount))
        ._emit(swap())
        ._emit(assert_ge("balance below minimum"))
    )
