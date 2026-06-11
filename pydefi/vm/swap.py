"""Multi-hop swap composer — direct pool calls.

V2 / V3 hops (quote and execution) emitted as SSA IR over
:class:`pydefi.vm.Program`.  Each ``_build_*`` helper takes a
:class:`Operand` representing ``amount_in`` and returns a :class:`Operand` for
``amount_out``, threading the result through subsequent calls in plain
Python instead of via an implicit EVM stack contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eth_abi import encode
from eth_contract.erc20 import ERC20
from eth_utils import keccak

from pydefi.abi.amm import UNISWAP_V2_PAIR, UNISWAP_V3_POOL, UNISWAP_V3_QUOTER_V2
from pydefi.types import Address, RouteSwap, SwapProtocol, SwapRoute, SwapTransaction
from pydefi.pathfinder.dag import RouteDAG
from pydefi.vm.context import Operand, Program

# ---------------------------------------------------------------------------
# Pool / quoter / token function ABI signatures (sourced from pydefi.abi)
# ---------------------------------------------------------------------------

_V3_POOL_SWAP_FN = UNISWAP_V3_POOL.fns.swap
_V2_PAIR_SWAP_FN = UNISWAP_V2_PAIR.fns.swap
_V2_PAIR_GET_RESERVES_FN = UNISWAP_V2_PAIR.fns.getReserves
_V3_QUOTER_QUOTE_EXACT_INPUT_FN = UNISWAP_V3_QUOTER_V2.fns.quoteExactInput
_ERC20_TRANSFER_FN = ERC20.fns.transfer

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
    """Encode the ``data`` field for a V3-style flash-swap callback."""
    return encode(["address"], [token_in])


def encode_v2_callback_data(token_in: Address, amount_owed: int) -> bytes:
    """Encode the ``data`` field for a V2-style flash-swap callback."""
    return encode(["address", "uint256"], [token_in, amount_owed])


def v3_pool_swap_calldata(
    recipient: Address,
    zero_for_one: bool,
    amount_in: int,
    sqrt_price_limit_x96: int,
    token_in: Address,
) -> bytes:
    """Build calldata for a direct ``pool.swap()`` call (Uniswap V3 pool)."""
    if sqrt_price_limit_x96 == 0:
        sqrt_price_limit_x96 = _SQRT_PRICE_MIN if zero_for_one else _SQRT_PRICE_MAX
    callback_data = encode_v3_callback_data(token_in)
    return _V3_POOL_SWAP_FN(recipient, zero_for_one, amount_in, sqrt_price_limit_x96, callback_data).data


def encode_v3_path(tokens: list[Address], fees: list[int]) -> bytes:
    """Encode a V3 multi-hop path as ABI-packed bytes."""
    if len(fees) != len(tokens) - 1:
        raise ValueError(f"encode_v3_path: len(fees) ({len(fees)}) must equal len(tokens)-1 ({len(tokens) - 1})")
    result = tokens[0]
    for fee, token in zip(fees, tokens[1:]):
        result += fee.to_bytes(3, "big")
        result += token
    return result


# ---------------------------------------------------------------------------
# SwapHop descriptor
# ---------------------------------------------------------------------------


@dataclass
class SwapHop:
    """Descriptor for one swap hop in a multi-hop DeFiVM program."""

    protocol: SwapProtocol
    pool: Address
    token_in: Address
    token_out: Address
    fee_bps: int
    recipient: Address
    zero_for_one: bool
    sqrt_price_limit_x96: int = field(default=0)


# ---------------------------------------------------------------------------
# Internal SSA hop builders
# ---------------------------------------------------------------------------


def _build_v3_pool_swap(prog: Program, amount_in: Operand, hop: SwapHop) -> Operand:
    """V3 pool direct swap. Returns ``amount_out`` as an SSA :class:`Operand`."""
    sqrt_price_limit_x96 = hop.sqrt_price_limit_x96 or (_SQRT_PRICE_MIN if hop.zero_for_one else _SQRT_PRICE_MAX)
    callback_data = encode_v3_callback_data(hop.token_in)

    success = prog.call_contract(
        hop.pool,
        _V3_POOL_SWAP_FN,
        hop.recipient,
        hop.zero_for_one,
        amount_in,
        sqrt_price_limit_x96,
        callback_data,
    )
    prog.assert_(success)

    # V3 returns (int256 amount0Delta, int256 amount1Delta).  The pool reports
    # negative deltas for tokens it sent OUT.  amount_out is the magnitude of
    # the delta of the token we received.
    raw_delta = prog.returndata_word(32 if hop.zero_for_one else 0)
    # Negate two's complement: -x  ==  ~x + 1
    return prog.add(prog.bit_not(raw_delta), 1)


def _build_v2_compute_out(prog: Program, amount_in: Operand, hop: SwapHop, fee_num: int) -> Operand:
    """Compute V2 ``amountOut`` from ``getReserves()`` (no transfer)."""
    success = prog.call_contract(hop.pool, _V2_PAIR_GET_RESERVES_FN)
    prog.assert_(success)

    if hop.zero_for_one:
        r_in = prog.returndata_word(0)
        r_out = prog.returndata_word(32)
    else:
        r_in = prog.returndata_word(32)
        r_out = prog.returndata_word(0)

    # amount_in_with_fee = amount_in * fee_num
    # numerator   = amount_in_with_fee * r_out
    # denominator = r_in * 10000 + amount_in_with_fee
    # amount_out  = numerator // denominator
    amount_in_with_fee = prog.mul(amount_in, fee_num)
    numerator = prog.mul(amount_in_with_fee, r_out)
    denominator = prog.add(prog.mul(r_in, 10000), amount_in_with_fee)
    return prog.div(numerator, denominator)


def _build_v2_quote(prog: Program, amount_in: Operand, hop: SwapHop) -> Operand:
    """V2 pair quote — view-only."""
    if not 0 <= hop.fee_bps < 10000:
        raise ValueError(f"hop.fee_bps must be in basis points within [0, 10000), got {hop.fee_bps}")
    return _build_v2_compute_out(prog, amount_in, hop, 10000 - hop.fee_bps)


def _build_v3_quote(prog: Program, amount_in: Operand, hop: SwapHop, quoter_address: Address) -> Operand:
    """V3 pool quote — calls ``quoter.quoteExactInput`` (view-only)."""
    packed_path = encode_v3_path([hop.token_in, hop.token_out], [hop.fee_bps * 100])
    success = prog.call_contract(
        quoter_address,
        _V3_QUOTER_QUOTE_EXACT_INPUT_FN,
        packed_path,
        amount_in,
    )
    prog.assert_(success)
    return prog.returndata_word(0)


def _build_v2_direct_swap(prog: Program, amount_in: Operand, hop: SwapHop) -> Operand:
    """V2 pair direct swap.  Returns ``amount_out`` as an SSA :class:`Operand`."""
    if not 0 <= hop.fee_bps < 10000:
        raise ValueError(f"hop.fee_bps must be in basis points within [0, 10000), got {hop.fee_bps}")

    amount_out = _build_v2_compute_out(prog, amount_in, hop, 10000 - hop.fee_bps)

    # Transfer amount_in to the pool.
    transfer_success = prog.call_contract(
        hop.token_in,
        _ERC20_TRANSFER_FN,
        hop.pool,
        amount_in,
    )
    prog.assert_(transfer_success)

    # Call pool.swap(amount0Out, amount1Out, to, data).
    if hop.zero_for_one:
        # token0 in, token1 out: amount0Out=0, amount1Out=amount_out
        swap_success = prog.call_contract(
            hop.pool,
            _V2_PAIR_SWAP_FN,
            0,
            amount_out,
            hop.recipient,
            b"",
        )
    else:
        swap_success = prog.call_contract(
            hop.pool,
            _V2_PAIR_SWAP_FN,
            amount_out,
            0,
            hop.recipient,
            b"",
        )
    prog.assert_(swap_success)
    return amount_out


# ---------------------------------------------------------------------------
# Protocol resolution
# ---------------------------------------------------------------------------


_PROTOCOL_LOOKUP: dict[str, SwapProtocol] = {
    "uniswapv2": SwapProtocol.UNISWAP_V2,
    "uniswap_v2": SwapProtocol.UNISWAP_V2,
    "uniswap v2": SwapProtocol.UNISWAP_V2,
    "uniswapv3": SwapProtocol.UNISWAP_V3,
    "uniswap_v3": SwapProtocol.UNISWAP_V3,
    "uniswap v3": SwapProtocol.UNISWAP_V3,
    "uniswapv4": SwapProtocol.UNISWAP_V4,
    "uniswap_v4": SwapProtocol.UNISWAP_V4,
    "uniswap v4": SwapProtocol.UNISWAP_V4,
}


def _pool_to_swap_protocol(protocol_name: str) -> SwapProtocol:
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
        token_out=swap_action.token_out.address,
        fee_bps=pool.fee_bps,
        recipient=recipient,
        zero_for_one=swap_action.zero_for_one(),
    )


def _build_route_swap(prog: Program, amount_in: Operand, action: RouteSwap, recipient: Address) -> Operand:
    hop = _swap_hop_from_route_swap(action, recipient=recipient)
    if hop.protocol == SwapProtocol.UNISWAP_V4:
        raise NotImplementedError(
            "Uniswap V4 execution is not supported by DeFiVM; build the swap via UniversalRouter (V4_SWAP) instead"
        )
    if hop.protocol == SwapProtocol.UNISWAP_V3:
        return _build_v3_pool_swap(prog, amount_in, hop)
    return _build_v2_direct_swap(prog, amount_in, hop)


# ---------------------------------------------------------------------------
# High-level transaction builder
# ---------------------------------------------------------------------------

_EXECUTE_SELECTOR: bytes = keccak(text="execute(bytes)")[:4]


def build_swap_transaction(
    dag: RouteDAG,
    amount_in: int,
    vm_address: str,
    recipient: str,
    *,
    min_final_out: int = 0,
) -> SwapTransaction:
    """Compile a :class:`~pydefi.types.RouteDAG` into a DeFiVM ``execute(bytes)``
    transaction.

    The DAG self-contains its inputs, so no transient-storage parameters are
    needed.  Programs that require runtime parameters should be invoked via
    a composer (CCTP/OFT/ApproveProxy) that DELEGATECALLs the EVM interpreter
    after staging params in its own transient slots — transient storage is
    tx-scoped under EIP-1153, so values do not survive across transactions.
    """
    from pydefi.vm.dag import build_execution_program_for_dag

    program = build_execution_program_for_dag(
        dag,
        amount_in=amount_in,
        vm_address=vm_address,
        recipient=recipient,
        min_final_out=min_final_out,
    )
    calldata = _EXECUTE_SELECTOR + encode(["bytes"], [bytes(program)])
    return SwapTransaction(to=vm_address, data=calldata)


def swap_route_to_hops(route: SwapRoute, vm_address: str, recipient: str) -> list[SwapHop]:
    """Convert a :class:`~pydefi.types.SwapRoute` into :class:`SwapHop` descriptors."""
    hops: list[SwapHop] = []
    steps = route.steps
    n = len(steps)

    for i, step in enumerate(steps):
        try:
            protocol = _pool_to_swap_protocol(step.protocol)
        except ValueError:
            raise ValueError(f"swap_route_to_hops: unrecognised protocol {step.protocol!r} on step {i}.")

        zero_for_one = step.token_in.address < step.token_out.address
        hop_recipient = recipient if i == n - 1 else vm_address

        hops.append(
            SwapHop(
                protocol=protocol,
                pool=step.pool_address,
                token_in=step.token_in.address,
                token_out=step.token_out.address,
                fee_bps=step.fee,
                recipient=Address(hop_recipient),
                zero_for_one=zero_for_one,
            )
        )
    return hops
