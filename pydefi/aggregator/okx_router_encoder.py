"""
Encoder for building calldata targeting the OKX DexRouter's ``dagSwapTo`` entry
point.

The encoder translates a RouteDAG (or raw edge descriptors) into the
``BaseRequest`` / ``RouterPath[]`` format that the OKX DexRouter expects,
then ABI-encodes the full ``dagSwapTo`` calldata.

Key concepts
------------
**RawData encoding (one ``uint256`` per edge)** ::

    ┌────────────────┬──────────┬───────────┬────────────┬──────────┬─────────────────────┐
    │ reverse (bit255)│ inputIdx │ outputIdx │ weight(bps)│ (unused) │ pool address (160b)  │
    └────────────────┴──────────┴───────────┴────────────┴──────────┴─────────────────────┘

**Bit masks** (from the DexRouter's ``CommonUtils.sol``):

- ``_ADDRESS_MASK``    = low 160 bits — pool/token address
- ``_WEIGHT_MASK``     = bits 160–175 — split weight in basis points (0–10000)
- ``_OUTPUT_INDEX_MASK`` = bits 176–183 — DAG output node index
- ``_INPUT_INDEX_MASK``  = bits 184–191 — DAG input node index
- ``_REVERSE_MASK``      = bit 255 — swap direction (1 = sellQuote, 0 = sellBase)

**Transfer modes** (packed into ``fromToken`` high bits):

- ``_MODE_LEGACY``    = 0x0000… — normal approve-proxy transfer
- ``_MODE_NO_TRANSFER`` = 0x0E00… (bit 251) — skip transfer
- ``_MODE_BY_INVEST``   = 0x0C00… (bit 250) — direct ERC-20 transfer
- ``_MODE_PERMIT2``     = 0x0A00… (bit 249) — Permit2 (reserved)

Usage
-----

    from pydefi.aggregator.okx_router_encoder import (
        build_dag_swap_calldata,
        encode_edge_raw_data,
    )

    raw_data = encode_edge_raw_data(
        pool_address=pool_address,
        reverse=False,
        weight_bps=10000,
        input_index=0,
        output_index=1,
    )

    calldata = build_dag_swap_calldata(
        order_id=42,
        receiver=receiver_address,
        from_token=from_address,
        to_token=to_address,
        amount=amount_in,
        min_return=min_amount_out,
        deadline=deadline,
        paths=paths,
    )
"""

from __future__ import annotations

from pydefi import Token
from pydefi._math import MAX_BPS
from pydefi.abi.dex_aggregator import OKX_DEX_ROUTER, BaseRequest, RouterPath
from pydefi.pathfinder.dag import RouteDAG
from pydefi.types import Address, RouteAction, RouteSplit, RouteSwap, SwapProtocol

# ---------------------------------------------------------------------------
# DexRouter bit-mask constants (from CommonUtils.sol)
# ---------------------------------------------------------------------------

_REVERSE_MASK: int = 1 << 255
_ADDRESS_MASK: int = (1 << 160) - 1

# Weight occupies bytes 12-13 of the uint256 word (bits 160-175)
_WEIGHT_OFFSET: int = 160

# Output index bytes 10-11 (bits 176-183)
_OUTPUT_INDEX_OFFSET: int = 176

# Input index bytes 8-9 (bits 184-191)
_INPUT_INDEX_OFFSET: int = 184

# Transfer mode masks — packed into the high bits of fromToken
_MODE_LEGACY: int = 0
_MODE_NO_TRANSFER: int = 1 << 251
_MODE_BY_INVEST: int = 1 << 250
_MODE_PERMIT2: int = 1 << 249

# dagSwapTo function selector
_DAG_SWAP_TO_SELECTOR: bytes = OKX_DEX_ROUTER.fns.dagSwapTo.selector


class RouterPathDescriptor:
    """Descriptor for one DAG node passed to :func:`build_dag_swap_calldata`.

    Attributes:
        mix_adapters: Adapter contract addresses, one per edge.
            Must be known a priori (chain-specific OKX adapter deployments).
        asset_to: Destination for each edge's pre-swap token transfer.
            Typically the pool/pair address.
        raw_data: Encoded edge parameters, one ``uint256`` per edge.
            Use :func:`encode_edge_raw_data` to build each word.
        extra_data: Bytes passed to the adapter's ``sellBase``/``sellQuote``.
            Usually empty for standard V2/V3 swaps.
        from_token: Source token address for this node (20 raw bytes).
        mode: Transfer mode for this node's fromToken.
            Defaults to ``_MODE_LEGACY``.
    """

    __slots__ = ("mix_adapters", "asset_to", "raw_data", "extra_data", "from_token", "mode")

    def __init__(
        self,
        mix_adapters: list[Address],
        asset_to: list[Address],
        raw_data: list[int],
        *,
        extra_data: list[bytes] | None = None,
        from_token: Address,
        mode: int = _MODE_LEGACY,
    ) -> None:
        self.mix_adapters = mix_adapters
        self.asset_to = asset_to
        self.raw_data = raw_data
        self.extra_data = extra_data or [b""] * len(mix_adapters)
        self.from_token = from_token
        self.mode = mode


# ---------------------------------------------------------------------------
# RawData encoder
# ---------------------------------------------------------------------------


def encode_edge_raw_data(
    pool_address: Address,
    *,
    reverse: bool = False,
    weight_bps: int = MAX_BPS,
    input_index: int = 0,
    output_index: int = 1,
) -> int:
    """Encode one edge's ``rawData`` word for the OKX DexRouter.

    Args:
        pool_address: The pool/pair address (20 bytes).
        reverse: If ``True`` the adapter calls ``sellQuote`` (reverse
            direction); otherwise ``sellBase``.
        weight_bps: Split weight in basis points (0–10000).  Defaults to
            10000 (100%, single edge).
        input_index: Source DAG node index.
        output_index: Target DAG node index.

    Returns:
        A ``uint256`` ready for the ``RouterPath.rawData`` array.

    Raises:
        ValueError: If any argument is out of range.
    """
    if not 0 <= weight_bps <= MAX_BPS:
        raise ValueError(f"weight_bps must be in [0, {MAX_BPS}], got {weight_bps}")
    if not 0 <= input_index <= 0xFF:
        raise ValueError(f"input_index must be in [0, 255], got {input_index}")
    if not 0 <= output_index <= 0xFF:
        raise ValueError(f"output_index must be in [0, 255], got {output_index}")
    if len(pool_address) != 20:
        raise ValueError(f"pool_address must be 20 bytes, got {len(pool_address)}")

    addr_int = int.from_bytes(bytes(pool_address), "big")

    raw = addr_int & _ADDRESS_MASK
    raw |= (weight_bps & 0xFFFF) << _WEIGHT_OFFSET
    raw |= (output_index & 0xFF) << _OUTPUT_INDEX_OFFSET
    raw |= (input_index & 0xFF) << _INPUT_INDEX_OFFSET
    if reverse:
        raw |= _REVERSE_MASK

    return raw


# ---------------------------------------------------------------------------
# Calldata builder
# ---------------------------------------------------------------------------


def _pack_from_token(address: Address, mode: int = _MODE_LEGACY) -> int:
    """Pack a 20-byte address with a transfer mode into a uint256.

    Args:
        address: The token address (20 bytes).
        mode: Transfer mode (``_MODE_LEGACY``, ``_MODE_NO_TRANSFER``,
            ``_MODE_BY_INVEST``, or ``_MODE_PERMIT2``).

    Returns:
        Packed uint256 (low 160 bits = address, high bits = mode).
    """
    addr_int = int.from_bytes(bytes(address), "big")
    return addr_int | mode


def build_dag_swap_calldata(
    *,
    order_id: int,
    receiver: Address,
    from_token: Address,
    to_token: Address,
    amount: int,
    min_return: int,
    deadline: int,
    paths: list[RouterPathDescriptor],
) -> bytes:
    """Build the full ``dagSwapTo`` calldata (selector + ABI arguments).

    Delegates to the canonical :class:`OKX_DEX_ROUTER` ABI so all struct
    encoding is handled by the shared :class:`BaseRequest` /
    :class:`RouterPath` definitions in :mod:`pydefi.abi.dex_aggregator`.

    Args:
        order_id: Arbitrary order tracker (emitted as an event).
        receiver: Address that receives the swap output tokens.
        from_token: Source token address (20 bytes).
        to_token: Destination token address (20 bytes).
        amount: Exact input amount (raw integer).
        min_return: Minimum acceptable output amount (slippage guard).
        deadline: Unix timestamp after which the transaction reverts.
        paths: List of :class:`RouterPathDescriptor`, one per DAG node.

    Returns:
        Complete calldata bytes ready for ``eth_sendTransaction.data``.
    """
    if not paths:
        raise ValueError("build_dag_swap_calldata: at least one RouterPath is required")

    base_req = BaseRequest(
        fromToken=_pack_from_token(from_token),
        toToken=to_token,
        fromTokenAmount=amount,
        minReturnAmount=min_return,
        deadLine=deadline,
    )

    router_paths: list[RouterPath] = []
    for p in paths:
        router_paths.append(
            RouterPath(
                mixAdapters=p.mix_adapters,
                assetTo=p.asset_to,
                rawData=p.raw_data,
                extraData=p.extra_data,
                fromToken=_pack_from_token(p.from_token, p.mode),
            )
        )

    return OKX_DEX_ROUTER.fns.dagSwapTo(order_id, receiver, base_req, router_paths).data


# ---------------------------------------------------------------------------
# RouteDAG → RouterPath converter
# ---------------------------------------------------------------------------

# Default OKX adapter addresses (Ethereum mainnet).
# These addresses must be verified for each target chain — the DexRouter
# deploys matched adapter sets.  Pass ``adapter_overrides`` for non-Ethereum
# chains or when the defaults are incorrect.
_DEFAULT_V3_ADAPTER = "0x6B2C512C92be770d28B8a71386A80a0C8E64C5E6"
_DEFAULT_V2_ADAPTER = "0x5a32DC56Ff11B53C5eB76d1B9D13332f3CB021AA"

# Protocol → default adapter map
_PROTOCOL_ADAPTER: dict[str, str] = {
    SwapProtocol.UNISWAP_V3: _DEFAULT_V3_ADAPTER,
    SwapProtocol.UNISWAP_V2: _DEFAULT_V2_ADAPTER,
}


# Normalisation map: pool protocol strings → canonical adapter lookup keys.
# The OKX DexRouter deploys protocol-specific adapters; we map common
# spellings ("UniswapV2", "Uniswap V2", "uniswapv2") to a canonical key.
_PROTOCOL_NORMALISE: dict[str, str] = {
    "uniswapv2": "uniswap_v2",
    "uniswap v2": "uniswap_v2",
    "uniswapv3": "uniswap_v3",
    "uniswap v3": "uniswap_v3",
}


def _protocol_to_adapter(
    protocol: str | SwapProtocol,
    adapter_overrides: dict[str, str] | None = None,
) -> Address:
    """Resolve a swap protocol to an OKX DexRouter adapter address."""
    raw = protocol if isinstance(protocol, str) else protocol.value
    key = raw.lower().replace(" ", "_").replace("-", "_")
    # Normalise through the explicit map, then fall back to the raw key.
    key = _PROTOCOL_NORMALISE.get(key, key)

    if adapter_overrides and key in adapter_overrides:
        addr = adapter_overrides[key]
    else:
        addr = _PROTOCOL_ADAPTER.get(key)
        if addr is None:
            raise ValueError(f"protocol {raw!r}: no OKX adapter address known; pass it via adapter_overrides")

    if isinstance(addr, str):
        return Address.fromhex(addr[2:] if addr.startswith("0x") else addr)
    if isinstance(addr, (bytes, bytearray, memoryview)):
        return Address(addr)
    return Address(addr)


def route_dag_to_router_paths(
    dag: RouteDAG,
    *,
    adapter_overrides: dict[str, str] | None = None,
    mode: int = _MODE_LEGACY,
) -> list[RouterPathDescriptor]:
    """Convert a pydefi :class:`~pydefi.types.RouteDAG` into OKX RouterPath descriptors.

    Each sequential swap or split becomes one ``RouterPath`` node.
    Edges are mapped to adapter addresses based on the pool protocol.

    **Supported topologies:**

    * Linear routes (no splits) — each swap is a separate node with one edge.
    * Split routes where every leg contains exactly **one** swap.  The split
      node has N edges (one per leg) and their weights sum to 10000.
    * Multi-hop legs (more than one swap inside a leg) and nested splits
      are **not** supported — the OKX RouterPath models each edge as a
      single adapter call.

    Args:
        dag: The route DAG to convert.  Must have ``merge()`` already called.
        adapter_overrides: Optional ``{protocol_name: adapter_address}``
            mapping to override default OKX adapter addresses.
        mode: Transfer mode for the initial fromToken.  Defaults to
            ``_MODE_LEGACY``.

    Returns:
        List of :class:`RouterPathDescriptor`, one per DAG node.

    Raises:
        ValueError: If the DAG contains unsupported topologies or a protocol
            has no known adapter.
    """
    if dag.token_in is None:
        raise ValueError("RouteDAG.from_token() must be called before conversion")
    if dag._split_stack:
        raise ValueError("RouteDAG has unmerged split legs — call merge() first")

    return _walk_actions(
        dag.token_in,
        list(dag.actions),
        mode=mode,
        adapter_overrides=adapter_overrides,
    )


def _walk_actions(
    token_in,
    actions: list[RouteAction],
    *,
    mode: int,
    adapter_overrides: dict[str, str] | None,
    start_index: int = 0,
) -> list[RouterPathDescriptor]:
    """Walk a flat action list (no nested sub-lists) into RouterPath nodes.

    Each action maps to one node.  Splits with multi-hop legs are rejected.
    """
    result: list[RouterPathDescriptor] = []
    current_token = token_in
    idx = start_index
    current_mode = mode

    for action in actions:
        if isinstance(action, RouteSwap):
            result.append(_swap_to_node(action, current_token, idx, current_mode, adapter_overrides))
            current_token = action.token_out
            idx += 1
            current_mode = _MODE_LEGACY

        elif isinstance(action, RouteSplit):
            # Verify: every leg must be exactly one swap (no multi-hop legs).
            for leg in action.legs:
                if len(leg.actions) != 1 or not isinstance(leg.actions[0], RouteSwap):
                    raise ValueError(
                        "route_dag_to_router_paths: split legs must contain exactly one RouteSwap. "
                        "Multi-hop legs are not supported by the OKX RouterPath format."
                    )
            result.append(_split_to_node(action, current_token, idx, current_mode, adapter_overrides))
            current_token = action.token_out
            idx += 1
            current_mode = _MODE_LEGACY

        else:
            raise TypeError(f"unsupported DAG action type: {type(action).__name__}")

    return result


def _swap_to_node(
    swap: RouteSwap,
    from_token: Token,
    node_index: int,
    mode: int,
    adapter_overrides: dict[str, str] | None,
) -> RouterPathDescriptor:
    """Encode a single RouteSwap as a one-edge RouterPath node."""
    pool = swap.pool
    adapter = _protocol_to_adapter(pool.protocol, adapter_overrides)

    raw = encode_edge_raw_data(
        pool_address=pool.pool_address,
        reverse=not swap.zero_for_one(),
        weight_bps=MAX_BPS,
        input_index=node_index,
        output_index=node_index + 1,
    )
    return RouterPathDescriptor(
        mix_adapters=[adapter],
        asset_to=[pool.pool_address],
        raw_data=[raw],
        from_token=from_token.address,
        mode=mode,
    )


def _split_to_node(
    split: RouteSplit,
    from_token: Token,
    node_index: int,
    mode: int,
    adapter_overrides: dict[str, str] | None,
) -> RouterPathDescriptor:
    """Encode a RouteSplit as one node with one edge per leg."""
    adapters: list[Address] = []
    asset_tos: list[Address] = []
    raw_datas: list[int] = []

    for leg in split.legs:
        swap = leg.actions[0]  # guaranteed single swap by caller
        pool = swap.pool
        adapter = _protocol_to_adapter(pool.protocol, adapter_overrides)

        adapters.append(adapter)
        asset_tos.append(pool.pool_address)
        raw_datas.append(
            encode_edge_raw_data(
                pool_address=pool.pool_address,
                reverse=not swap.zero_for_one(),
                weight_bps=leg.fraction_bps,
                input_index=node_index,
                output_index=node_index + 1,
            )
        )

    return RouterPathDescriptor(
        mix_adapters=adapters,
        asset_to=asset_tos,
        raw_data=raw_datas,
        from_token=from_token.address,
        mode=mode,
    )
