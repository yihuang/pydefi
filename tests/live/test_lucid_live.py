"""Live integration tests for the Lucid bridge.

Parametrized over both supported source chains:

* MANTRA (5888) — Hyperlane adapter, pinned in ``deployments.py``.
* Kite (2366) — LayerZero adapter, pinned in ``deployments.py``.

Run with::

    pytest -m live tests/live/test_lucid_live.py
"""

from __future__ import annotations

import pytest
from web3 import AsyncWeb3

from pydefi.abi.bridge import LUCID_ASSET_CONTROLLER
from pydefi.bridge.lucid import LucidBridge
from pydefi.deployments import get_address, get_token
from pydefi.rpc import get_w3
from pydefi.types import Address, ChainId, TokenAmount

# Per-source lane table: controller -> (token symbol on source, destination chain id).
_LANES: dict[int, dict[str, tuple[str, int]]] = {
    ChainId.MANTRA: {
        "LUCID_USDC_CONTROLLER": ("USDC", ChainId.BASE),
        "LUCID_WETH_CONTROLLER": ("WETH", ChainId.BASE),
        "LUCID_USDT_CONTROLLER": ("USDT", ChainId.OPTIMISM),
    },
    ChainId.KITE: {
        "LUCID_USDC_CONTROLLER": ("USDC.e", ChainId.AVALANCHE),
        "LUCID_WETH_CONTROLLER": ("WETH", ChainId.AVALANCHE),
        "LUCID_USDT_CONTROLLER": ("USDT", ChainId.CELO),
    },
}

_SOURCES = [ChainId.MANTRA, ChainId.KITE]


@pytest.fixture(params=_SOURCES, ids=["mantra", "kite"])
async def src(request) -> tuple[int, AsyncWeb3]:
    """Yield ``(src_chain_id, w3)`` for each supported source chain."""
    src_chain_id = request.param
    w3 = await get_w3(src_chain_id)
    return src_chain_id, w3


def _ctrl_addr(name: str, src_chain_id: int) -> Address:
    return get_address(name, src_chain_id)


_ADAPTER_BY_SRC = {
    ChainId.KITE: "LUCID_LAYERZERO_ADAPTER",
    ChainId.MANTRA: "LUCID_HYPERLANE_ADAPTER",
}


def _adapter_addr(src_chain_id: int) -> Address:
    return get_address(_ADAPTER_BY_SRC[src_chain_id], src_chain_id)


def _all_controllers() -> list[tuple[int, str]]:
    return [(src, ctrl) for src in _SOURCES for ctrl in _LANES[src]]


def _id(case: tuple[int, str]) -> str:
    return f"{ChainId(case[0]).name.lower()}-{case[1].removeprefix('LUCID_').removesuffix('_CONTROLLER').lower()}"


@pytest.mark.live
class TestLucidControllersLive:
    """Read-only checks against every deployed Lucid controller."""

    async def test_src_rpc_chain_id(self, src):
        src_chain_id, w3 = src
        assert await w3.eth.chain_id == src_chain_id

    @pytest.mark.parametrize("case", _all_controllers(), ids=_id)
    async def test_controller_wraps_expected_token(self, case):
        src_chain_id, controller_name = case
        w3 = await get_w3(src_chain_id)
        token_name = _LANES[src_chain_id][controller_name][0]

        ctrl = get_address(controller_name, src_chain_id)
        expected = get_token(token_name, src_chain_id)
        on_chain = await LUCID_ASSET_CONTROLLER.fns.token().call(w3, to=ctrl)
        assert on_chain.lower() == ("0x" + bytes(expected.address).hex()).lower()

    @pytest.mark.parametrize("case", _all_controllers(), ids=_id)
    async def test_controller_destination_active(self, case):
        """Wiring is ours to assert and must come first, pause flags are operator's and only skip."""
        src_chain_id, controller_name = case
        w3 = await get_w3(src_chain_id)
        dst_chain_id = _LANES[src_chain_id][controller_name][1]
        ctrl = get_address(controller_name, src_chain_id)

        dest_ctrl = await LUCID_ASSET_CONTROLLER.fns.getControllerForChain(dst_chain_id).call(w3, to=ctrl)
        assert int(dest_ctrl, 16) != 0, f"{controller_name}: no destination configured for chain {dst_chain_id}"

        if await LUCID_ASSET_CONTROLLER.fns.paused().call(w3, to=ctrl):
            pytest.skip(f"{controller_name} on {ChainId(src_chain_id).name} is globally paused")
        if await LUCID_ASSET_CONTROLLER.fns.transfersPausedTo(dst_chain_id).call(w3, to=ctrl):
            pytest.skip(f"{controller_name}: transfers to {ChainId(dst_chain_id).name} are paused")


@pytest.mark.live
class TestLucidBridgeLive:
    """End-to-end exercise of ``LucidBridge`` against live contracts on each source."""

    async def _client(self, src_chain_id: int, w3: AsyncWeb3, controller_name: str) -> LucidBridge:
        return LucidBridge(
            w3=w3,
            src_chain_id=src_chain_id,
            dst_chain_id=_LANES[src_chain_id][controller_name][1],
            controller_address=_ctrl_addr(controller_name, src_chain_id),
            adapter_address=_adapter_addr(src_chain_id),
        )

    async def test_quote_native_fee_positive_usdc(self, src):
        src_chain_id, w3 = src
        bridge = await self._client(src_chain_id, w3, "LUCID_USDC_CONTROLLER")
        fee = await bridge.quote_native_fee(amount=10**6, recipient=Address(b"\xaa" * 20))
        assert isinstance(fee, int) and fee > 0

    async def test_get_quote_shape_usdc(self, src):
        src_chain_id, w3 = src
        bridge = await self._client(src_chain_id, w3, "LUCID_USDC_CONTROLLER")
        token_name = _LANES[src_chain_id]["LUCID_USDC_CONTROLLER"][0]
        usdc = get_token(token_name, src_chain_id)
        amount_in = TokenAmount.from_human(usdc, "10")
        # token_out only needs matching symbol/decimals; LucidBridge does not
        # validate the destination address.
        quote = await bridge.get_quote(usdc, usdc, amount_in)

        assert quote.protocol == "Lucid"
        assert quote.amount_out.amount == amount_in.amount
        assert quote.bridge_fee.amount > 0
        assert quote.bridge_fee.token.symbol == "NATIVE"

    async def test_build_bridge_tx_shape_usdc(self, src):
        src_chain_id, w3 = src
        bridge = await self._client(src_chain_id, w3, "LUCID_USDC_CONTROLLER")
        token_name = _LANES[src_chain_id]["LUCID_USDC_CONTROLLER"][0]
        usdc = get_token(token_name, src_chain_id)
        amount_in = TokenAmount.from_human(usdc, "1")
        recipient = Address(b"\xaa" * 20)
        tx = await bridge.build_bridge_tx(usdc, usdc, amount_in, recipient)

        assert tx["to"] == _ctrl_addr("LUCID_USDC_CONTROLLER", src_chain_id)
        assert tx["data"].startswith("0x")
        assert int(tx["value"]) > 0
        assert int(tx["gas"]) > 0
