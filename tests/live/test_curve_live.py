"""Live integration tests for Curve Finance against a public Ethereum RPC.

Covers the on-chain ``CurvePool`` quotes plus the local-pricing work: local
``get_dy`` / ``get_dx`` cross-checked against each pool's on-chain view
(stableswap legacy/NG, cryptoswap), registry discovery + kind auto-detection,
and meta-pool underlying pricing.
"""

import pytest

from pydefi.abi.amm import CURVE_POOL
from pydefi.amm.curve import CurveMetaPool, CurvePool, CurvePoolKind
from pydefi.types import Address, ChainId, Token, TokenAmount
from tests.addrs import DAI, USDC, USDT, WETH

# Curve 3pool on Ethereum mainnet
CURVE_3POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"

# Stable-NG: USDe / USDC factory pool (coins: USDe(18), USDC(6))
CURVE_USDE_USDC_NG = "0x02950460E2b9529D0E00284A5fA2d7bDF3fA4d72"
# Factory plain (A_PRECISION, no stored_rates): USDC / crvUSD (coins: USDC(6), crvUSD(18))
CURVE_USDC_CRVUSD_FACTORY = "0x4DEcE678ceceb27446b35C672dC7d61F30bAD69E"
# Cryptoswap (Curve V2): TricryptoUSDC (coins: USDC, WBTC, WETH)
CURVE_TRICRYPTO_USDC = "0x7F86Bf177Dd4F3494b841a37e810A34dD56c829B"
# Factory meta-pool: MIM / 3CRV (coin 0 = MIM, coin 1 = 3pool LP)
CURVE_MIM_3CRV = "0x5a6A4D54456819380173272A5E8E9B9904BdF41B"

MIM = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address("0x99D8a9C45b2ecA8864373A26D1459e3Dff1e17F3"),
    symbol="MIM",
    decimals=18,
)

USDE = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address("0x4c9EDD5852cd905f086C759E8383e09bff1E68B3"),
    symbol="USDe",
    decimals=18,
)
CRVUSD = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address("0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E"),
    symbol="crvUSD",
    decimals=18,
)
WBTC = Token(
    chain_id=ChainId.ETHEREUM,
    address=Address("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"),
    symbol="WBTC",
    decimals=8,
)


@pytest.fixture
def threepool(eth_w3) -> CurvePool:
    """An on-chain-quoting CurvePool bound to the Curve 3pool (DAI/USDC/USDT)."""
    return CurvePool(w3=eth_w3, pool_address=CURVE_3POOL, tokens=[DAI, USDC, USDT])


@pytest.mark.live
class TestCurveLive:
    """Live on-chain tests for CurvePool (3pool)."""

    async def test_get_dy_dai_to_usdc(self, threepool):
        """1 000 DAI → USDC should return approximately 1 000 USDC (±1%)."""
        dy = await threepool.get_dy(DAI, USDC, 1_000 * 10**18)
        assert 990 * 10**6 < dy < 1_010 * 10**6, f"DAI→USDC out of range: {dy / 10**6:.4f}"

    async def test_get_dy_usdc_to_dai(self, threepool):
        """1 000 USDC → DAI should return approximately 1 000 DAI (±1%)."""
        dy = await threepool.get_dy(USDC, DAI, 1_000 * 10**6)
        assert 990 * 10**18 < dy < 1_010 * 10**18, f"USDC→DAI out of range: {dy / 10**18:.4f}"

    async def test_get_dy_usdc_to_usdt(self, threepool):
        """1 000 USDC → USDT should return approximately 1 000 USDT (±1%)."""
        dy = await threepool.get_dy(USDC, USDT, 1_000 * 10**6)
        assert 990 * 10**6 < dy < 1_010 * 10**6, f"USDC→USDT out of range: {dy / 10**6:.4f}"

    async def test_get_amounts_out(self, threepool):
        """get_amounts_out wrapper returns a two-element list."""
        result = await threepool.get_amounts_out(TokenAmount.from_human(DAI, "100"), [DAI, USDC])
        assert len(result) == 2
        assert result[0].token == DAI
        assert result[1].token == USDC
        assert result[1].amount > 0

    async def test_build_swap_route(self, threepool):
        """build_swap_route should return a valid SwapRoute for DAI → USDC."""
        route = await threepool.build_swap_route(TokenAmount.from_human(DAI, "100"), USDC)
        assert route.token_in == DAI
        assert route.token_out == USDC
        assert len(route.steps) == 1
        assert route.steps[0].protocol == "Curve"
        assert route.amount_out.amount > 0


@pytest.mark.live
class TestCurveLocalPricing:
    """Cross-check local Curve math against the pool's on-chain ``get_dy``.

    A relative tolerance is used (rather than bit-exact equality) because
    ``load_state`` and the reference ``get_dy`` call may land on adjacent blocks
    on a public RPC.  Bit-exact regression values are locked in
    ``tests/test_curve_math.py``.
    """

    @staticmethod
    async def _assert_matches(eth_w3, addr, tokens, kind, src, dst, dx, tol):
        # State-loaded pool prices locally; a fresh pool falls back to on-chain.
        local_pool = CurvePool(w3=eth_w3, pool_address=addr, tokens=tokens, kind=kind)
        await local_pool.load_state()
        onchain = await CurvePool(w3=eth_w3, pool_address=addr, tokens=tokens, kind=kind).get_dy(src, dst, dx)
        local = local_pool.get_dy_local(src, dst, dx)
        assert local > 0 and onchain > 0
        assert abs(local - onchain) / onchain < tol, f"local={local} onchain={onchain}"

    async def test_legacy_3pool_local_matches_onchain(self, eth_w3):
        await self._assert_matches(
            eth_w3, CURVE_3POOL, [DAI, USDC, USDT], CurvePoolKind.STABLE_LEGACY, DAI, USDC, 50_000 * 10**18, 1e-5
        )

    async def test_stable_ng_local_matches_onchain(self, eth_w3):
        await self._assert_matches(
            eth_w3, CURVE_USDE_USDC_NG, [USDE, USDC], CurvePoolKind.STABLE_NG, USDE, USDC, 100_000 * 10**18, 1e-4
        )

    async def test_stable_factory_local_matches_onchain(self, eth_w3):
        await self._assert_matches(
            eth_w3,
            CURVE_USDC_CRVUSD_FACTORY,
            [USDC, CRVUSD],
            CurvePoolKind.STABLE_FACTORY,
            USDC,
            CRVUSD,
            100_000 * 10**6,
            1e-4,
        )

    async def test_crypto_v2_tricrypto_local_matches_onchain(self, eth_w3):
        # Cryptoswap reuses the pool's stored D; tolerate minor block drift.
        await self._assert_matches(
            eth_w3, CURVE_TRICRYPTO_USDC, [USDC, WBTC, WETH], CurvePoolKind.CRYPTO_V2, USDC, WETH, 50_000 * 10**6, 1e-3
        )

    async def test_stable_ng_get_dx_matches_onchain(self, eth_w3):
        # Exact-output: local closed-form get_dx vs the pool's on-chain get_dx.
        tokens = [USDE, USDC]
        pool = CurvePool(w3=eth_w3, pool_address=CURVE_USDE_USDC_NG, tokens=tokens, kind=CurvePoolKind.STABLE_NG)
        await pool.load_state()
        dy = 50_000 * 10**6  # want 50k USDC out
        onchain = await CURVE_POOL.fns.get_dx(0, 1, dy).call(eth_w3, to=CURVE_USDE_USDC_NG)
        local = pool.get_dx_local(USDE, USDC, dy)
        assert local > 0
        assert abs(local - onchain) / onchain < 1e-4, f"local={local} onchain={onchain}"


@pytest.mark.live
class TestCurveDiscovery:
    """Registry-based pool discovery and pool-kind auto-detection."""

    async def test_detect_kind(self, eth_w3):
        assert await CurvePool.detect_kind(eth_w3, CURVE_3POOL) == CurvePoolKind.STABLE_LEGACY
        assert await CurvePool.detect_kind(eth_w3, CURVE_USDC_CRVUSD_FACTORY) == CurvePoolKind.STABLE_FACTORY
        assert await CurvePool.detect_kind(eth_w3, CURVE_USDE_USDC_NG) == CurvePoolKind.STABLE_NG
        assert await CurvePool.detect_kind(eth_w3, CURVE_TRICRYPTO_USDC) == CurvePoolKind.CRYPTO_V2

    async def test_find_pools_includes_3pool(self, eth_w3):
        pools = await CurvePool.find_pools(eth_w3, DAI, USDC)
        assert any(p.lower() == CURVE_3POOL.lower() for p in pools)

    async def test_discover_prices_locally(self, eth_w3):
        # Discover a DAI/USDC pool end-to-end and check it prices vs on-chain.
        pool = await CurvePool.discover(eth_w3, DAI, USDC)
        assert pool is not None
        assert {t.address for t in pool.tokens} >= {DAI.address, USDC.address}
        dx = 10_000 * 10**18
        local = pool.get_dy_local(DAI, USDC, dx)
        onchain = await CurvePool(eth_w3, pool.router_address, pool.tokens, kind=pool.kind).get_dy(DAI, USDC, dx)
        assert abs(local - onchain) / onchain < 1e-3, f"local={local} onchain={onchain}"


@pytest.mark.live
class TestCurveMetaPool:
    """Meta-pool underlying pricing vs the pool's on-chain get_dy_underlying."""

    async def _meta(self, eth_w3) -> CurveMetaPool:
        base = CurvePool(eth_w3, CURVE_3POOL, [DAI, USDC, USDT], kind=CurvePoolKind.STABLE_LEGACY)
        meta = CurveMetaPool(eth_w3, CURVE_MIM_3CRV, MIM, base, kind=CurvePoolKind.STABLE_FACTORY)
        await meta.load_state()
        return meta

    async def test_primary_to_base_coin(self, eth_w3):
        meta = await self._meta(eth_w3)
        dx = 50_000 * 10**18  # MIM -> USDC (underlying j=2)
        local = meta.get_dy_underlying(MIM, USDC, dx)
        onchain = await CURVE_POOL.fns.get_dy_underlying(0, 2, dx).call(eth_w3, to=CURVE_MIM_3CRV)
        assert local > 0
        assert abs(local - onchain) / onchain < 2e-3, f"local={local} onchain={onchain}"

    async def test_base_coin_to_primary(self, eth_w3):
        meta = await self._meta(eth_w3)
        dx = 50_000 * 10**6  # USDT (underlying i=3) -> MIM
        local = meta.get_dy_underlying(USDT, MIM, dx)
        onchain = await CURVE_POOL.fns.get_dy_underlying(3, 0, dx).call(eth_w3, to=CURVE_MIM_3CRV)
        assert local > 0
        assert abs(local - onchain) / onchain < 2e-3, f"local={local} onchain={onchain}"
