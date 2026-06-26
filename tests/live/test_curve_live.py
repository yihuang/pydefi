"""Live integration tests for Curve Finance.

* async RPC checks for :class:`pydefi.amm.curve.CurvePool`
* local-pricing parity against on-chain views
* registry discovery and pool-kind detection
* meta-pool underlying pricing
* titanoboa fork-based bit-exact cross-checks for :mod:`pydefi.amm.curve_math`
"""

from __future__ import annotations

import json
from typing import Any, cast

import boa
import pytest

from pydefi.abi.amm import CURVE_POOL
from pydefi.amm import curve_math as m
from pydefi.amm.curve import CurveMetaPool, CurvePool, CurvePoolKind
from pydefi.types import Address, ChainId, Token, TokenAmount
from tests.addrs import DAI, USDC, USDT, WETH
from tests.live.conftest import ETH_RPC_URL

# Curve 3pool on Ethereum mainnet
CURVE_3POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"

# Stable-NG: USDe / USDC factory pool (coins: USDe(18), USDC(6))
CURVE_USDE_USDC_NG = "0x02950460E2b9529D0E00284A5fA2d7bDF3fA4d72"
# Factory plain (A_PRECISION, no stored_rates): USDC / crvUSD (coins: USDC(6), crvUSD(18))
CURVE_USDC_CRVUSD_FACTORY = "0x4DEcE678ceceb27446b35C672dC7d61F30bAD69E"
# Cryptoswap (Curve V2): TricryptoUSDC (coins: USDC, WBTC, WETH)
CURVE_TRICRYPTO_USDC = "0x7F86Bf177Dd4F3494b841a37e810A34dD56c829B"
# Cryptoswap legacy tricrypto2 and its deployed Math3 helper.
CURVE_TRICRYPTO2 = "0xD51a44d3FaE010294C616388b506AcdA1bfAAE46"
TRICRYPTO2_MATH = "0x8F68f4810CcE3194B6cB6F3d50fa58c2c9bDD1d5"
# Factory meta-pool: MIM / 3CRV (coin 0 = MIM, coin 1 = 3pool LP)
CURVE_MIM_3CRV = "0x5a6A4D54456819380173272A5E8E9B9904BdF41B"
THREE_CRV = "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490"
# Stable-NG meta: USD1 / crv2pool (coin 0 = USD1(18), coin 1 = NG 2pool LP)
CURVE_USD1_NG_META = "0xC09e82f81Cb811DB0922dD48206fc2e212322caf"
# Its NG base pool: USDC/USDT "crv2pool"
CURVE_2POOL_NG = "0x4f493B7dE8aAC7d55F71853688b1F7C8F0243C85"

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

THREEPOOL_DECIMALS = [18, 6, 6]
USDE_USDC_DECIMALS = [18, 6]
TRICRYPTO_DECIMALS = [6, 8, 18]


def _view(name: str, ins: list[str], outs: list[str]) -> dict:
    return {
        "type": "function",
        "stateMutability": "view",
        "name": name,
        "inputs": [{"name": f"arg{k}", "type": t} for k, t in enumerate(ins)],
        "outputs": [{"name": "", "type": t} for t in outs],
    }


STABLE_ABI = json.dumps(
    [
        _view("get_dy", ["int128", "int128", "uint256"], ["uint256"]),
        _view("get_dx", ["int128", "int128", "uint256"], ["uint256"]),
        _view("get_dy_underlying", ["int128", "int128", "uint256"], ["uint256"]),
        _view("dynamic_fee", ["int128", "int128"], ["uint256"]),
        _view("coins", ["uint256"], ["address"]),
        _view("balances", ["uint256"], ["uint256"]),
        _view("A", [], ["uint256"]),
        _view("A_precise", [], ["uint256"]),
        _view("fee", [], ["uint256"]),
        _view("stored_rates", [], ["uint256[]"]),
        _view("offpeg_fee_multiplier", [], ["uint256"]),
        _view("get_virtual_price", [], ["uint256"]),
        _view("totalSupply", [], ["uint256"]),
        _view("calc_token_amount", ["uint256[3]", "bool"], ["uint256"]),
        _view("calc_withdraw_one_coin", ["uint256", "int128"], ["uint256"]),
    ]
)

CRYPTO_ABI = json.dumps(
    [
        _view("get_dy", ["uint256", "uint256", "uint256"], ["uint256"]),
        _view("balances", ["uint256"], ["uint256"]),
        _view("A", [], ["uint256"]),
        _view("gamma", [], ["uint256"]),
        _view("D", [], ["uint256"]),
        _view("mid_fee", [], ["uint256"]),
        _view("out_fee", [], ["uint256"]),
        _view("fee_gamma", [], ["uint256"]),
        _view("price_scale", ["uint256"], ["uint256"]),
        _view("fee_calc", ["uint256[3]"], ["uint256"]),
    ]
)

MATH3_ABI = json.dumps(
    [
        _view("newton_D", ["uint256", "uint256", "uint256[3]"], ["uint256"]),
        _view("newton_y", ["uint256", "uint256", "uint256[3]", "uint256", "uint256"], ["uint256"]),
        _view("geometric_mean", ["uint256[3]", "bool"], ["uint256"]),
    ]
)

ERC20_ABI = json.dumps(
    [
        _view("balanceOf", ["address"], ["uint256"]),
        _view("totalSupply", [], ["uint256"]),
        {
            "type": "function",
            "stateMutability": "nonpayable",
            "name": "approve",
            "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
            "outputs": [{"name": "", "type": "bool"}],
        },
    ]
)


def assert_rel_close(local: int, onchain: int, tol: float):
    assert local > 0 and onchain > 0
    assert abs(local - onchain) / onchain < tol, f"local={local} onchain={onchain}"


@pytest.fixture
def threepool(eth_w3) -> CurvePool:
    """An on-chain-quoting CurvePool bound to the Curve 3pool (DAI/USDC/USDT)."""
    return CurvePool(w3=eth_w3, pool_address=CURVE_3POOL, tokens=[DAI, USDC, USDT])


@pytest.fixture(scope="module")
def boa_fork():
    boa.fork(ETH_RPC_URL)


def read_stable_state(pool: Any, decimals: list[int], *, legacy: bool = False, ng: bool = False) -> dict:
    return {
        "balances": [pool.balances(k) for k in range(len(decimals))],
        "rates": list(pool.stored_rates()) if ng else [10 ** (36 - d) for d in decimals],
        "amp": pool.A() if legacy else pool.A_precise(),
        "fee": pool.fee(),
        "a_precision": 1 if legacy else m.A_PRECISION,
        "ng_d_form": ng,
        "offpeg_fee_multiplier": pool.offpeg_fee_multiplier() if ng else 0,
        "legacy_fee_order": legacy,
    }


def read_crypto_state(pool: Any, decimals: list[int]) -> dict:
    return {
        "balances": [pool.balances(k) for k in range(len(decimals))],
        "precisions": [10 ** (18 - d) for d in decimals],
        "price_scale": [m.PRECISION, pool.price_scale(0), pool.price_scale(1)],
        "amp": pool.A(),
        "gamma": pool.gamma(),
        "d": pool.D(),
        "mid_fee": pool.mid_fee(),
        "out_fee": pool.out_fee(),
        "fee_gamma": pool.fee_gamma(),
    }


def crypto_xp(state: dict, balances: list[int] | None = None) -> list[int]:
    bals = balances if balances is not None else state["balances"]
    return [b * p * s // m.PRECISION for b, p, s in zip(bals, state["precisions"], state["price_scale"])]


@pytest.mark.live
class TestCurveLive:
    """Live on-chain tests for CurvePool (3pool)."""

    @pytest.mark.parametrize(
        ("src", "dst", "dx", "low", "high", "scale", "label"),
        [
            (DAI, USDC, 1_000 * 10**18, 990 * 10**6, 1_010 * 10**6, 10**6, "DAI→USDC"),
            (USDC, DAI, 1_000 * 10**6, 990 * 10**18, 1_010 * 10**18, 10**18, "USDC→DAI"),
            (USDC, USDT, 1_000 * 10**6, 990 * 10**6, 1_010 * 10**6, 10**6, "USDC→USDT"),
        ],
    )
    async def test_get_dy_rough_parity(self, threepool, src, dst, dx, low, high, scale, label):
        """1 000-unit swaps on 3pool should stay within ±1% in either direction."""
        dy = await threepool.get_dy(src, dst, dx)
        assert low < dy < high, f"{label} out of range: {dy / scale:.4f}"

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

    async def test_legacy_3pool_local_matches_onchain(self, eth_w3):
        local_pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
            kind=CurvePoolKind.STABLE_LEGACY,
        )
        await local_pool.load_state()
        onchain = await CurvePool(
            w3=eth_w3,
            pool_address=CURVE_3POOL,
            tokens=[DAI, USDC, USDT],
            kind=CurvePoolKind.STABLE_LEGACY,
        ).get_dy(DAI, USDC, 50_000 * 10**18)
        local = local_pool.get_dy_local(DAI, USDC, 50_000 * 10**18)
        assert_rel_close(local, onchain, 1e-5)

    async def test_stable_ng_local_matches_onchain(self, eth_w3):
        local_pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_USDE_USDC_NG,
            tokens=[USDE, USDC],
            kind=CurvePoolKind.STABLE_NG,
        )
        await local_pool.load_state()
        onchain = await CurvePool(
            w3=eth_w3,
            pool_address=CURVE_USDE_USDC_NG,
            tokens=[USDE, USDC],
            kind=CurvePoolKind.STABLE_NG,
        ).get_dy(USDE, USDC, 100_000 * 10**18)
        local = local_pool.get_dy_local(USDE, USDC, 100_000 * 10**18)
        assert_rel_close(local, onchain, 1e-4)

    async def test_stable_factory_local_matches_onchain(self, eth_w3):
        local_pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_USDC_CRVUSD_FACTORY,
            tokens=[USDC, CRVUSD],
            kind=CurvePoolKind.STABLE_FACTORY,
        )
        await local_pool.load_state()
        onchain = await CurvePool(
            w3=eth_w3,
            pool_address=CURVE_USDC_CRVUSD_FACTORY,
            tokens=[USDC, CRVUSD],
            kind=CurvePoolKind.STABLE_FACTORY,
        ).get_dy(USDC, CRVUSD, 100_000 * 10**6)
        local = local_pool.get_dy_local(USDC, CRVUSD, 100_000 * 10**6)
        assert_rel_close(local, onchain, 1e-4)

    async def test_crypto_v2_tricrypto_local_matches_onchain(self, eth_w3):
        # Cryptoswap reuses the pool's stored D; tolerate minor block drift.
        local_pool = CurvePool(
            w3=eth_w3,
            pool_address=CURVE_TRICRYPTO_USDC,
            tokens=[USDC, WBTC, WETH],
            kind=CurvePoolKind.CRYPTO_V2,
        )
        await local_pool.load_state()
        onchain = await CurvePool(
            w3=eth_w3,
            pool_address=CURVE_TRICRYPTO_USDC,
            tokens=[USDC, WBTC, WETH],
            kind=CurvePoolKind.CRYPTO_V2,
        ).get_dy(USDC, WETH, 50_000 * 10**6)
        local = local_pool.get_dy_local(USDC, WETH, 50_000 * 10**6)
        assert_rel_close(local, onchain, 1e-3)

    async def test_stable_ng_get_dx_matches_onchain(self, eth_w3):
        # Exact-output: local closed-form get_dx vs the pool's on-chain get_dx.
        tokens = [USDE, USDC]
        pool = CurvePool(w3=eth_w3, pool_address=CURVE_USDE_USDC_NG, tokens=tokens, kind=CurvePoolKind.STABLE_NG)
        await pool.load_state()
        dy = 50_000 * 10**6  # want 50k USDC out
        onchain = await CURVE_POOL.fns.get_dx(0, 1, dy).call(eth_w3, to=CURVE_USDE_USDC_NG)
        local = pool.get_dx_local(USDE, USDC, dy)
        assert_rel_close(local, onchain, 1e-4)


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
        assert_rel_close(local, onchain, 1e-3)


@pytest.mark.live
class TestCurveMetaPool:
    """Meta-pool underlying pricing vs the pool's on-chain get_dy_underlying."""

    async def test_primary_to_base_coin(self, eth_w3):
        base = CurvePool(eth_w3, CURVE_3POOL, [DAI, USDC, USDT], kind=CurvePoolKind.STABLE_LEGACY)
        meta = CurveMetaPool(eth_w3, CURVE_MIM_3CRV, MIM, base, kind=CurvePoolKind.STABLE_FACTORY)
        await meta.load_state()
        dx = 50_000 * 10**18  # MIM -> USDC (underlying j=2)
        local = meta.get_dy_underlying(MIM, USDC, dx)
        onchain = await CURVE_POOL.fns.get_dy_underlying(0, 2, dx).call(eth_w3, to=CURVE_MIM_3CRV)
        assert_rel_close(local, onchain, 2e-3)

    async def test_base_coin_to_primary(self, eth_w3):
        base = CurvePool(eth_w3, CURVE_3POOL, [DAI, USDC, USDT], kind=CurvePoolKind.STABLE_LEGACY)
        meta = CurveMetaPool(eth_w3, CURVE_MIM_3CRV, MIM, base, kind=CurvePoolKind.STABLE_FACTORY)
        await meta.load_state()
        dx = 50_000 * 10**6  # USDT (underlying i=3) -> MIM
        local = meta.get_dy_underlying(USDT, MIM, dx)
        onchain = await CURVE_POOL.fns.get_dy_underlying(3, 0, dx).call(eth_w3, to=CURVE_MIM_3CRV)
        assert_rel_close(local, onchain, 2e-3)


@pytest.mark.live
@pytest.mark.usefixtures("boa_fork")
class TestCurveBoaBitExact:
    @pytest.mark.parametrize(
        ("pool_addr", "decimals", "pairs", "legacy", "ng"),
        [
            (CURVE_3POOL, THREEPOOL_DECIMALS, [(0, 1), (1, 0), (1, 2)], True, False),
            (CURVE_USDC_CRVUSD_FACTORY, [6, 18], [(0, 1), (1, 0)], False, False),
            (CURVE_USDE_USDC_NG, USDE_USDC_DECIMALS, [(0, 1), (1, 0)], False, True),
        ],
    )
    def test_stable_get_dy_bit_exact(self, pool_addr, decimals, pairs, legacy, ng):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(pool_addr)
        state = read_stable_state(pool, decimals, legacy=legacy, ng=ng)
        for i, j in pairs:
            for amt in (1, 1_000, 100_000):
                dx = amt * 10 ** decimals[i]
                local = m.stable_get_dy(i, j, dx, **state)
                onchain = pool.get_dy(i, j, dx)
                assert local == onchain, f"(i={i}, j={j}, dx={dx}): local={local} onchain={onchain}"

    def test_stable_ng_get_dx(self):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_USDE_USDC_NG)
        state = read_stable_state(pool, USDE_USDC_DECIMALS, ng=True)
        for dy in (1_000 * 10**6, 50_000 * 10**6):
            assert m.stable_get_dx(0, 1, dy, **state) == pool.get_dx(0, 1, dy)

    def test_legacy_get_dx_inversion(self):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_3POOL)
        state = read_stable_state(pool, THREEPOOL_DECIMALS, legacy=True)
        for amt in (1_000, 50_000):
            target = amt * 10**6
            dx = m.stable_get_dx(0, 1, target, **state)
            recovered = pool.get_dy(0, 1, dx)
            assert abs(recovered - target) <= max(2, target // 10**6), f"target={target} recovered={recovered} dx={dx}"

    def test_factory_get_dx_inversion(self):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_USDC_CRVUSD_FACTORY)
        state = read_stable_state(pool, [6, 18])
        for amt in (1_000, 50_000):
            target = amt * 10**18
            dx = m.stable_get_dx(0, 1, target, **state)
            recovered = pool.get_dy(0, 1, dx)
            assert abs(recovered - target) <= max(2, target // 10**6), f"target={target} recovered={recovered} dx={dx}"

    def test_ng_dynamic_fee(self):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_USDE_USDC_NG)
        state = read_stable_state(pool, USDE_USDC_DECIMALS, ng=True)
        xp = m._xp_mem(state["rates"], state["balances"])
        for i, j in ((0, 1), (1, 0)):
            local = m.stable_dynamic_fee(xp[i], xp[j], state["fee"], state["offpeg_fee_multiplier"])
            assert local == pool.dynamic_fee(i, j)

    def test_legacy_calc_token_amount(self):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_3POOL)
        state = read_stable_state(pool, THREEPOOL_DECIMALS, legacy=True)
        three_crv: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(THREE_CRV)
        supply: int = three_crv.totalSupply()
        for amounts, is_deposit in [
            ([1_000 * 10**18, 0, 0], True),
            ([0, 500 * 10**6, 250 * 10**6], True),
            ([100 * 10**18, 100 * 10**6, 0], False),
        ]:
            local = m.stable_calc_token_amount(
                amounts,
                state["balances"],
                state["rates"],
                state["amp"],
                supply,
                is_deposit=is_deposit,
                a_precision=1,
            )
            assert local == pool.calc_token_amount(amounts, is_deposit), f"{amounts} deposit={is_deposit}"

    def test_legacy_calc_withdraw_one_coin(self):
        pool: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_3POOL)
        state = read_stable_state(pool, THREEPOOL_DECIMALS, legacy=True)
        three_crv: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(THREE_CRV)
        supply: int = three_crv.totalSupply()
        lp = 10_000 * 10**18
        for i in range(3):
            local = m.stable_calc_withdraw_one_coin(
                lp,
                i,
                state["balances"],
                state["rates"],
                state["amp"],
                state["fee"],
                supply,
                a_precision=1,
            )
            assert local == pool.calc_withdraw_one_coin(lp, i), f"coin {i}"

    def test_tricrypto2_bit_exact(self):
        pool: Any = boa.loads_abi(CRYPTO_ABI, name="CurveCrypto").at(CURVE_TRICRYPTO2)
        state = read_crypto_state(pool, TRICRYPTO_DECIMALS)
        for i, j in ((0, 2), (2, 0), (1, 2)):
            for amt in (1, 1_000):
                dx = amt * 10 ** TRICRYPTO_DECIMALS[i]
                local = m.crypto_get_dy(i, j, dx, **state)
                onchain = pool.get_dy(i, j, dx)
                assert local == onchain, f"(i={i}, j={j}, dx={dx}): local={local} onchain={onchain}"

    def test_tricrypto_ng_close(self):
        pool: Any = boa.loads_abi(CRYPTO_ABI, name="CurveCrypto").at(CURVE_TRICRYPTO_USDC)
        state = read_crypto_state(pool, TRICRYPTO_DECIMALS)
        for i, j, dx in ((0, 2, 10_000 * 10**6), (2, 0, 10 * 10**18)):
            local = m.crypto_get_dy(i, j, dx, **state)
            onchain = pool.get_dy(i, j, dx)
            assert abs(local - onchain) <= max(2, onchain // 10**10), f"local={local} onchain={onchain}"

    def test_newton_D_and_y_match_deployed_math(self):
        pool: Any = boa.loads_abi(CRYPTO_ABI, name="CurveCrypto").at(CURVE_TRICRYPTO2)
        math3: Any = boa.loads_abi(MATH3_ABI, name="Math3").at(TRICRYPTO2_MATH)
        state = read_crypto_state(pool, TRICRYPTO_DECIMALS)
        xp = crypto_xp(state)
        x_desc = sorted(xp, reverse=True)
        assert m._geometric_mean(x_desc) == math3.geometric_mean(x_desc, False)
        assert m.newton_D(state["amp"], state["gamma"], xp) == math3.newton_D(state["amp"], state["gamma"], xp)
        d = state["d"]
        for j in range(3):
            local = m.newton_y(state["amp"], state["gamma"], xp, d, j)
            assert local == math3.newton_y(state["amp"], state["gamma"], xp, d, j), f"coin {j}"

    def test_crypto_fee_matches_fee_calc(self):
        pool: Any = boa.loads_abi(CRYPTO_ABI, name="CurveCrypto").at(CURVE_TRICRYPTO2)
        state = read_crypto_state(pool, TRICRYPTO_DECIMALS)
        skewed = list(state["balances"])
        skewed[0] += 5_000_000 * 10**6
        for xp in (crypto_xp(state), crypto_xp(state, skewed)):
            local = m.crypto_fee(xp, state["mid_fee"], state["out_fee"], state["fee_gamma"])
            assert local == pool.fee_calc(xp)

    def test_crypto_exact_out_search_minimal(self):
        pool: Any = boa.loads_abi(CRYPTO_ABI, name="CurveCrypto").at(CURVE_TRICRYPTO2)
        state = read_crypto_state(pool, TRICRYPTO_DECIMALS)
        target = 1 * 10**18
        lo, hi = 0, 1
        while m.crypto_get_dy(0, 2, hi, **state) < target:
            hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if m.crypto_get_dy(0, 2, mid, **state) >= target:
                hi = mid
            else:
                lo = mid
        assert pool.get_dy(0, 2, hi) >= target > pool.get_dy(0, 2, hi - 1)

    def test_build_exchange_tx_executes(self):
        dx = 1_000 * 10**18
        pool_view: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_3POOL)
        quoted: int = pool_view.get_dy(0, 1, dx)
        pool = CurvePool(w3=cast(Any, None), pool_address=CURVE_3POOL, tokens=[DAI, USDC, USDT])
        tx = pool.build_exchange_tx(TokenAmount(token=DAI, amount=dx), USDC, min_amount_out=quoted - 2)

        dai: Any = boa.loads_abi(ERC20_ABI, name="ERC20").at(DAI.address.to_0x_hex())
        usdc: Any = boa.loads_abi(ERC20_ABI, name="ERC20").at(USDC.address.to_0x_hex())
        trader = boa.env.generate_address()
        boa.deal(dai, trader, dx)
        with boa.env.prank(trader):
            dai.approve(CURVE_3POOL, dx)
            boa.env.raw_call(tx["to"], data=bytes.fromhex(tx["data"][2:]))
        received = usdc.balanceOf(trader)
        assert abs(received - quoted) <= 2, f"received={received} quoted={quoted}"

    def test_build_exchange_underlying_tx_executes(self):
        mim_token = Token(
            chain_id=ChainId.ETHEREUM,
            address=Address("0x99D8a9C45b2ecA8864373A26D1459e3Dff1e17F3"),
            symbol="MIM",
            decimals=18,
        )
        dx = 1_000 * 10**18
        meta_view: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_MIM_3CRV)
        quoted: int = meta_view.get_dy_underlying(1, 0, dx)

        base = CurvePool(w3=cast(Any, None), pool_address=CURVE_3POOL, tokens=[DAI, USDC, USDT])
        meta = CurveMetaPool(
            w3=cast(Any, None),
            pool_address=CURVE_MIM_3CRV,
            primary_token=mim_token,
            base_pool=base,
        )
        tx = meta.build_exchange_underlying_tx(
            TokenAmount(token=DAI, amount=dx),
            mim_token,
            min_amount_out=quoted * 999 // 1000,
        )

        dai: Any = boa.loads_abi(ERC20_ABI, name="ERC20").at(DAI.address.to_0x_hex())
        mim: Any = boa.loads_abi(ERC20_ABI, name="ERC20").at(mim_token.address.to_0x_hex())
        trader = boa.env.generate_address()
        boa.deal(dai, trader, dx)
        with boa.env.prank(trader):
            dai.approve(CURVE_MIM_3CRV, dx)
            boa.env.raw_call(tx["to"], data=bytes.fromhex(tx["data"][2:]))
        received = mim.balanceOf(trader)
        assert abs(received - quoted) <= quoted // 1000, f"received={received} quoted={quoted}"

    def test_mim_3crv_underlying_bit_exact(self):
        meta: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_MIM_3CRV)
        base: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_3POOL)
        base_state = read_stable_state(base, THREEPOOL_DECIMALS, legacy=True)
        three_crv: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(meta.coins(1))
        base_state["total_supply"] = three_crv.totalSupply()
        meta_state = {
            "balances": [meta.balances(0), meta.balances(1)],
            "rates": [10**18, base.get_virtual_price()],
            "amp": meta.A_precise(),
            "fee": meta.fee(),
            "a_precision": m.A_PRECISION,
            "ng_d_form": False,
        }
        for i, j, dx in ((0, 2, 50_000 * 10**18), (3, 0, 50_000 * 10**6), (1, 2, 10_000 * 10**18)):
            local = m.meta_get_dy_underlying(i, j, dx, meta_state, base_state)
            onchain = meta.get_dy_underlying(i, j, dx)
            assert local == onchain, f"(i={i}, j={j}): local={local} onchain={onchain}"

    def test_ng_meta_underlying_bit_exact(self):
        meta: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_USD1_NG_META)
        base: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(CURVE_2POOL_NG)
        base_state = read_stable_state(base, [6, 6], ng=True)
        ng_lp: Any = boa.loads_abi(STABLE_ABI, name="CurveStable").at(meta.coins(1))
        base_state["total_supply"] = ng_lp.totalSupply()
        meta_state = {
            "balances": [meta.balances(0), meta.balances(1)],
            "rates": list(meta.stored_rates()),
            "amp": meta.A() * m.A_PRECISION,
            "fee": meta.fee(),
            "a_precision": m.A_PRECISION,
            "ng_d_form": True,
            "offpeg_fee_multiplier": meta.offpeg_fee_multiplier(),
        }
        for i, j, dx in ((0, 1, 10_000 * 10**18), (2, 0, 10_000 * 10**6), (1, 2, 10_000 * 10**6)):
            local = m.meta_get_dy_underlying(i, j, dx, meta_state, base_state)
            onchain = meta.get_dy_underlying(i, j, dx)
            assert local == onchain, f"(i={i}, j={j}): local={local} onchain={onchain}"
