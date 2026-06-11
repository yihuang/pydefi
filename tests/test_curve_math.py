"""Unit tests for local Curve pricing math (no network required).

Covers the stableswap invariant (legacy plain / factory / Stable-NG) and the
cryptoswap (Curve V2) invariant in :mod:`pydefi.amm.curve_math`, plus the
:class:`CurveStableEdge` / :class:`CurveCryptoEdge` pathfinder edges and the
local-pricing path on :class:`CurvePool`.

Each pool's parameters live in a single config dict whose keys match both the
``*_get_dy`` kwargs and the edge / pool-state fields, so the same config drives
the math call, the edge, and the stubbed pool.  Golden integer values lock the
math against regressions; the bit-close cross-check against the real on-chain
``get_dy`` lives in ``tests/live/test_curve_live.py``.
"""

from __future__ import annotations

import pytest

from pydefi.abi.amm import CURVE_POOL, CURVE_V2_POOL
from pydefi.amm import curve_math as m
from pydefi.amm.curve import CurveMetaPool, CurvePool, CurvePoolKind
from pydefi.exceptions import InsufficientLiquidityError, PoolDataError
from pydefi.pathfinder.graph import CurveCryptoEdge, CurveStableEdge, PoolGraph
from pydefi.types import Address, Token, TokenAmount

ADDR = Address("0x" + "11" * 20)


def _tok(symbol: str, decimals: int, byte: str) -> Token:
    return Token(chain_id=1, address=Address("0x" + byte * 20), symbol=symbol, decimals=decimals)


DAI = _tok("DAI", 18, "d0")
USDC = _tok("USDC", 6, "c0")
USDT = _tok("USDT", 6, "d7")
WETH = _tok("WETH", 18, "ee")

# Balanced 3pool-style legacy stableswap ($100M each); no A_PRECISION.
THREEPOOL = {
    "balances": [100_000_000 * 10**18, 100_000_000 * 10**6, 100_000_000 * 10**6],
    "rates": [10**18, 10**30, 10**30],  # 10**(36 - decimals)
    "amp": 2000,
    "fee": 1_000_000,  # 1 bp in 1e10 units
    "a_precision": 1,
    "ng_d_form": False,
    "offpeg_fee_multiplier": 0,
    "legacy_fee_order": True,
}
# 2-coin Stable-NG pool ($50M each): A_PRECISION, NG D form, off-peg dynamic fee.
NG = {
    "balances": [50_000_000 * 10**18, 50_000_000 * 10**6],
    "rates": [10**18, 10**30],
    "amp": 200_000,  # A_precise() = 2000 * 100
    "fee": 1_000_000,
    "a_precision": 100,
    "ng_d_form": True,
    "offpeg_fee_multiplier": 20_000_000_000,
    "legacy_fee_order": False,
}
# Factory plain pool: same coins as NG but classic D form and a flat fee.
FACTORY = {**NG, "ng_d_form": False, "offpeg_fee_multiplier": 0}
# Tricrypto-style Curve V2 pool: USDT(6) / WBTC(8) / WETH(18).
CRYPTO = {
    "balances": [10_000_000 * 10**6, 100 * 10**8, 3000 * 10**18],
    "precisions": [10**12, 10**10, 10**0],
    "price_scale": [10**18, 100_000 * 10**18, 3333 * 10**18],
    "amp": 1_707_629,
    "gamma": 11_809_167_828_997,
    "mid_fee": 3_000_000,
    "out_fee": 30_000_000,
    "fee_gamma": 500_000_000_000_000,
}


# The config dicts' keys mirror the *_get_dy / *_get_dx parameters, so they
# unpack directly into the math calls.
def stable_dy(cfg: dict, i: int, j: int, dx: int, **overrides) -> int:
    """Price a stableswap swap from a pool config dict."""
    return m.stable_get_dy(i, j, dx, **{**cfg, **overrides})


def stable_dx(cfg: dict, i: int, j: int, dy: int) -> int:
    """Exact-output stableswap quote from a pool config dict."""
    return m.stable_get_dx(i, j, dy, **cfg)


def crypto_dy(cfg: dict, i: int, j: int, dx: int, d: int | None = None) -> int:
    """Price a cryptoswap (Curve V2) swap from a pool config dict."""
    return m.crypto_get_dy(i, j, dx, **cfg, d=d)


def strictly_increasing(values: list[int]) -> bool:
    """True if *values* is sorted ascending with no duplicates."""
    return values == sorted(values) and len(set(values)) == len(values)


def decode_exchange(fn, tx: dict) -> tuple:
    """Decode an exchange-call tx dict's calldata via its ABI function *fn*."""
    return fn.decode_input(bytes.fromhex(tx["data"][2:]))


# Precomputed 3pool normalised balances + invariant (shared by several tests).
THREE_XP = m._xp_mem(THREEPOOL["rates"], THREEPOOL["balances"])
THREE_D = m.stable_get_D(THREE_XP, THREEPOOL["amp"], a_precision=1)


def _stub_pool(kind: CurvePoolKind) -> CurvePool:
    """A CurvePool with an injected legacy-3pool state snapshot (no network)."""
    pool = CurvePool(w3=None, pool_address=ADDR, tokens=[DAI, USDC, USDT], kind=kind)
    pool._state = dict(THREEPOOL)
    return pool


def _stub_crypto_pool() -> CurvePool:
    """A CurvePool with an injected Tricrypto state snapshot (no network)."""
    pool = CurvePool(w3=None, pool_address=ADDR, tokens=[USDT, USDC, WETH], kind=CurvePoolKind.CRYPTO_V2)
    pool._state = {**CRYPTO, "d": None}
    return pool


# ---------------------------------------------------------------------------
# Stableswap invariant
# ---------------------------------------------------------------------------


class TestStableD:
    def test_balanced_pool_D_equals_total_normalised(self):
        assert THREE_D == sum(THREE_XP) == 300_000_000 * 10**18

    def test_empty_pool_returns_zero(self):
        assert m.stable_get_D([0, 0, 0], THREEPOOL["amp"], a_precision=1) == 0

    def test_ng_and_classic_D_forms_agree_when_balanced(self):
        xp = m._xp_mem(NG["rates"], NG["balances"])
        d_classic = m.stable_get_D(xp, NG["amp"], a_precision=100, ng_d_form=False)
        d_ng = m.stable_get_D(xp, NG["amp"], a_precision=100, ng_d_form=True)
        assert d_classic == d_ng == sum(xp)


class TestStableGetY:
    def test_get_y_preserves_invariant_when_balanced(self):
        # coin 0 unchanged → coin 1's solved balance stays ≈ xp[1].
        y = m.stable_get_y(0, 1, THREE_XP[0], THREE_XP, THREEPOOL["amp"], THREE_D, a_precision=1)
        assert abs(y - THREE_XP[1]) <= 2

    def test_get_y_rejects_same_coin(self):
        with pytest.raises(ValueError):
            m.stable_get_y(0, 0, THREE_XP[0], THREE_XP, THREEPOOL["amp"], THREE_D, a_precision=1)


class TestStableGetDy:
    def test_golden_values(self):
        assert stable_dy(THREEPOOL, 0, 1, 1000 * 10**18) == 999_899_996  # ≈ 999.9 USDC / 1000 DAI
        assert stable_dy(NG, 0, 1, 1000 * 10**18) == 999_899_990
        assert stable_dy(FACTORY, 0, 1, 1000 * 10**18) == 999_899_990

    def test_balanced_swap_near_one_to_one(self):
        assert 998 * 10**6 < stable_dy(THREEPOOL, 0, 1, 1000 * 10**18) < 1000 * 10**6

    def test_output_monotonic_in_input(self):
        outs = [stable_dy(THREEPOOL, 0, 1, amt * 10**18) for amt in (100, 1_000, 10_000, 100_000)]
        assert strictly_increasing(outs)

    def test_fee_reduces_output(self):
        no_fee = stable_dy(THREEPOOL, 0, 1, 1000 * 10**18, fee=0)
        with_fee = stable_dy(THREEPOOL, 0, 1, 1000 * 10**18, fee=4_000_000)
        assert with_fee < no_fee

    def test_dynamic_fee_grows_with_imbalance(self):
        flat = m.stable_dynamic_fee(100, 100, NG["fee"], NG["offpeg_fee_multiplier"])
        skewed = m.stable_dynamic_fee(10, 190, NG["fee"], NG["offpeg_fee_multiplier"])
        assert skewed > flat == NG["fee"]

    def test_dynamic_fee_flat_when_multiplier_low(self):
        assert m.stable_dynamic_fee(10, 190, NG["fee"], m.FEE_DENOMINATOR) == NG["fee"]


class TestStableGetDx:
    @pytest.mark.parametrize("cfg", [THREEPOOL, NG, FACTORY])
    def test_roundtrip_dy_then_dx(self, cfg):
        # dx -> dy -> dx should recover the input to within rounding (a few wei).
        dx = 1000 * 10**18
        recovered = stable_dx(cfg, 0, 1, stable_dy(cfg, 0, 1, dx))
        assert abs(recovered - dx) < 10**12  # < 1e-6 of a 1e18 token

    def test_dx_grows_with_target(self):
        assert strictly_increasing([stable_dx(THREEPOOL, 0, 1, amt * 10**6) for amt in (100, 1_000, 10_000)])


# ---------------------------------------------------------------------------
# Cryptoswap invariant (Curve V2)
# ---------------------------------------------------------------------------


class TestCryptoMath:
    def test_golden_value(self):
        # 10k USDT → ETH at ~3333 USDT/ETH ≈ 3 ETH (minus fee).
        dy = crypto_dy(CRYPTO, 0, 2, 10_000 * 10**6)
        assert dy == 2_999_314_588_290_567_597
        assert 2.9 * 10**18 < dy < 3.0 * 10**18

    def test_output_monotonic_in_input(self):
        assert strictly_increasing([crypto_dy(CRYPTO, 0, 2, amt * 10**6) for amt in (1_000, 10_000, 50_000)])

    def test_recomputed_D_matches_passed_D(self):
        # newton_D over current balances ≈ the stored-D path (no ramp).
        xp = [
            CRYPTO["balances"][k] * CRYPTO["precisions"][k] * CRYPTO["price_scale"][k] // m.PRECISION for k in range(3)
        ]
        d = m.newton_D(CRYPTO["amp"], CRYPTO["gamma"], xp)
        assert crypto_dy(CRYPTO, 0, 2, 10_000 * 10**6, d=d) == crypto_dy(CRYPTO, 0, 2, 10_000 * 10**6)

    def test_crypto_fee_interpolates(self):
        args = (CRYPTO["mid_fee"], CRYPTO["out_fee"], CRYPTO["fee_gamma"])
        fee_balanced = m.crypto_fee([10**24, 10**24, 10**24], *args)
        fee_skewed = m.crypto_fee([10**24, 10**20, 10**20], *args)
        assert CRYPTO["mid_fee"] <= fee_balanced < fee_skewed <= CRYPTO["out_fee"]


# ---------------------------------------------------------------------------
# Stableswap meta-pool (underlying swaps)
# ---------------------------------------------------------------------------

# Base = balanced 3pool; meta = a primary 18-dec coin paired with the base LP.
META_BASE = {**THREEPOOL, "total_supply": 300_000_000 * 10**18}  # balanced D, vprice≈1
META = {
    "balances": [50_000_000 * 10**18, 50_000_000 * 10**18],  # [primary, base LP]
    "rates": [10**18, 10**18],  # primary 18-dec; base LP virtual price ≈ 1e18
    "amp": 200_000,
    "fee": 1_000_000,
    "a_precision": 100,
    "ng_d_form": False,
}


class TestMetaPool:
    def test_both_legs_in_base_pool_delegate(self):
        # Underlying (1, 2) = base coins (0, 1): delegates to the base pool's get_dy.
        dx = 1000 * 10**18
        got = m.meta_get_dy_underlying(1, 2, dx, META, META_BASE)
        assert got == stable_dy(THREEPOOL, 0, 1, dx)

    def test_primary_to_base_coin_near_peg(self):
        # 1000 primary -> base coin 0 (DAI, 18-dec) ≈ 1000, minus fees.
        dy = m.meta_get_dy_underlying(0, 1, 1000 * 10**18, META, META_BASE)
        assert 990 * 10**18 < dy < 1000 * 10**18

    def test_base_coin_to_primary_near_peg(self):
        # 1000 base coin 1 (USDC, 6-dec) -> primary (18-dec) ≈ 1000.
        dy = m.meta_get_dy_underlying(2, 0, 1000 * 10**6, META, META_BASE)
        assert 990 * 10**18 < dy < 1000 * 10**18

    def test_primary_to_base_monotonic(self):
        outs = [m.meta_get_dy_underlying(0, 2, amt * 10**18, META, META_BASE) for amt in (100, 1_000, 10_000)]
        assert strictly_increasing(outs)


# ---------------------------------------------------------------------------
# Pathfinder edges
# ---------------------------------------------------------------------------


class TestCurveEdges:
    def test_stable_edge_amount_out_matches_math(self):
        edge = CurveStableEdge(token_in=DAI, token_out=USDC, pool_address=ADDR, protocol="Curve", i=0, j=1, **THREEPOOL)
        assert edge.amount_out(1000 * 10**18) == stable_dy(THREEPOOL, 0, 1, 1000 * 10**18)
        assert edge.amount_out(0) == 0
        assert edge.spot_price > 0

    def test_crypto_edge_amount_out_matches_math(self):
        edge = CurveCryptoEdge(
            token_in=USDT, token_out=WETH, pool_address=ADDR, protocol="CurveV2", i=0, j=2, d=None, **CRYPTO
        )
        assert edge.amount_out(10_000 * 10**6) == crypto_dy(CRYPTO, 0, 2, 10_000 * 10**6)
        assert edge.spot_price > 0

    def test_edges_estimate_finite_price_impact(self):
        # Both edge kinds share the probe-based impact estimate; a large trade
        # must register real (non-NaN) impact even though reserves are unset.
        stable = CurveStableEdge(
            token_in=DAI, token_out=USDC, pool_address=ADDR, protocol="Curve", i=0, j=1, **THREEPOOL
        )
        crypto = CurveCryptoEdge(
            token_in=USDT, token_out=WETH, pool_address=ADDR, protocol="CurveV2", i=0, j=2, d=None, **CRYPTO
        )
        for edge, big in ((stable, 10_000_000 * 10**18), (crypto, 1_000_000 * 10**6)):
            impact = edge.estimate_price_impact(big)
            assert not impact.is_nan() and 0 < impact < 1

    def test_zero_for_one_does_not_raise(self):
        # Inherited PoolEdge.zero_for_one needs extra['is_token0_in'] and would
        # raise; Curve edges fall back to coin-index ordering.
        edge = CurveStableEdge(token_in=DAI, token_out=USDC, pool_address=ADDR, protocol="Curve", i=0, j=1, **THREEPOOL)
        assert edge.zero_for_one(USDC.address) is True
        back = CurveStableEdge(token_in=USDC, token_out=DAI, pool_address=ADDR, protocol="Curve", i=1, j=0, **THREEPOOL)
        assert back.zero_for_one(DAI.address) is False

    def test_edge_keys_unique_for_multi_coin_pool(self):
        # A 3-coin pool yields several edges sharing (pool, token_in); the
        # router's split index must key on token_out too, or edges overwrite
        # each other and find_best_split resolves steps to the wrong edge.
        from pydefi.pathfinder.router import _edge_key

        graph = PoolGraph()
        graph.add_curve_pool(_stub_pool(CurvePoolKind.STABLE_LEGACY))
        keys = {_edge_key(e) for e in graph}
        assert len(keys) == len(graph) == 6

    def test_add_curve_pool_sets_real_fee_bps(self):
        # Router copies edge.fee_bps into SwapStep.fee — it must reflect the
        # pool's actual fee, not the PoolEdge default of 30 bps.
        stable_graph, crypto_graph = PoolGraph(), PoolGraph()
        stable_graph.add_curve_pool(_stub_pool(CurvePoolKind.STABLE_LEGACY))
        crypto_graph.add_curve_pool(_stub_crypto_pool())
        assert {e.fee_bps for e in stable_graph.edges_from(DAI)} == {THREEPOOL["fee"] * 10_000 // m.FEE_DENOMINATOR}
        assert {e.fee_bps for e in crypto_graph.edges_from(USDT)} == {CRYPTO["mid_fee"] * 10_000 // m.FEE_DENOMINATOR}

    def test_graph_routes_through_curve_pool(self):
        graph = PoolGraph()
        graph.add_curve_pool(_stub_pool(CurvePoolKind.STABLE_LEGACY))
        assert len(graph) == 6  # six directed edges for a 3-coin pool
        route = graph.find_best_route_gas_aware(
            DAI, USDC, amount_in=1000 * 10**18, weight_fn=lambda e, a: e.log_weight(a), max_hops=2
        )
        assert len(route) == 1
        assert route[0].token_out == USDC

    def test_add_curve_pool_requires_loaded_state(self):
        pool = CurvePool(w3=None, pool_address=ADDR, tokens=[DAI, USDC, USDT])
        with pytest.raises(ValueError):
            PoolGraph().add_curve_pool(pool)


# ---------------------------------------------------------------------------
# CurvePool local-pricing dispatch
# ---------------------------------------------------------------------------


class TestCurvePoolLocal:
    def test_get_dy_local_matches_math(self):
        pool = _stub_pool(CurvePoolKind.STABLE_LEGACY)
        assert pool.get_dy_local(DAI, USDC, 1000 * 10**18) == stable_dy(THREEPOOL, 0, 1, 1000 * 10**18)

    def test_get_dy_local_without_state_raises(self):
        pool = CurvePool(w3=None, pool_address=ADDR, tokens=[DAI, USDC, USDT])
        with pytest.raises(PoolDataError):
            pool.get_dy_local(DAI, USDC, 10**18)

    def test_get_dx_local_stable_matches_math(self):
        pool = _stub_pool(CurvePoolKind.STABLE_LEGACY)
        assert pool.get_dx_local(DAI, USDC, 1000 * 10**6) == m.stable_get_dx(0, 1, 1000 * 10**6, **THREEPOOL)

    def test_get_dx_local_crypto_roundtrips(self):
        pool = _stub_crypto_pool()
        dx = 10_000 * 10**6  # USDT -> WETH
        dy = pool.get_dy_local(USDT, WETH, dx)
        recovered = pool.get_dx_local(USDT, WETH, dy)
        # Binary search lands on the smallest dx whose dy >= target: within 1 unit.
        assert abs(recovered - dx) <= 1

    def test_get_dx_local_without_state_raises(self):
        pool = CurvePool(w3=None, pool_address=ADDR, tokens=[DAI, USDC, USDT])
        with pytest.raises(PoolDataError):
            pool.get_dx_local(DAI, USDC, 10**6)

    def test_get_dx_local_crypto_over_capacity_raises(self):
        # The pool holds ~3000 WETH; no input can ever produce 1M WETH out.
        pool = _stub_crypto_pool()
        with pytest.raises(InsufficientLiquidityError):
            pool.get_dx_local(USDT, WETH, 1_000_000 * 10**18)

    async def test_get_dy_prefers_local_state(self):
        # With state loaded, get_dy must not touch w3 (which is None here).
        pool = _stub_pool(CurvePoolKind.STABLE_LEGACY)
        assert await pool.get_dy(DAI, USDC, 1000 * 10**18) == pool.get_dy_local(DAI, USDC, 1000 * 10**18)

    async def test_build_swap_route_reports_fee_in_bps(self):
        pool = _stub_pool(CurvePoolKind.STABLE_LEGACY)
        route = await pool.build_swap_route(TokenAmount(token=DAI, amount=1000 * 10**18), USDC)
        # SwapStep.fee is basis points: THREEPOOL's 1e6/1e10 fee is 1 bp.
        assert route.steps[0].fee == THREEPOOL["fee"] * 10_000 // m.FEE_DENOMINATOR == 1

    def test_protocol_name_by_kind(self):
        assert CurvePool(w3=None, pool_address=ADDR, tokens=[DAI, USDC]).protocol_name == "Curve"
        crypto = CurvePool(w3=None, pool_address=ADDR, tokens=[USDT, WETH], kind=CurvePoolKind.CRYPTO_V2)
        assert crypto.protocol_name == "CurveV2"


class TestCurveExecution:
    def test_build_exchange_tx_encodes_indices_and_min(self):
        pool = _stub_pool(CurvePoolKind.STABLE_LEGACY)
        dx = 1000 * 10**18
        tx = pool.build_exchange_tx(TokenAmount(token=DAI, amount=dx), USDC, slippage_bps=50)
        assert tx == {"to": ADDR, "data": tx["data"], "value": "0"}
        expected_min = pool.get_dy_local(DAI, USDC, dx) * (10_000 - 50) // 10_000
        assert decode_exchange(CURVE_POOL.fns.exchange, tx) == (0, 1, dx, expected_min)

    def test_build_exchange_tx_explicit_min(self):
        pool = _stub_pool(CurvePoolKind.STABLE_LEGACY)
        tx = pool.build_exchange_tx(TokenAmount(token=USDC, amount=10**6), USDT, min_amount_out=42)
        i, j, _sent, min_out = decode_exchange(CURVE_POOL.fns.exchange, tx)
        assert (i, j, min_out) == (1, 2, 42)

    def test_build_exchange_tx_crypto_uses_v2_abi(self):
        pool = _stub_crypto_pool()
        tx = pool.build_exchange_tx(TokenAmount(token=USDT, amount=10**6), WETH, min_amount_out=1)
        # Decodes against the uint256-index Curve V2 exchange ABI.
        i, j, _sent, min_out = decode_exchange(CURVE_V2_POOL.fns.exchange, tx)
        assert (i, j, min_out) == (0, 2, 1)

    def test_meta_build_exchange_underlying_tx(self):
        base = CurvePool(w3=None, pool_address=ADDR, tokens=[DAI, USDC, USDT])
        meta = CurveMetaPool(w3=None, pool_address=ADDR, primary_token=WETH, base_pool=base)
        tx = meta.build_exchange_underlying_tx(TokenAmount(token=WETH, amount=10**18), USDC, min_amount_out=7)
        # underlying coin indices: WETH=0 (primary), USDC=2 (base coin 1)
        i, j, _sent, min_out = decode_exchange(CURVE_POOL.fns.exchange_underlying, tx)
        assert (i, j, min_out) == (0, 2, 7)
