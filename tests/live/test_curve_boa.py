"""Bit-exact cross-check of :mod:`pydefi.amm.curve_math` against real Curve bytecode.
uv run --all-extras --with "titanoboa>=0.2.8" pytest tests/live/test_curve_boa.py -m live
"""

from __future__ import annotations

import json
import os

import pytest

boa = pytest.importorskip("boa")

from pydefi.amm import curve_math as m  # noqa: E402  (after importorskip by design)

ETH_RPC_URL = os.environ.get("ETH_RPC_URL") or "https://eth.drpc.org"

# Mainnet pools (one per CurvePoolKind, plus both tricrypto generations + meta).
CURVE_3POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"  # DAI/USDC/USDT, legacy
CURVE_USDC_CRVUSD = "0x4DEcE678ceceb27446b35C672dC7d61F30bAD69E"  # factory plain
CURVE_USDE_USDC_NG = "0x02950460E2b9529D0E00284A5fA2d7bDF3fA4d72"  # Stable-NG
CURVE_TRICRYPTO2 = "0xD51a44d3FaE010294C616388b506AcdA1bfAAE46"  # USDT/WBTC/WETH, newton_y
CURVE_TRICRYPTO_USDC = "0x7F86Bf177Dd4F3494b841a37e810A34dD56c829B"  # ng, analytic get_y
CURVE_MIM_3CRV = "0x5a6A4D54456819380173272A5E8E9B9904BdF41B"  # factory meta


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
        _view("coins", ["uint256"], ["address"]),
        _view("balances", ["uint256"], ["uint256"]),
        _view("A", [], ["uint256"]),
        _view("A_precise", [], ["uint256"]),
        _view("fee", [], ["uint256"]),
        _view("stored_rates", [], ["uint256[]"]),
        _view("offpeg_fee_multiplier", [], ["uint256"]),
        _view("get_virtual_price", [], ["uint256"]),
        _view("totalSupply", [], ["uint256"]),
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
    ]
)


@pytest.fixture(scope="module", autouse=True)
def fork():
    """Pin the whole module to one forked mainnet block."""
    boa.fork(ETH_RPC_URL)


def stable_at(addr: str):
    return boa.loads_abi(STABLE_ABI, name="CurveStable").at(addr)


def crypto_at(addr: str):
    return boa.loads_abi(CRYPTO_ABI, name="CurveCrypto").at(addr)


def read_stable_state(pool, decimals: list[int], *, legacy: bool = False, ng: bool = False) -> dict:
    """Snapshot a stableswap pool through boa (mirrors CurvePool._load_stable_state)."""
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


def read_crypto_state(pool, decimals: list[int]) -> dict:
    """Snapshot a 3-coin cryptoswap pool through boa (mirrors _load_crypto_state)."""
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


def assert_stable_bit_exact(pool, state: dict, decimals: list[int], pairs: list[tuple[int, int]]):
    for i, j in pairs:
        for amt in (1, 1_000, 100_000):
            dx = amt * 10 ** decimals[i]
            local = m.stable_get_dy(i, j, dx, **state)
            onchain = pool.get_dy(i, j, dx)
            print("mm-onchain", onchain, "local", local)
            assert local == onchain, f"(i={i}, j={j}, dx={dx}): local={local} onchain={onchain}"


@pytest.mark.live
class TestBoaStableswap:
    """Local stableswap math must equal the deployed bytecode exactly."""

    def test_legacy_3pool(self):
        pool = stable_at(CURVE_3POOL)
        state = read_stable_state(pool, [18, 6, 6], legacy=True)
        assert_stable_bit_exact(pool, state, [18, 6, 6], [(0, 1), (1, 0), (1, 2)])

    def test_factory_usdc_crvusd(self):
        pool = stable_at(CURVE_USDC_CRVUSD)
        state = read_stable_state(pool, [6, 18])
        assert_stable_bit_exact(pool, state, [6, 18], [(0, 1), (1, 0)])

    def test_stable_ng_usde_usdc(self):
        pool = stable_at(CURVE_USDE_USDC_NG)
        state = read_stable_state(pool, [18, 6], ng=True)
        assert_stable_bit_exact(pool, state, [18, 6], [(0, 1), (1, 0)])

    def test_stable_ng_get_dx(self):
        pool = stable_at(CURVE_USDE_USDC_NG)
        state = read_stable_state(pool, [18, 6], ng=True)
        for dy in (1_000 * 10**6, 50_000 * 10**6):
            assert m.stable_get_dx(0, 1, dy, **state) == pool.get_dx(0, 1, dy)


@pytest.mark.live
class TestBoaCryptoswap:
    def test_tricrypto2_bit_exact(self):
        # The classic pool runs the same newton_y we ported: exact equality.
        pool = crypto_at(CURVE_TRICRYPTO2)
        state = read_crypto_state(pool, [6, 8, 18])
        for i, j in ((0, 2), (2, 0), (1, 2)):
            for amt in (1, 1_000):
                dx = amt * 10 ** [6, 8, 18][i]
                local = m.crypto_get_dy(i, j, dx, **state)
                onchain = pool.get_dy(i, j, dx)
                assert local == onchain, f"(i={i}, j={j}, dx={dx}): local={local} onchain={onchain}"

    def test_tricrypto_ng_close(self):
        # tricrypto-ng solves y analytically (MATH.get_y), not via newton_y:
        # agreement is to solver precision, not bit-for-bit.
        pool = crypto_at(CURVE_TRICRYPTO_USDC)
        state = read_crypto_state(pool, [6, 8, 18])
        for i, j, dx in ((0, 2, 10_000 * 10**6), (2, 0, 10 * 10**18)):
            local = m.crypto_get_dy(i, j, dx, **state)
            onchain = pool.get_dy(i, j, dx)
            assert abs(local - onchain) <= max(2, onchain // 10**12), f"local={local} onchain={onchain}"


@pytest.mark.live
class TestBoaMetaPool:
    def test_mim_3crv_underlying_bit_exact(self):
        meta = stable_at(CURVE_MIM_3CRV)
        base = stable_at(CURVE_3POOL)
        base_state = read_stable_state(base, [18, 6, 6], legacy=True)
        base_state["total_supply"] = stable_at(meta.coins(1)).totalSupply()
        meta_state = {
            "balances": [meta.balances(0), meta.balances(1)],
            "rates": [10**18, base.get_virtual_price()],  # MIM is 18-dec
            "amp": meta.A_precise(),
            "fee": meta.fee(),
            "a_precision": m.A_PRECISION,
            "ng_d_form": False,
        }
        for i, j, dx in ((0, 2, 50_000 * 10**18), (3, 0, 50_000 * 10**6), (1, 2, 10_000 * 10**18)):
            local = m.meta_get_dy_underlying(i, j, dx, meta_state, base_state)
            onchain = meta.get_dy_underlying(i, j, dx)
            assert local == onchain, f"(i={i}, j={j}): local={local} onchain={onchain}"
