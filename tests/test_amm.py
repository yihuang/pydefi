"""Tests for pydefi.amm (no live node required)."""

from decimal import Decimal

import pytest

from pydefi.amm.uniswap_v2 import UniswapV2
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.types import ChainId, Token, TokenAmount


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WETH = Token(chain_id=ChainId.ETHEREUM, address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", symbol="WETH", decimals=18)
USDC = Token(chain_id=ChainId.ETHEREUM, address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", symbol="USDC", decimals=6)
DAI = Token(chain_id=ChainId.ETHEREUM, address="0x6B175474E89094C44Da98b954EedeAC495271d0F", symbol="DAI", decimals=18)

ROUTER_V2 = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
ROUTER_V3 = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
QUOTER_V3 = "0x61fFE014bA17989E743c5F6cB21bF9697530B21e"


# ---------------------------------------------------------------------------
# UniswapV2 pure math tests (no network calls)
# ---------------------------------------------------------------------------

class TestUniswapV2Math:
    def test_get_amount_out_basic(self):
        # 1 ETH in, 1000 ETH reserve, 2_000_000 USDC reserve → ~1997 USDC
        out = UniswapV2.get_amount_out(
            amount_in=10 ** 18,
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
        )
        # rough check: 1 ETH at $2000 minus 0.3% fee
        assert out > 1_990 * 10 ** 6
        assert out < 2_000 * 10 ** 6

    def test_get_amount_out_zero_reserve_raises(self):
        with pytest.raises(InsufficientLiquidityError):
            UniswapV2.get_amount_out(10 ** 18, 0, 10 ** 18)

    def test_get_amount_in_basic(self):
        # To buy 1000 USDC from pool with 1000 ETH / 2_000_000 USDC
        amount_in = UniswapV2.get_amount_in(
            amount_out=1_000 * 10 ** 6,
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
        )
        # Should require slightly more than 0.5 ETH (fee overhead)
        assert amount_in > 0.5 * 10 ** 18
        assert amount_in < 0.505 * 10 ** 18

    def test_get_amount_in_insufficient_reserve_raises(self):
        with pytest.raises(InsufficientLiquidityError):
            UniswapV2.get_amount_in(
                amount_out=2_000_000 * 10 ** 6,  # more than available
                reserve_in=1_000 * 10 ** 18,
                reserve_out=1_999_999 * 10 ** 6,
            )

    def test_get_amount_out_custom_fee(self):
        # Lower fee should give higher output
        out_standard = UniswapV2.get_amount_out(10 ** 18, 10 ** 21, 10 ** 21, fee_bps=30)
        out_low_fee = UniswapV2.get_amount_out(10 ** 18, 10 ** 21, 10 ** 21, fee_bps=5)
        assert out_low_fee > out_standard

    def test_spot_price(self):
        # 1000 WETH, 2_000_000 USDC → spot = 2000 USDC/WETH
        price = UniswapV2.spot_price(
            reserve_in=1_000 * 10 ** 18,
            reserve_out=2_000_000 * 10 ** 6,
            decimals_in=18,
            decimals_out=6,
        )
        assert price == Decimal("2000")

    def test_spot_price_zero_reserve(self):
        price = UniswapV2.spot_price(0, 10 ** 18)
        assert price == Decimal(0)

    def test_roundtrip_amount(self):
        """get_amount_in(get_amount_out(x)) ≈ x — within 1 wei due to integer division."""
        reserve_in = 1_000 * 10 ** 18
        reserve_out = 2_000_000 * 10 ** 6
        amount_in = 10 ** 18

        amount_out = UniswapV2.get_amount_out(amount_in, reserve_in, reserve_out)
        amount_in_back = UniswapV2.get_amount_in(amount_out, reserve_in, reserve_out)
        # Integer division may produce a value slightly below amount_in (floor rounding).
        # The difference must be within a negligible fraction of the input (< 0.01%).
        assert abs(amount_in_back - amount_in) < amount_in // 10_000

    def test_apply_slippage(self):
        from pydefi.amm.base import BaseAMM
        result = BaseAMM._apply_slippage(1_000_000, 50)
        assert result == 995_000

    def test_apply_slippage_zero(self):
        from pydefi.amm.base import BaseAMM
        result = BaseAMM._apply_slippage(1_000_000, 0)
        assert result == 1_000_000


# ---------------------------------------------------------------------------
# UniswapV3 math tests (no network calls)
# ---------------------------------------------------------------------------

class TestUniswapV3Math:
    def test_sqrt_price_to_price_equal_decimals(self):
        # At 1:1 ratio with equal decimals: sqrtPrice = 2^96
        sqrt_price_x96 = 2 ** 96
        price = UniswapV3.sqrt_price_to_price(sqrt_price_x96, 18, 18)
        assert abs(price - Decimal(1)) < Decimal("0.001")

    def test_sqrt_price_to_price_usdc_eth(self):
        # sqrtPriceX96 for ETH/USDC at ~$2000
        # price = (sqrtPriceX96 / 2^96)^2 * 10^(18-6)
        # For $2000/ETH: sqrtPriceX96 ≈ 1.77 * 10^24 (rough)
        sqrt_price_x96 = 1_771_595_571_142_957_116_569_145_374  # ~$2000
        price = UniswapV3.sqrt_price_to_price(sqrt_price_x96, 6, 18)
        # price is token0 (USDC) per token1 (ETH), approximately 1/2000
        assert price > Decimal(0)

    def test_encode_path_two_tokens(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        # 20 bytes (address) + 3 bytes (fee) + 20 bytes (address) = 43 bytes
        assert len(path) == 43

    def test_encode_path_three_tokens(self):
        path = UniswapV3._encode_path([WETH, DAI, USDC], [3000, 100])
        # 20 + 3 + 20 + 3 + 20 = 66 bytes
        assert len(path) == 66

    def test_encode_path_fee_mismatch_raises(self):
        with pytest.raises(ValueError):
            UniswapV3._encode_path([WETH, USDC], [3000, 500])  # 2 tokens, 2 fees — invalid

    def test_encode_path_contains_token_addresses(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        weth_bytes = bytes.fromhex(WETH.address[2:].lower())
        usdc_bytes = bytes.fromhex(USDC.address[2:].lower())
        assert weth_bytes in path
        assert usdc_bytes in path


# ---------------------------------------------------------------------------
# UniswapV2 instance (no live calls)
# ---------------------------------------------------------------------------

class TestUniswapV2Instance:
    def test_protocol_name_default(self):
        uniswap = UniswapV2(w3=None, router_address=ROUTER_V2)
        assert uniswap.protocol_name == "UniswapV2"

    def test_protocol_name_custom(self):
        sushi = UniswapV2(w3=None, router_address=ROUTER_V2, protocol_name="SushiSwap")
        assert sushi.protocol_name == "SushiSwap"

    def test_router_address_stored(self):
        uniswap = UniswapV2(w3=None, router_address=ROUTER_V2)
        assert uniswap.router_address == ROUTER_V2

    def test_get_pair_contract(self):
        uniswap = UniswapV2(w3=None, router_address=ROUTER_V2)
        pair = uniswap.get_pair_contract("0x" + "AB" * 20)
        assert pair is not None


# ---------------------------------------------------------------------------
# UniswapV3 instance (no live calls)
# ---------------------------------------------------------------------------

class TestUniswapV3Instance:
    def test_protocol_name(self):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3)
        assert v3.protocol_name == "UniswapV3"

    def test_default_fee(self):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3)
        assert v3.default_fee == 3000

    def test_custom_default_fee(self):
        v3 = UniswapV3(w3=None, router_address=ROUTER_V3, quoter_address=QUOTER_V3, default_fee=500)
        assert v3.default_fee == 500
