"""Tests for pydefi.types"""

from decimal import Decimal

from pydefi.types import (
    BridgeQuote,
    ChainId,
    SwapRoute,
    SwapStep,
    Token,
    TokenAmount,
)

# ---------------------------------------------------------------------------
# Token tests
# ---------------------------------------------------------------------------


class TestToken:
    def setup_method(self):
        self.eth = Token(
            chain_id=ChainId.ETHEREUM,
            address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            decimals=18,
            name="Wrapped Ether",
        )
        self.usdc = Token(
            chain_id=ChainId.ETHEREUM,
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            symbol="USDC",
            decimals=6,
        )
        self.native = Token(
            chain_id=ChainId.ETHEREUM,
            address="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
            symbol="ETH",
            decimals=18,
        )

    def test_str_representation(self):
        assert str(self.eth) == "WETH(1)"
        assert str(self.usdc) == "USDC(1)"

    def test_is_native_false(self):
        assert not self.eth.is_native()
        assert not self.usdc.is_native()

    def test_is_native_true(self):
        assert self.native.is_native()

    def test_is_native_case_insensitive(self):
        upper = Token(
            chain_id=1,
            address="0xEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
            symbol="ETH",
        )
        assert upper.is_native()

    def test_token_equality(self):
        eth2 = Token(
            chain_id=ChainId.ETHEREUM,
            address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            decimals=18,
            name="Wrapped Ether",
        )
        assert self.eth == eth2

    def test_token_inequality_different_chain(self):
        eth_arb = Token(
            chain_id=ChainId.ARBITRUM,
            address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            decimals=18,
        )
        assert self.eth != eth_arb

    def test_chain_id_enum(self):
        assert ChainId.ETHEREUM == 1
        assert ChainId.ARBITRUM == 42161
        assert ChainId.BASE == 8453


# ---------------------------------------------------------------------------
# TokenAmount tests
# ---------------------------------------------------------------------------


class TestTokenAmount:
    def setup_method(self):
        self.usdc = Token(
            chain_id=1,
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            symbol="USDC",
            decimals=6,
        )
        self.weth = Token(
            chain_id=1,
            address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            symbol="WETH",
            decimals=18,
        )

    def test_human_amount_usdc(self):
        amount = TokenAmount(token=self.usdc, amount=1_000_000)
        assert amount.human_amount == Decimal("1")

    def test_human_amount_weth(self):
        amount = TokenAmount(token=self.weth, amount=10**18)
        assert amount.human_amount == Decimal("1")

    def test_from_human_usdc(self):
        amount = TokenAmount.from_human(self.usdc, "100.5")
        assert amount.amount == 100_500_000

    def test_from_human_weth(self):
        amount = TokenAmount.from_human(self.weth, "0.5")
        assert amount.amount == 5 * 10**17

    def test_from_human_integer(self):
        amount = TokenAmount.from_human(self.usdc, 50)
        assert amount.amount == 50_000_000

    def test_repr(self):
        amount = TokenAmount(token=self.usdc, amount=1_000_000)
        assert "1" in repr(amount)
        assert "USDC" in repr(amount)

    def test_small_amount(self):
        amount = TokenAmount.from_human(self.usdc, "0.000001")
        assert amount.amount == 1

    def test_large_amount(self):
        amount = TokenAmount.from_human(self.weth, "1000000")
        assert amount.amount == 1_000_000 * 10**18


# ---------------------------------------------------------------------------
# SwapRoute tests
# ---------------------------------------------------------------------------


class TestSwapRoute:
    def setup_method(self):
        self.weth = Token(chain_id=1, address="0x" + "C0" * 20, symbol="WETH", decimals=18)
        self.usdc = Token(chain_id=1, address="0x" + "A0" * 20, symbol="USDC", decimals=6)
        self.dai = Token(chain_id=1, address="0x" + "D0" * 20, symbol="DAI", decimals=18)

    def test_single_hop_route(self):
        step = SwapStep(
            token_in=self.weth,
            token_out=self.usdc,
            pool_address="0x" + "11" * 20,
            protocol="UniswapV2",
            fee=3000,
        )
        route = SwapRoute(
            steps=[step],
            amount_in=TokenAmount.from_human(self.weth, "1"),
            amount_out=TokenAmount.from_human(self.usdc, "2000"),
            price_impact=Decimal("0.001"),
        )
        assert route.token_in == self.weth
        assert route.token_out == self.usdc
        assert len(route.steps) == 1

    def test_multi_hop_route(self):
        step1 = SwapStep(
            token_in=self.weth,
            token_out=self.dai,
            pool_address="0x" + "11" * 20,
            protocol="UniswapV2",
        )
        step2 = SwapStep(
            token_in=self.dai,
            token_out=self.usdc,
            pool_address="0x" + "22" * 20,
            protocol="Curve",
        )
        route = SwapRoute(
            steps=[step1, step2],
            amount_in=TokenAmount.from_human(self.weth, "1"),
            amount_out=TokenAmount.from_human(self.usdc, "1999"),
        )
        assert route.token_in == self.weth
        assert route.token_out == self.usdc
        assert len(route.steps) == 2

    def test_route_repr(self):
        step = SwapStep(
            token_in=self.weth,
            token_out=self.usdc,
            pool_address="0x" + "11" * 20,
            protocol="UniswapV2",
        )
        route = SwapRoute(
            steps=[step],
            amount_in=TokenAmount.from_human(self.weth, "1"),
            amount_out=TokenAmount.from_human(self.usdc, "2000"),
        )
        r = repr(route)
        assert "WETH" in r
        assert "USDC" in r


# ---------------------------------------------------------------------------
# BridgeQuote tests
# ---------------------------------------------------------------------------


class TestBridgeQuote:
    def setup_method(self):
        self.usdc_eth = Token(chain_id=1, address="0x" + "A0" * 20, symbol="USDC", decimals=6)
        self.usdc_arb = Token(chain_id=42161, address="0x" + "A1" * 20, symbol="USDC", decimals=6)

    def test_bridge_quote_creation(self):
        amount_in = TokenAmount.from_human(self.usdc_eth, "1000")
        amount_out = TokenAmount.from_human(self.usdc_arb, "999.4")
        fee = TokenAmount.from_human(self.usdc_eth, "0.6")

        quote = BridgeQuote(
            token_in=self.usdc_eth,
            token_out=self.usdc_arb,
            amount_in=amount_in,
            amount_out=amount_out,
            bridge_fee=fee,
            estimated_time_seconds=180,
            protocol="Stargate",
        )

        assert quote.protocol == "Stargate"
        assert quote.estimated_time_seconds == 180
        assert quote.amount_out.human_amount < quote.amount_in.human_amount
