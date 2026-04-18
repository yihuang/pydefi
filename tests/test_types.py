"""Tests for pydefi.types"""

from decimal import Decimal

import pytest

from pydefi.types import (
    Address,
    BridgeQuote,
    ChainId,
    RouteDAG,
    RouteSplit,
    RouteSwap,
    SwapRoute,
    SwapStep,
    Token,
    TokenAmount,
)
from tests.addrs import USDC, WETH

# ---------------------------------------------------------------------------
# Token tests
# ---------------------------------------------------------------------------


class TestToken:
    def setup_method(self):
        self.eth = Token(
            chain_id=ChainId.ETHEREUM,
            address=WETH.address,
            symbol="WETH",
            decimals=18,
            name="Wrapped Ether",
        )
        self.usdc = Token(
            chain_id=ChainId.ETHEREUM,
            address=USDC.address,
            symbol="USDC",
            decimals=6,
        )
        self.native = Token(
            chain_id=ChainId.ETHEREUM,
            address=Address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),
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
            address=Address("0xEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"),
            symbol="ETH",
        )
        assert upper.is_native()

    def test_token_equality(self):
        eth2 = Token(
            chain_id=ChainId.ETHEREUM,
            address=WETH.address,
            symbol="WETH",
            decimals=18,
            name="Wrapped Ether",
        )
        assert self.eth == eth2

    def test_token_inequality_different_chain(self):
        eth_arb = Token(
            chain_id=ChainId.ARBITRUM,
            address=WETH.address,
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
            address=USDC.address,
            symbol="USDC",
            decimals=6,
        )
        self.weth = Token(
            chain_id=1,
            address=WETH.address,
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
        self.weth = Token(chain_id=1, address=Address("0x" + "C0" * 20), symbol="WETH", decimals=18)
        self.usdc = Token(chain_id=1, address=Address("0x" + "A0" * 20), symbol="USDC", decimals=6)
        self.dai = Token(chain_id=1, address=Address("0x" + "D0" * 20), symbol="DAI", decimals=18)

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
# RouteDAG tests
# ---------------------------------------------------------------------------


class TestRouteDAG:
    def setup_method(self):
        self.token0 = Token(chain_id=1, address="0x" + "10" * 20, symbol="T0")
        self.token1 = Token(chain_id=1, address="0x" + "11" * 20, symbol="T1")
        self.token2 = Token(chain_id=1, address="0x" + "12" * 20, symbol="T2")
        self.token3 = Token(chain_id=1, address="0x" + "13" * 20, symbol="T3")
        self.token4 = Token(chain_id=1, address="0x" + "14" * 20, symbol="T4")

    def test_split_merge_dag_construction(self):
        dag = (
            RouteDAG()
            .from_token(self.token0)
            .swap(self.token1, "pool1")
            .split()
            .leg(5000)
            .swap(self.token2, "pool2")
            .swap(self.token3, "pool3")
            .leg(5000)
            .swap(self.token3, "pool4")
            .merge()
            .swap(self.token4, "pool5")
        )

        payload = dag.to_dict()
        assert payload["token_in"] == self.token0
        assert len(payload["actions"]) == 3
        assert isinstance(payload["actions"][0], RouteSwap)
        assert isinstance(payload["actions"][1], RouteSplit)
        assert isinstance(payload["actions"][2], RouteSwap)
        assert [leg.fraction_bps for leg in payload["actions"][1].legs] == [5000, 5000]

    def test_merge_requires_sum_exactly_10000(self):
        # Two split() calls create two legs: 3000 + 3000 = 6000, which is invalid.
        dag = (
            RouteDAG()
            .from_token(self.token0)
            .split()
            .leg(3000)
            .swap(self.token1, "pool1")
            .leg(3000)
            .swap(self.token1, "pool2")
        )
        with pytest.raises(ValueError, match="fraction_bps"):
            dag.merge()

    def test_merge_requires_same_end_token(self):
        dag = (
            RouteDAG()
            .from_token(self.token0)
            .split()
            .leg(5000)
            .swap(self.token1, "pool1")
            .leg(5000)
            .swap(self.token2, "pool2")
        )
        with pytest.raises(ValueError, match="same token"):
            dag.merge()

    def test_to_dict_rejects_unmerged_split(self):
        dag = RouteDAG().from_token(self.token0).split().leg(10000).swap(self.token1, "pool1")
        with pytest.raises(ValueError, match="unmerged"):
            dag.to_dict()

    def test_nested_split_is_explicit_and_supported(self):
        dag = (
            RouteDAG()
            .from_token(self.token0)
            .split()
            .leg(5000)
            .swap(self.token1, "pool1")
            .leg(5000)
            .swap(self.token1, "pool_seed")
            .split()
            .leg(5000)
            .swap(self.token1, "pool2")
            .leg(5000)
            .swap(self.token1, "pool3")
            .merge()
            .merge()
        )

        payload = dag.to_dict()
        outer = payload["actions"][0]
        assert isinstance(outer, RouteSplit)
        assert len(outer.legs) == 2
        assert isinstance(outer.legs[1].actions[0], RouteSwap)
        nested = outer.legs[1].actions[1]
        assert isinstance(nested, RouteSplit)
        assert [leg.fraction_bps for leg in nested.legs] == [5000, 5000]

    def test_nested_split_from_empty_active_leg(self):
        dag = (
            RouteDAG()
            .from_token(self.token0)
            .split()
            .leg(5000)
            .split()
            .leg(5000)
            .swap(self.token1, "pool1")
            .leg(5000)
            .swap(self.token1, "pool2")
            .merge()
            .leg(5000)
            .swap(self.token1, "pool3")
            .merge()
        )

        payload = dag.to_dict()
        outer = payload["actions"][0]
        assert isinstance(outer, RouteSplit)
        assert isinstance(outer.legs[0].actions[0], RouteSplit)
        nested = outer.legs[0].actions[0]
        assert isinstance(nested, RouteSplit)
        assert [leg.fraction_bps for leg in nested.legs] == [5000, 5000]

    def test_swap_inside_split_requires_leg(self):
        dag = RouteDAG().from_token(self.token0).split()
        with pytest.raises(ValueError, match=r"leg\(\) must be called before swap\(\) inside split\(\)"):
            dag.swap(self.token1, "pool1")


# ---------------------------------------------------------------------------
# BridgeQuote tests
# ---------------------------------------------------------------------------


class TestBridgeQuote:
    def setup_method(self):
        self.usdc_eth = Token(chain_id=1, address=Address("0x" + "A0" * 20), symbol="USDC", decimals=6)
        self.usdc_arb = Token(chain_id=42161, address=Address("0x" + "A1" * 20), symbol="USDC", decimals=6)

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
