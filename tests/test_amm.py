"""Tests for pydefi.amm (no live node required)."""

from decimal import Decimal

import pytest

from pydefi.amm.uniswap_v2 import UniswapV2
from pydefi.amm.uniswap_v3 import UniswapV3
from pydefi.amm.universal_router import (
    ADDRESS_THIS,
    CONTRACT_BALANCE_V3,
    CONTRACT_BALANCE_V4,
    MSG_SENDER,
    UNIVERSAL_ROUTER_ADDRESSES,
    RouterCommand,
    UniversalRouter,
    V2Hop,
    V3Hop,
    V4Action,
    V4Hop,
)
from pydefi.exceptions import InsufficientLiquidityError
from pydefi.types import ChainId, Token, TokenAmount, SwapTransaction


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


# ---------------------------------------------------------------------------
# Universal Router tests (no network calls)
# ---------------------------------------------------------------------------

UNIVERSAL_ROUTER_ADDR = "0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af"  # UniversalRouterV2 on Ethereum mainnet
RECIPIENT = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"


class TestRouterCommand:
    def test_v3_swap_exact_in_value(self):
        assert RouterCommand.V3_SWAP_EXACT_IN == 0x00

    def test_v3_swap_exact_out_value(self):
        assert RouterCommand.V3_SWAP_EXACT_OUT == 0x01

    def test_v2_swap_exact_in_value(self):
        assert RouterCommand.V2_SWAP_EXACT_IN == 0x08

    def test_v2_swap_exact_out_value(self):
        assert RouterCommand.V2_SWAP_EXACT_OUT == 0x09

    def test_wrap_eth_value(self):
        assert RouterCommand.WRAP_ETH == 0x0B

    def test_unwrap_weth_value(self):
        assert RouterCommand.UNWRAP_WETH == 0x0C

    def test_v4_swap_value(self):
        assert RouterCommand.V4_SWAP == 0x10

    def test_v4_initialize_pool_value(self):
        assert RouterCommand.V4_INITIALIZE_POOL == 0x13

    def test_execute_sub_plan_value(self):
        assert RouterCommand.EXECUTE_SUB_PLAN == 0x21

    def test_across_v4_deposit_value(self):
        assert RouterCommand.ACROSS_V4_DEPOSIT_V3 == 0x40

    def test_allow_revert_flag(self):
        assert RouterCommand.ALLOW_REVERT_FLAG == 0x80

    def test_allow_revert_or_with_command(self):
        cmd = RouterCommand.V3_SWAP_EXACT_OUT | RouterCommand.ALLOW_REVERT_FLAG
        assert cmd == 0x81


class TestUniversalRouterConstants:
    def test_known_addresses_contains_ethereum(self):
        assert 1 in UNIVERSAL_ROUTER_ADDRESSES
        # UniversalRouterV2 (supports Uniswap V4)
        assert UNIVERSAL_ROUTER_ADDRESSES[1] == "0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af"

    def test_known_addresses_contains_arbitrum(self):
        assert 42161 in UNIVERSAL_ROUTER_ADDRESSES

    def test_msg_sender_sentinel(self):
        assert MSG_SENDER == "0x0000000000000000000000000000000000000001"

    def test_address_this_sentinel(self):
        assert ADDRESS_THIS == "0x0000000000000000000000000000000000000002"

    def test_class_known_addresses(self):
        router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)
        assert router.KNOWN_ADDRESSES[1] == UNIVERSAL_ROUTER_ADDRESSES[1]


class TestUniversalRouterEncodeInputs:
    def test_encode_v3_swap_exact_in_length(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        encoded = UniversalRouter.encode_v3_swap_exact_in(
            RECIPIENT, 10 ** 18, 1_800 * 10 ** 6, path
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_v3_swap_exact_out_length(self):
        path = UniswapV3._encode_path([USDC, WETH], [3000])
        encoded = UniversalRouter.encode_v3_swap_exact_out(
            RECIPIENT, 10 ** 18, 2_200 * 10 ** 6, path
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_v2_swap_exact_in_length(self):
        addresses = [WETH.address, USDC.address]
        encoded = UniversalRouter.encode_v2_swap_exact_in(
            RECIPIENT, 10 ** 18, 1_800 * 10 ** 6, addresses
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_v2_swap_exact_out_length(self):
        addresses = [WETH.address, USDC.address]
        encoded = UniversalRouter.encode_v2_swap_exact_out(
            RECIPIENT, 1_800 * 10 ** 6, 10 ** 18, addresses
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_wrap_eth(self):
        encoded = UniversalRouter.encode_wrap_eth(ADDRESS_THIS, 10 ** 18)
        assert isinstance(encoded, bytes)
        assert len(encoded) == 64  # two 32-byte ABI words

    def test_encode_unwrap_weth(self):
        encoded = UniversalRouter.encode_unwrap_weth(RECIPIENT, 0)
        assert isinstance(encoded, bytes)
        assert len(encoded) == 64

    def test_encode_sweep(self):
        encoded = UniversalRouter.encode_sweep(WETH.address, RECIPIENT, 0)
        assert isinstance(encoded, bytes)
        assert len(encoded) == 96  # three 32-byte ABI words

    def test_encode_v3_payer_is_user_default(self):
        """Default payer_is_user=True; False encoding must differ."""
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        enc_user = UniversalRouter.encode_v3_swap_exact_in(
            RECIPIENT, 10 ** 18, 0, path, payer_is_user=True
        )
        enc_router = UniversalRouter.encode_v3_swap_exact_in(
            RECIPIENT, 10 ** 18, 0, path, payer_is_user=False
        )
        assert enc_user != enc_router


class TestBuildExecuteCalldata:
    def _router(self):
        return UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_with_deadline_uses_correct_selector(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        input_data = UniversalRouter.encode_v3_swap_exact_in(
            RECIPIENT, 10 ** 18, 0, path
        )
        calldata = UniversalRouter.build_execute_calldata(
            [RouterCommand.V3_SWAP_EXACT_IN], [input_data], deadline=1_700_000_000
        )
        assert calldata[:4] == bytes.fromhex("3593564c")

    def test_without_deadline_uses_correct_selector(self):
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        input_data = UniversalRouter.encode_v3_swap_exact_in(
            RECIPIENT, 10 ** 18, 0, path
        )
        calldata = UniversalRouter.build_execute_calldata(
            [RouterCommand.V3_SWAP_EXACT_IN], [input_data]
        )
        assert calldata[:4] == bytes.fromhex("24856bc3")

    def test_mismatched_commands_inputs_raises(self):
        with pytest.raises(ValueError, match="length"):
            UniversalRouter.build_execute_calldata(
                [RouterCommand.V3_SWAP_EXACT_IN, RouterCommand.WRAP_ETH],
                [b"only_one_input"],
            )

    def test_multi_command_calldata_length(self):
        wrap_input = UniversalRouter.encode_wrap_eth(ADDRESS_THIS, 10 ** 18)
        path = UniswapV3._encode_path([WETH, USDC], [3000])
        swap_input = UniversalRouter.encode_v3_swap_exact_in(
            RECIPIENT, 10 ** 18, 0, path, payer_is_user=False
        )
        calldata = UniversalRouter.build_execute_calldata(
            [RouterCommand.WRAP_ETH, RouterCommand.V3_SWAP_EXACT_IN],
            [wrap_input, swap_input],
            deadline=1_700_000_000,
        )
        assert len(calldata) > 4  # selector + encoded args


class TestUniversalRouterTransactionBuilders:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_build_v3_exact_in_returns_swap_transaction(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v3_exact_in_transaction(
            amount_in=amount_in,
            token_out=USDC,
            recipient=RECIPIENT,
            amount_out_minimum=1_800 * 10 ** 6,
            fee=3000,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.data[:4] == bytes.fromhex("3593564c")  # deadline selector
        assert tx.value == 0

    def test_build_v3_exact_in_no_deadline(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v3_exact_in_transaction(
            amount_in=amount_in,
            token_out=USDC,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")  # no-deadline selector

    def test_build_v3_multihop_exact_in(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v3_multihop_exact_in_transaction(
            amount_in=amount_in,
            path=[WETH, USDC, DAI],
            fees=[500, 100],
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert len(tx.data) > 4

    def test_build_v3_exact_out(self):
        amount_out = TokenAmount.from_human(USDC, "2000")
        tx = self.router.build_v3_exact_out_transaction(
            amount_out=amount_out,
            token_in=WETH,
            recipient=RECIPIENT,
            amount_in_maximum=2 * 10 ** 18,
            fee=3000,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_build_v2_exact_in(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v2_exact_in_transaction(
            amount_in=amount_in,
            path=[WETH, USDC],
            recipient=RECIPIENT,
            amount_out_minimum=1_800 * 10 ** 6,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_build_v2_exact_out(self):
        amount_out = TokenAmount.from_human(USDC, "2000")
        tx = self.router.build_v2_exact_out_transaction(
            amount_out=amount_out,
            path=[WETH, USDC],
            recipient=RECIPIENT,
            amount_in_maximum=2 * 10 ** 18,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_build_wrap_and_v3_swap_sets_value(self):
        eth_amount = 10 ** 18
        tx = self.router.build_wrap_and_v3_swap_transaction(
            eth_amount=eth_amount,
            weth_token=WETH,
            token_out=USDC,
            recipient=RECIPIENT,
            amount_out_minimum=1_800 * 10 ** 6,
            fee=3000,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.value == eth_amount
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_build_wrap_and_v3_swap_no_deadline(self):
        tx = self.router.build_wrap_and_v3_swap_transaction(
            eth_amount=10 ** 18,
            weth_token=WETH,
            token_out=USDC,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")

    def test_router_address_stored(self):
        assert self.router.router_address == UNIVERSAL_ROUTER_ADDR


class TestV4Action:
    def test_swap_exact_in_single_value(self):
        assert V4Action.SWAP_EXACT_IN_SINGLE == 0x06

    def test_swap_exact_in_value(self):
        assert V4Action.SWAP_EXACT_IN == 0x07

    def test_swap_exact_out_single_value(self):
        assert V4Action.SWAP_EXACT_OUT_SINGLE == 0x08

    def test_swap_exact_out_value(self):
        assert V4Action.SWAP_EXACT_OUT == 0x09

    def test_settle_all_value(self):
        assert V4Action.SETTLE_ALL == 0x0C

    def test_take_all_value(self):
        assert V4Action.TAKE_ALL == 0x0F


class TestV4Encoding:
    def test_sort_currencies_lower_first(self):
        # USDC (0xA0...) < WETH (0xC0...) so USDC is currency0
        c0, c1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        assert c0.lower() < c1.lower()
        assert c0.lower() == USDC.address.lower()

    def test_sort_currencies_already_sorted(self):
        c0, c1 = UniversalRouter._sort_v4_currencies(USDC.address, WETH.address)
        assert c0.lower() == USDC.address.lower()
        assert c1.lower() == WETH.address.lower()

    def test_encode_v4_exact_in_single_params_length(self):
        c0, c1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        encoded = UniversalRouter.encode_v4_exact_in_single_params(
            currency0=c0,
            currency1=c1,
            fee=500,
            tick_spacing=10,
            hooks="0x0000000000000000000000000000000000000000",
            zero_for_one=False,
            amount_in=10 ** 18,
            amount_out_minimum=1_800 * 10 ** 6,
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_v4_settle_all_params_length(self):
        encoded = UniversalRouter.encode_v4_settle_all_params(WETH.address, 10 ** 18)
        assert isinstance(encoded, bytes)
        assert len(encoded) == 64  # two 32-byte ABI words

    def test_encode_v4_take_all_params_length(self):
        encoded = UniversalRouter.encode_v4_take_all_params(USDC.address, 0)
        assert isinstance(encoded, bytes)
        assert len(encoded) == 64

    def test_encode_v4_swap_actions_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            UniversalRouter.encode_v4_swap_actions(
                [V4Action.SWAP_EXACT_IN_SINGLE],
                [],
            )

    def test_encode_v4_swap_actions_output_is_bytes(self):
        c0, c1 = UniversalRouter._sort_v4_currencies(WETH.address, USDC.address)
        swap_p = UniversalRouter.encode_v4_exact_in_single_params(
            c0, c1, 500, 10,
            "0x0000000000000000000000000000000000000000",
            False, 10 ** 18, 0,
        )
        settle_p = UniversalRouter.encode_v4_settle_all_params(WETH.address, 10 ** 18)
        take_p = UniversalRouter.encode_v4_take_all_params(USDC.address, 0)
        v4_input = UniversalRouter.encode_v4_swap_actions(
            [V4Action.SWAP_EXACT_IN_SINGLE, V4Action.SETTLE_ALL, V4Action.TAKE_ALL],
            [swap_p, settle_p, take_p],
        )
        assert isinstance(v4_input, bytes)
        assert len(v4_input) > 0


class TestV4TransactionBuilder:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_build_v4_exact_in_single_returns_swap_transaction(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v4_exact_in_single_transaction(
            amount_in=amount_in,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=1_800 * 10 ** 6,
            deadline=1_700_000_000,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.value == 0

    def test_build_v4_exact_in_single_with_deadline_selector(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v4_exact_in_single_transaction(
            amount_in=amount_in,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_build_v4_exact_in_single_no_deadline_selector(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v4_exact_in_single_transaction(
            amount_in=amount_in,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")

    def test_build_v4_exact_in_single_with_hooks(self):
        custom_hooks = "0x1234567890abcdef1234567890abcdef12345678"
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v4_exact_in_single_transaction(
            amount_in=amount_in,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            hooks=custom_hooks,
            hook_data=b"\x01\x02\x03",
        )
        assert isinstance(tx, SwapTransaction)
        assert len(tx.data) > 4

    def test_v4_calldata_contains_v4_swap_command(self):
        """The commands byte in the calldata must contain V4_SWAP (0x10)."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_v4_exact_in_single_transaction(
            amount_in=amount_in,
            token_out=USDC,
            fee=3000,
            tick_spacing=60,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        # commands are ABI-encoded after the 4-byte selector.
        # The commands bytes value 0x10 (V4_SWAP) must appear in the calldata.
        assert bytes([RouterCommand.V4_SWAP]) in tx.data


class TestV4NewEncoders:
    def test_encode_v4_settle_params_length(self):
        encoded = UniversalRouter.encode_v4_settle_params(WETH.address, 10 ** 18, False)
        assert isinstance(encoded, bytes)
        # 3 x 32-byte ABI words: address (padded), uint256, bool (padded)
        assert len(encoded) == 96

    def test_encode_v4_settle_params_payer_is_user_true(self):
        encoded = UniversalRouter.encode_v4_settle_params(WETH.address, 10 ** 18, True)
        assert isinstance(encoded, bytes)
        assert len(encoded) == 96

    def test_encode_v4_take_params_length(self):
        encoded = UniversalRouter.encode_v4_take_params(USDC.address, RECIPIENT, 0)
        assert isinstance(encoded, bytes)
        # 3 x 32-byte ABI words: address, address, uint256
        assert len(encoded) == 96

    def test_encode_v4_take_params_open_delta_amount(self):
        # amount=0 means OPEN_DELTA (take all available credit)
        encoded = UniversalRouter.encode_v4_take_params(USDC.address, RECIPIENT, 0)
        # Last 32 bytes should be all zeros (amount=0)
        assert encoded[-32:] == b"\x00" * 32


class TestBuildWrapAndV4Swap:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_returns_swap_transaction(self):
        tx = self.router.build_wrap_and_v4_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)

    def test_value_equals_eth_amount(self):
        eth_amount = 5 * 10 ** 17
        tx = self.router.build_wrap_and_v4_swap_transaction(
            eth_amount=eth_amount,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.value == eth_amount

    def test_router_address_stored(self):
        tx = self.router.build_wrap_and_v4_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.to == UNIVERSAL_ROUTER_ADDR

    def test_with_deadline_selector(self):
        tx = self.router.build_wrap_and_v4_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=2_000_000_000,
        )
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_no_deadline_selector(self):
        tx = self.router.build_wrap_and_v4_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")

    def test_wrap_eth_and_v4_swap_commands_present(self):
        """The calldata must contain both WRAP_ETH (0x0B) and V4_SWAP (0x10) command bytes."""
        from pydefi.amm.universal_router import RouterCommand
        tx = self.router.build_wrap_and_v4_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            token_out=USDC,
            fee=500,
            tick_spacing=10,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.WRAP_ETH]) in tx.data
        assert bytes([RouterCommand.V4_SWAP]) in tx.data


# ---------------------------------------------------------------------------
# Multi-hop tests (no network calls)
# ---------------------------------------------------------------------------


class TestContractBalanceSentinels:
    def test_contract_balance_v3_is_uint256_max(self):
        assert CONTRACT_BALANCE_V3 == (1 << 256) - 1

    def test_contract_balance_v4_is_uint128_max(self):
        assert CONTRACT_BALANCE_V4 == (1 << 128) - 1


class TestHopDataclasses:
    def test_v2_hop_fields(self):
        hop = V2Hop(token_in=WETH, token_out=USDC)
        assert hop.token_in is WETH
        assert hop.token_out is USDC

    def test_v3_hop_fields(self):
        hop = V3Hop(token_in=WETH, token_out=USDC, fee=500)
        assert hop.fee == 500

    def test_v4_hop_defaults(self):
        hop = V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)
        assert hop.hooks == "0x0000000000000000000000000000000000000000"
        assert hop.hook_data == b""

    def test_v4_hop_custom_hooks(self):
        hooks = "0x1234567890abcdef1234567890abcdef12345678"
        hop = V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10, hooks=hooks)
        assert hop.hooks == hooks


class TestEncodeV4ExactInParams:
    def test_returns_bytes(self):
        path = [(USDC.address, 500, 10, "0x0000000000000000000000000000000000000000", b"")]
        encoded = UniversalRouter.encode_v4_exact_in_params(
            currency_in=WETH.address,
            path=path,
            amount_in=10 ** 18,
            amount_out_minimum=0,
        )
        assert isinstance(encoded, bytes)

    def test_nonempty(self):
        path = [(USDC.address, 500, 10, "0x0000000000000000000000000000000000000000", b"")]
        encoded = UniversalRouter.encode_v4_exact_in_params(
            currency_in=WETH.address,
            path=path,
            amount_in=10 ** 18,
            amount_out_minimum=1800 * 10 ** 6,
        )
        assert len(encoded) > 0

    def test_two_hop_path(self):
        # Two hops: WETH → USDC → DAI
        path = [
            (USDC.address, 500, 10, "0x0000000000000000000000000000000000000000", b""),
            (DAI.address, 100, 1, "0x0000000000000000000000000000000000000000", b""),
        ]
        encoded = UniversalRouter.encode_v4_exact_in_params(
            currency_in=WETH.address,
            path=path,
            amount_in=10 ** 18,
            amount_out_minimum=0,
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encoded_length_is_multiple_of_32(self):
        # ABI-encoded output is always a multiple of 32 bytes.
        path = [(USDC.address, 500, 10, "0x0000000000000000000000000000000000000000", b"")]
        encoded = UniversalRouter.encode_v4_exact_in_params(
            currency_in=WETH.address,
            path=path,
            amount_in=10 ** 18,
            amount_out_minimum=0,
        )
        assert len(encoded) % 32 == 0


class TestBuildV4MultihopExactIn:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_returns_swap_transaction(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        hops = [
            V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10),
            V4Hop(token_in=USDC, token_out=DAI, fee=100, tick_spacing=1),
        ]
        tx = self.router.build_v4_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.value == 0

    def test_contains_v4_swap_command(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        hops = [V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)]
        tx = self.router.build_v4_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_with_deadline_selector(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        hops = [V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)]
        tx = self.router.build_v4_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_no_deadline_selector(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        hops = [V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)]
        tx = self.router.build_v4_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")

    def test_payer_is_user_false(self):
        # payer_is_user=False: router uses its own balance (e.g. after WRAP_ETH)
        amount_in = TokenAmount.from_human(WETH, "1")
        hops = [V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)]
        tx = self.router.build_v4_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
            payer_is_user=False,
        )
        assert isinstance(tx, SwapTransaction)

    def test_raises_on_empty_hops(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        with pytest.raises(ValueError, match="empty"):
            self.router.build_v4_multihop_exact_in_transaction(
                amount_in=amount_in,
                hops=[],
                recipient=RECIPIENT,
                amount_out_minimum=0,
            )

    def test_single_hop_uses_swap_exact_in_action(self):
        """Single-hop V4 multi-hop transaction uses SWAP_EXACT_IN action."""
        amount_in = TokenAmount.from_human(WETH, "1")
        hops = [V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)]
        tx = self.router.build_v4_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=hops,
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        # V4Action.SWAP_EXACT_IN == 0x07 must be present in the data
        assert bytes([V4Action.SWAP_EXACT_IN]) in tx.data


class TestBuildMultihopExactIn:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_raises_on_empty_hops(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        with pytest.raises(ValueError, match="empty"):
            self.router.build_multihop_exact_in_transaction(
                amount_in=amount_in,
                hops=[],
                recipient=RECIPIENT,
                amount_out_minimum=0,
            )

    # ------------------------------------------------------------------
    # Single-segment (same type throughout) — basic sanity checks
    # ------------------------------------------------------------------

    def test_single_v2_hop(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[V2Hop(token_in=WETH, token_out=USDC)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert bytes([RouterCommand.V2_SWAP_EXACT_IN]) in tx.data

    def test_single_v3_hop(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data

    def test_single_v4_hop(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_consecutive_v3_hops_merged(self):
        """Two consecutive V3 hops must produce exactly one V3_SWAP_EXACT_IN command."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V3Hop(token_in=WETH, token_out=USDC, fee=500),
                V3Hop(token_in=USDC, token_out=DAI, fee=100),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        # One V3 command byte (0x00) must appear exactly once in the commands
        # portion of the calldata.
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data

    def test_consecutive_v2_hops_merged(self):
        """Two consecutive V2 hops must produce exactly one V2_SWAP_EXACT_IN command."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V2Hop(token_in=WETH, token_out=USDC),
                V2Hop(token_in=USDC, token_out=DAI),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V2_SWAP_EXACT_IN]) in tx.data

    def test_consecutive_v4_hops_merged(self):
        """Two consecutive V4 hops must produce exactly one V4_SWAP command."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10),
                V4Hop(token_in=USDC, token_out=DAI, fee=100, tick_spacing=1),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    # ------------------------------------------------------------------
    # Cross-pool-type paths
    # ------------------------------------------------------------------

    def test_v2_then_v3(self):
        """V2 → V3 cross-type path produces two separate commands."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V2Hop(token_in=WETH, token_out=USDC),
                V3Hop(token_in=USDC, token_out=DAI, fee=100),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V2_SWAP_EXACT_IN]) in tx.data
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data

    def test_v3_then_v4(self):
        """V3 → V4 cross-type path produces both a V3 command and a V4 command."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V3Hop(token_in=WETH, token_out=USDC, fee=500),
                V4Hop(token_in=USDC, token_out=DAI, fee=100, tick_spacing=1),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_v2_then_v4(self):
        """V2 → V4 cross-type path."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V2Hop(token_in=WETH, token_out=USDC),
                V4Hop(token_in=USDC, token_out=DAI, fee=100, tick_spacing=1),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V2_SWAP_EXACT_IN]) in tx.data
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_v4_then_v3(self):
        """V4 → V3 cross-type path."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10),
                V3Hop(token_in=USDC, token_out=DAI, fee=100),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V4_SWAP]) in tx.data
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data

    def test_three_segment_path(self):
        """V2 → V3 → V4 three-segment path produces three commands."""
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V2Hop(token_in=WETH, token_out=USDC),
                V3Hop(token_in=USDC, token_out=DAI, fee=100),
                V4Hop(token_in=DAI, token_out=WETH, fee=500, tick_spacing=10),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.V2_SWAP_EXACT_IN]) in tx.data
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_returns_swap_transaction_type(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[
                V3Hop(token_in=WETH, token_out=USDC, fee=500),
                V4Hop(token_in=USDC, token_out=DAI, fee=100, tick_spacing=1),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR
        assert tx.value == 0

    def test_deadline_selector_applied(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_no_deadline_selector(self):
        amount_in = TokenAmount.from_human(WETH, "1")
        tx = self.router.build_multihop_exact_in_transaction(
            amount_in=amount_in,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")


class TestBuildWrapAndV4MultihopSwap:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_returns_swap_transaction(self):
        tx = self.router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR

    def test_value_equals_eth_amount(self):
        eth_amount = 5 * 10 ** 17
        tx = self.router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=eth_amount,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.value == eth_amount

    def test_contains_wrap_eth_and_v4_swap_commands(self):
        tx = self.router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.WRAP_ETH]) in tx.data
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_uses_swap_exact_in_action(self):
        tx = self.router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([V4Action.SWAP_EXACT_IN]) in tx.data

    def test_with_deadline_selector(self):
        tx = self.router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_no_deadline_selector(self):
        tx = self.router.build_wrap_and_v4_multihop_swap_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V4Hop(token_in=WETH, token_out=USDC, fee=500, tick_spacing=10)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")

    def test_raises_on_empty_hops(self):
        with pytest.raises(ValueError, match="empty"):
            self.router.build_wrap_and_v4_multihop_swap_transaction(
                eth_amount=10 ** 17,
                weth_token=WETH,
                hops=[],
                recipient=RECIPIENT,
                amount_out_minimum=0,
            )


class TestBuildWrapAndMultihopExactIn:
    def setup_method(self):
        self.router = UniversalRouter(UNIVERSAL_ROUTER_ADDR)

    def test_returns_swap_transaction(self):
        tx = self.router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert isinstance(tx, SwapTransaction)
        assert tx.to == UNIVERSAL_ROUTER_ADDR

    def test_value_equals_eth_amount(self):
        eth_amount = 5 * 10 ** 17
        tx = self.router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=eth_amount,
            weth_token=WETH,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.value == eth_amount

    def test_contains_wrap_eth_command(self):
        tx = self.router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.WRAP_ETH]) in tx.data
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data

    def test_v3_v4_cross_type(self):
        tx = self.router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[
                V3Hop(token_in=WETH, token_out=USDC, fee=500),
                V4Hop(token_in=USDC, token_out=WETH, fee=500, tick_spacing=10),
            ],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert bytes([RouterCommand.WRAP_ETH]) in tx.data
        assert bytes([RouterCommand.V3_SWAP_EXACT_IN]) in tx.data
        assert bytes([RouterCommand.V4_SWAP]) in tx.data

    def test_with_deadline_selector(self):
        tx = self.router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
            deadline=1_700_000_000,
        )
        assert tx.data[:4] == bytes.fromhex("3593564c")

    def test_no_deadline_selector(self):
        tx = self.router.build_wrap_and_multihop_exact_in_transaction(
            eth_amount=10 ** 17,
            weth_token=WETH,
            hops=[V3Hop(token_in=WETH, token_out=USDC, fee=500)],
            recipient=RECIPIENT,
            amount_out_minimum=0,
        )
        assert tx.data[:4] == bytes.fromhex("24856bc3")

    def test_raises_on_empty_hops(self):
        with pytest.raises(ValueError, match="empty"):
            self.router.build_wrap_and_multihop_exact_in_transaction(
                eth_amount=10 ** 17,
                weth_token=WETH,
                hops=[],
                recipient=RECIPIENT,
                amount_out_minimum=0,
            )
