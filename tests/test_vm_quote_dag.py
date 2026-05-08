from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from eth_abi import encode

from pydefi.abi import DeFiVM
from pydefi.pathfinder.graph import PoolEdge
from pydefi.types import Address, RouteDAG
from pydefi.vm.swap import quote_dag
from tests.addrs import USDC, WETH

POOL_A: Address = Address("0x" + "11" * 20)
VM_ADDRESS: Address = Address("0x" + "22" * 20)


@pytest.fixture
def v2_dag() -> RouteDAG:
    edge = PoolEdge(
        token_in=WETH,
        token_out=USDC,
        pool_address=POOL_A,
        protocol="UniswapV2",
        reserve_in=1_000 * 10**18,
        reserve_out=2_000_000 * 10**6,
        fee_bps=30,
        extra={"is_token0_in": True},
    )
    return RouteDAG().from_token(WETH).swap(USDC, edge)


def _mock_w3(returndata: bytes) -> MagicMock:
    w3 = MagicMock()
    w3.eth.call = AsyncMock(return_value=returndata)
    return w3


class TestQuoteDag:
    @pytest.mark.parametrize(
        "returndata, expected",
        [
            (encode(["uint256"], [1_999_398_407]), 1_999_398_407),
            (encode(["uint256"], [0]), 0),
            (b"", 0),  # empty returndata → degenerate liquidity
        ],
    )
    async def test_decodes_returndata(self, v2_dag, returndata, expected):
        w3 = _mock_w3(returndata)
        out = await quote_dag(v2_dag, amount_in=10**18, w3=w3, vm_address=VM_ADDRESS)
        assert out == expected
        w3.eth.call.assert_awaited_once()

    async def test_calldata_targets_defivm_execute(self, v2_dag):
        w3 = _mock_w3(encode(["uint256"], [42]))
        await quote_dag(v2_dag, amount_in=10**18, w3=w3, vm_address=VM_ADDRESS)
        (call_kwargs,) = w3.eth.call.await_args.args
        assert call_kwargs["to"] == VM_ADDRESS
        execute_selector = bytes(DeFiVM.fns.execute(b"").data[:4])
        assert bytes(call_kwargs["data"]).startswith(execute_selector)
