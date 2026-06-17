"""
pydefi.abi — centralised ABI definitions for all supported DeFi protocols.

Import Contract objects and ABIStruct classes from the sub-modules::

    from pydefi.abi.amm import UNISWAP_V2_ROUTER, UNISWAP_V3_ROUTER
    from pydefi.abi.bridge import CCTP_TOKEN_MESSENGER_V2, LAYERZERO_OFT
"""

from .amm import (
    CURVE_METAREGISTRY,
    CURVE_POOL,
    CURVE_REGISTRY,
    CURVE_V2_POOL,
    UNISWAP_V2_FACTORY,
    UNISWAP_V2_PAIR,
    UNISWAP_V2_ROUTER,
    UNISWAP_V3_FACTORY,
    UNISWAP_V3_POOL,
    UNISWAP_V3_QUOTER_V2,
    UNISWAP_V3_ROUTER,
    ExactInputParams,
    ExactInputSingleParams,
    ExactOutputSingleParams,
    QuoteExactInputSingleParams,
    QuoteExactOutputSingleParams,
)
from .bridge import (
    ACROSS_SPOKE_POOL,
    CCTP_TOKEN_MESSENGER_V2,
    GASZIP,
    LAYERZERO_OFT,
    MAYAN_FORWARDER,
    MAYAN_SWIFT_V2,
    STARGATE_FACTORY,
    STARGATE_POOL,
    STARGATE_ROUTER,
    MayanSwiftOrderParams,
    MessagingFee,
    OFTSendParam,
)
from .dex_aggregator import OKX_DEX_ROUTER, BaseRequest, RouterPath
from .lending import (
    AAVE_V3_ADDRESSES_PROVIDER,
    AAVE_V3_DATA_PROVIDER,
    AAVE_V3_ORACLE,
    AAVE_V3_POOL,
    COMPOUND_V3_COMET,
    CometAssetInfo,
    ReserveConfigurationMap,
    ReserveDataLegacy,
)
from .vm import DeFiVM

__all__ = [
    # amm
    "CURVE_POOL",
    "CURVE_METAREGISTRY",
    "CURVE_REGISTRY",
    "CURVE_V2_POOL",
    "UNISWAP_V2_FACTORY",
    "UNISWAP_V2_PAIR",
    "UNISWAP_V2_ROUTER",
    "UNISWAP_V3_FACTORY",
    "UNISWAP_V3_POOL",
    "UNISWAP_V3_QUOTER_V2",
    "UNISWAP_V3_ROUTER",
    "ExactInputParams",
    "ExactInputSingleParams",
    "ExactOutputSingleParams",
    "QuoteExactInputSingleParams",
    "QuoteExactOutputSingleParams",
    # dex aggregator
    "OKX_DEX_ROUTER",
    "BaseRequest",
    "RouterPath",
    # bridge
    "ACROSS_SPOKE_POOL",
    "CCTP_TOKEN_MESSENGER_V2",
    "GASZIP",
    "LAYERZERO_OFT",
    "MAYAN_FORWARDER",
    "MAYAN_SWIFT_V2",
    "STARGATE_FACTORY",
    "STARGATE_POOL",
    "STARGATE_ROUTER",
    "MayanSwiftOrderParams",
    "MessagingFee",
    "OFTSendParam",
    # lending
    "AAVE_V3_ADDRESSES_PROVIDER",
    "AAVE_V3_DATA_PROVIDER",
    "AAVE_V3_ORACLE",
    "AAVE_V3_POOL",
    "COMPOUND_V3_COMET",
    "CometAssetInfo",
    "ReserveConfigurationMap",
    "ReserveDataLegacy",
    # vm
    "DeFiVM",
]
