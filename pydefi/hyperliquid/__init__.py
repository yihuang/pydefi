"""Hyperliquid L1 (HyperCore) and HyperEVM utilities.

This module provides:

* :class:`~pydefi.hyperliquid.client.HyperliquidClient` — async HTTP client
  for the Hyperliquid L1 info and exchange APIs, plus a helper to create an
  :class:`~web3.AsyncWeb3` instance for HyperEVM.

* Signing utilities in :mod:`~pydefi.hyperliquid.signing` — EIP-712 helpers
  for both phantom-agent signing (trading actions) and user-signed action
  signing (transfers, withdrawals, etc.).

Quick-start::

    from pydefi.hyperliquid import HyperliquidClient

    client = HyperliquidClient()

    # Read-only info query (no credentials needed)
    meta = await client.get_meta()
    mids = await client.get_all_mids()

    # HyperEVM access (chain ID 999)
    w3 = client.make_evm_w3()
    block = await w3.eth.get_block("latest")

    # Signed action (requires private key)
    result = await client.usd_send(
        private_key="0x...",
        destination="0x...",
        amount="10.0",
    )

CCTP bridge to HyperCore
-------------------------
HyperCore (``ChainId.HYPERCORE = 1337``) is Hyperliquid's L1 chain.  To bridge
USDC from any supported EVM chain directly to a HyperCore address use
``dst_chain_id=ChainId.HYPERCORE``.  The CCTP bridge uses
``depositForBurnWithHook`` to mint USDC on HyperEVM and the ``CctpForwarder``
contract automatically routes it to HyperCore::

    from pydefi.bridge import CCTP
    from pydefi.types import ChainId

    bridge = CCTP(w3=eth_w3, src_chain_id=ChainId.ETHEREUM, dst_chain_id=ChainId.HYPERCORE)
    quote = await bridge.get_quote(usdc_eth, usdc_hypercore, amount_in)
    # recipient gets USDC credited to their HyperCore perp balance
    tx = await bridge.build_bridge_tx(usdc_eth, usdc_hypercore, amount_in, recipient)

Withdrawing from HyperCore
---------------------------
To withdraw USDC from HyperCore back to an EVM chain use
:meth:`~pydefi.hyperliquid.client.HyperliquidClient.send_to_evm_with_data`::

    # Withdraw 10 USDC from HyperCore perp to Arbitrum mainnet
    await client.send_to_evm_with_data(
        private_key=pk,
        destination_recipient="0xYourAddress",
        amount="10",
        destination_chain_id=3,       # Arbitrum CCTP domain
        signature_chain_id="0xa4b1",  # Arbitrum EVM chain ID
    )

To bridge USDC from HyperCore to HyperEVM use
:meth:`~pydefi.hyperliquid.client.HyperliquidClient.send_asset` with the USDC
system address as destination::

    # Bridge USDC from HyperCore spot to HyperEVM wallet
    await client.send_asset(
        private_key=pk,
        destination="0x2000000000000000000000000000000000000000",
        token="USDC",
        amount="10",
        source_dex="spot",
        destination_dex="spot",
    )
"""

from pydefi.hyperliquid.client import HyperliquidClient
from pydefi.hyperliquid.signing import (
    APPROVE_AGENT_SIGN_TYPES,
    APPROVE_BUILDER_FEE_SIGN_TYPES,
    SEND_ASSET_SIGN_TYPES,
    SEND_TO_EVM_WITH_DATA_SIGN_TYPES,
    SPOT_TRANSFER_SIGN_TYPES,
    USD_CLASS_TRANSFER_SIGN_TYPES,
    USD_SEND_SIGN_TYPES,
    WITHDRAW_SIGN_TYPES,
    action_hash,
    sign_approve_agent_action,
    sign_approve_builder_fee_action,
    sign_inner,
    sign_l1_action,
    sign_send_asset_action,
    sign_send_to_evm_with_data_action,
    sign_spot_transfer_action,
    sign_usd_class_transfer_action,
    sign_usd_transfer_action,
    sign_user_signed_action,
    sign_withdraw_action,
)

__all__ = [
    # Client
    "HyperliquidClient",
    # Signing helpers
    "action_hash",
    "sign_inner",
    "sign_l1_action",
    "sign_user_signed_action",
    "sign_usd_transfer_action",
    "sign_spot_transfer_action",
    "sign_withdraw_action",
    "sign_usd_class_transfer_action",
    "sign_send_asset_action",
    "sign_send_to_evm_with_data_action",
    "sign_approve_agent_action",
    "sign_approve_builder_fee_action",
    # EIP-712 type definitions
    "USD_SEND_SIGN_TYPES",
    "SPOT_TRANSFER_SIGN_TYPES",
    "WITHDRAW_SIGN_TYPES",
    "USD_CLASS_TRANSFER_SIGN_TYPES",
    "SEND_ASSET_SIGN_TYPES",
    "SEND_TO_EVM_WITH_DATA_SIGN_TYPES",
    "APPROVE_AGENT_SIGN_TYPES",
    "APPROVE_BUILDER_FEE_SIGN_TYPES",
]
