import eth_contract.contract
from eth_abi.codec import ABICodec
from eth_abi.decoding import AddressDecoder
from eth_abi.registry import registry as default_registry

from ..types import Address


class AddressBytesDecoder(AddressDecoder):
    @staticmethod
    def decoder_fn(data):
        return Address(data)

registry = default_registry.copy()
registry.unregister_decoder("address")
registry.register_decoder("address", AddressBytesDecoder)

codec = ABICodec(registry)

# automagically patch eth_contract
eth_contract.contract._abi_codec = codec
