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

# Custom codec: decoded addresses are Address (HexBytes) instead of checksum strings.
# Pass this codec explicitly when decoding ABI data (e.g. codec.decode([...], data)).
codec = ABICodec(registry)
