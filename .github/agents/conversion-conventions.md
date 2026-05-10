# Conversion Conventions

This document describes the canonical patterns for converting between hex
strings and bytes for EVM addresses and 32-byte hashes/topics throughout the
`pydefi` codebase.

---

## 1. Type Aliases

Two type aliases are defined in `pydefi.types` and re-exported from `pydefi`:

| Alias | Underlying type | Represents |
|-------|-----------------|------------|
| `Address` | `HexBytes` | 20-byte EVM address |
| `Hash` | `HexBytes` | 32-byte hash, log topic, or padded value |

Because `HexBytes` is a `bytes` subclass, `Address` and `Hash` values can be
used anywhere `bytes` is expected without extra wrapping.

For other bytes values with variable length (e.g. signatures, call data), use `bytes` directly without an alias.

---

## 2. The "Conversions at Peripheries" Rule

> **Do conversions at peripheries; pass `Address` / `Hash` (HexBytes) internally.**

A *periphery* is any point where an external value (e.g. a string from a JSON
response or user input) first enters the system.  Peripheries include:

- Public API request/response handlers (e.g. `build_bridge_tx`, `get_quote`)
- Test code that constructs addresses from literal strings
- CLI entry points

Convert a string to `HexBytes` **once at the boundary** and use the bytes
value everywhere thereafter.  All library methods (bridge, AMM, VM builders,
utilities) are considered *internal* — they accept and pass `Address`/`Hash`
values without doing any string-to-bytes conversion.

### Converting strings to `Address` / `Hash`

```python
from hexbytes import HexBytes
from pydefi.types import Address, Hash

# At a periphery (public API entry point or test code):
addr: Address = HexBytes("0xAbCd…")     # 0x-prefixed checksummed or lowercase
topic: Hash   = HexBytes("0xdeadbeef…") # 0x-prefixed 32-byte hash
```

`HexBytes` accepts:
- `str` — `0x`-prefixed hex (e.g. `"0xabc123"`) or bare hex (e.g. `"abc123"`)
- `bytes` / `bytearray` — copied as-is
- `int` — treated as a single byte

### Do NOT wrap HexBytes in bytes()

`HexBytes` is already a `bytes` subclass; the extra conversion is redundant:

```python
# ✗ wrong — bytes() wrapper is unnecessary
raw = bytes(HexBytes(token_address))

# ✓ correct — HexBytes IS bytes
raw = HexBytes(token_address)
```

---

## 3. Padding an Address to 32 Bytes

For ABI-encoding or LayerZero / CCTP / Wormhole payloads that need a 32-byte
left-padded address (``bytes32``), use the shared helper from `pydefi._utils`:

```python
from hexbytes import HexBytes
from pydefi._utils import address_to_bytes32
from pydefi.types import Address, Hash

# Convert at periphery, then pass Address internally:
addr: Address = HexBytes(some_address_str)
padded: Hash  = address_to_bytes32(addr)   # 32-byte left-padded result
```

`address_to_bytes32(address: Address) -> Hash` returns a `HexBytes` value of
exactly 32 bytes, with the 20-byte address in the rightmost 20 bytes.

### Wormhole / SWIFT token encoding

For bridge payloads that encode a token address as a ``bytes32`` (with native
tokens mapping to 32 zero bytes), use:

```python
from pydefi._utils import token_to_bytes32

token_b32: Hash = token_to_bytes32(HexBytes(token_address))
```

`token_to_bytes32(address: Address) -> Hash` returns 32 zero bytes for native
token sentinels (zero address or `0xEeEe…` form) and `address_to_bytes32`
padding for all other ERC-20 token addresses.

### Re-use Constants

`pydefi.types` modules has defined constants for the native token sentinels, re-use them instead of hardcoding the values:

```python
from pydefi.types import ZERO_ADDRESS, NATIVE_ADDRESSES

# ✗ wrong — hardcoded values
token = Address("0x0000000000000000000000000000000000000000")
token_address in ("0x0000000000000000000000000000000000000000", "0xEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEe")

# ✓ correct — use constants
token = ZERO_ADDRESS
token_address in NATIVE_ADDRESSES:
```

---

## 4. Summary

| Situation | Preferred pattern |
|-----------|-------------------|
| String address at a periphery | `HexBytes(addr_str)` → `Address` |
| All internal library methods | Accept and pass `Address`/`Hash` directly |
| Address-to-bytes32 padding | `address_to_bytes32(addr)` |
| Native-token-aware bytes32 | `token_to_bytes32(addr)` |
| Comparing two addresses | `HexBytes(a) == HexBytes(b)` (case-insensitive, handles checksum) |
| Converting `Address`/`HexBytes` to 0x-prefixed string | `address.to_0x_hex()` |
