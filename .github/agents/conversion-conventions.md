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

---

## 2. The "Conversions at Peripheries" Rule

> **Do conversions at peripheries; pass `Address` / `Hash` (HexBytes) internally.**

A *periphery* is any point where an external value (e.g. a string from a JSON
response or a user argument) first enters the system.  Convert it to `HexBytes`
once at that boundary and use the bytes value everywhere thereafter.

### Converting strings to `Address` / `Hash`

```python
from hexbytes import HexBytes
from pydefi.types import Address, Hash

# 0x-prefixed checksummed or lowercase hex string → Address
addr: Address = HexBytes("0xAbCd…")

# 0x-prefixed 32-byte hash string → Hash
topic: Hash = HexBytes("0xdeadbeef…")
```

`HexBytes` accepts:
- `str` — must be `0x`-prefixed hex (e.g. `"0xabc123"`)
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

addr: Address = HexBytes(some_address_str)   # convert at periphery
padded: Hash  = address_to_bytes32(addr)     # 32-byte left-padded result
```

`address_to_bytes32(address: Address) -> Hash` returns a `HexBytes` value of
exactly 32 bytes, with the 20-byte address in the rightmost 20 bytes.

---

## 4. DeFiVM `push_addr`

`push_addr` in `pydefi.vm.program` (and the matching `Program.push_addr` method
in `pydefi.vm.builder`) expects an `Address` (i.e. 20-byte `bytes`/`HexBytes`).
Convert strings at the call site:

```python
from hexbytes import HexBytes
from pydefi.vm.program import push_addr

# ✗ wrong — passing a string directly
program = push_addr("0xAbCd…")

# ✓ correct — convert to Address at the periphery
program = push_addr(HexBytes("0xAbCd…"))
```

Higher-level builder methods (`call_contract`, `call_with_patches`) still accept
plain `str` addresses and perform the conversion internally — they are themselves
peripheries that accept user-provided strings.

---

## 5. Summary

| Situation | Preferred pattern |
|-----------|-------------------|
| String address from an API / user input | `HexBytes(addr_str)` → `Address` |
| Address in low-level program builder | `push_addr(HexBytes(addr_str))` |
| Address-to-bytes32 padding | `address_to_bytes32(HexBytes(addr_str))` |
| Comparing two addresses | `HexBytes(a) == HexBytes(b)` (case-insensitive, handles checksum) |
| Converting `Address`/`HexBytes` to 0x-prefixed string | `address.to_0x_hex()` |
