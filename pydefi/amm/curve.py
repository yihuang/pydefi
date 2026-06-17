"""
Curve Finance pool integration (stableswap + cryptoswap / Curve V2).

Swaps can be priced two ways: **on-chain** via the pool's ``get_dy`` view (the
default), or **locally** in Python after :meth:`CurvePool.load_state` snapshots
the pool — zero RPC per quote, for the pathfinder and simulation.  The pool
flavour (:class:`CurvePoolKind`) selects the local-pricing math in
:mod:`pydefi.amm.curve_math`.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from eth_contract import Contract
from eth_contract.erc20 import ERC20
from web3 import AsyncWeb3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError

from pydefi._math import apply_slippage
from pydefi.abi.amm import CURVE_METAREGISTRY, CURVE_POOL, CURVE_REGISTRY, CURVE_V2_POOL
from pydefi.amm import curve_math
from pydefi.exceptions import InsufficientLiquidityError, PoolDataError
from pydefi.types import Address, SwapRoute, SwapStep, Token, TokenAmount

#: Curve MetaRegistry on Ethereum mainnet — default for :meth:`CurvePool.discover`.
CURVE_METAREGISTRY_ETHEREUM = "0xF98B45FA17DE75FB1aD0e7aFD971b0ca00e379fC"


class CurvePoolKind(str, Enum):
    """The Curve pool flavour, selecting which local-pricing math applies."""

    #: Original plain stableswap pool (e.g. 3pool); no ``A_PRECISION``.
    STABLE_LEGACY = "stable_legacy"
    #: Factory plain stableswap pool (``A_PRECISION`` = 100, flat fee).
    STABLE_FACTORY = "stable_factory"
    #: Stable-NG pool (``A_PRECISION`` = 100, dynamic off-peg fee, stored rates).
    STABLE_NG = "stable_ng"
    #: Cryptoswap / Curve V2 pool (Twocrypto / Tricrypto).
    CRYPTO_V2 = "crypto_v2"

    @property
    def is_crypto(self) -> bool:
        """``True`` for cryptoswap (Curve V2) pools."""
        return self is CurvePoolKind.CRYPTO_V2


class CurvePool:
    """Curve Finance pool integration.

    Each instance is bound to a *single* Curve pool.  Use
    :class:`CurveRegistry` (or query the registry on-chain) to discover the
    correct pool for a given token pair.

    Args:
        w3: :class:`~web3.AsyncWeb3` instance.
        pool_address: Address of the Curve pool contract.
        tokens: Ordered list of tokens in the pool (matching the pool's
            ``coins`` array).
        kind: The pool flavour (see :class:`CurvePoolKind`).  Selects the local
            pricing math; defaults to ``STABLE_LEGACY``.
        use_underlying: If ``True`` use the ``_underlying`` variants of
            ``get_dy`` / ``exchange`` (for lending pools like Compound).  Only
            affects on-chain pricing; local pricing operates on wrapped coins.
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        pool_address: str,
        tokens: list[Token],
        kind: CurvePoolKind = CurvePoolKind.STABLE_LEGACY,
        use_underlying: bool = False,
    ) -> None:
        self.w3 = w3
        self.router_address = pool_address
        self._tokens = tokens
        self._kind = kind
        self._use_underlying = use_underlying
        #: Local pool-state snapshot, populated by :meth:`load_state`.
        self._state: dict | None = None

    @property
    def protocol_name(self) -> str:
        return "CurveV2" if self._kind.is_crypto else "Curve"

    @property
    def kind(self) -> CurvePoolKind:
        return self._kind

    @property
    def tokens(self) -> list[Token]:
        """Tokens registered in this pool (ordered by pool index)."""
        return list(self._tokens)

    @property
    def has_local_state(self) -> bool:
        """``True`` once :meth:`load_state` has snapshotted the pool."""
        return self._state is not None

    def _coin_index(self, token: Token) -> int:
        """Return the pool index for *token*."""
        for i, t in enumerate(self._tokens):
            if t.address == token.address:
                return i
        raise ValueError(f"Token {token.symbol} not found in this Curve pool")

    @property
    def _contract(self) -> Contract:
        """The ABI bound to this pool's swap-type (crypto uses uint256 indices)."""
        return CURVE_V2_POOL if self._kind.is_crypto else CURVE_POOL

    async def _call(self, fn):
        """Call a bound contract function on this pool, returning its result."""
        return await fn.call(self.w3, to=self.router_address)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    async def detect_kind(w3: AsyncWeb3, pool_address: str) -> CurvePoolKind:
        """Probe *pool_address* to determine its :class:`CurvePoolKind`.

        Distinguishing views: ``gamma()`` exists only on cryptoswap pools,
        ``offpeg_fee_multiplier()`` only on Stable-NG, and ``A_precise()`` only
        on the newer (factory) stableswap pools.

        Only a contract-level miss (revert / undecodable output) counts as "the
        view doesn't exist" — network errors propagate rather than silently
        misclassifying the pool.
        """

        async def has(fn) -> bool:
            try:
                await fn.call(w3, to=pool_address)
                return True
            except (ContractLogicError, BadFunctionCallOutput):
                return False

        if await has(CURVE_V2_POOL.fns.gamma()):
            return CurvePoolKind.CRYPTO_V2
        if await has(CURVE_POOL.fns.offpeg_fee_multiplier()):
            return CurvePoolKind.STABLE_NG
        if await has(CURVE_POOL.fns.A_precise()):
            return CurvePoolKind.STABLE_FACTORY
        return CurvePoolKind.STABLE_LEGACY

    @staticmethod
    async def find_pools(
        w3: AsyncWeb3,
        token_a: Token,
        token_b: Token,
        registry_address: str = CURVE_METAREGISTRY_ETHEREUM,
    ) -> list[str]:
        """Return Curve pool addresses trading *token_a* ↔ *token_b*.

        Queries the Curve MetaRegistry (pass *registry_address* for non-mainnet
        chains).  Ordered by the registry; zero addresses are dropped.
        """
        pools = await CURVE_METAREGISTRY.fns.find_pools_for_coins(token_a.address, token_b.address).call(
            w3, to=registry_address
        )
        # The MetaRegistry returns lowercase addresses; checksum them for web3.
        return [w3.to_checksum_address(p) for p in pools if int(p, 16) != 0]

    @classmethod
    async def discover(
        cls,
        w3: AsyncWeb3,
        token_a: Token,
        token_b: Token,
        *,
        registry_address: str = CURVE_METAREGISTRY_ETHEREUM,
        index: int = 0,
        load: bool = True,
    ) -> CurvePool | None:
        """Find and build a :class:`CurvePool` for the *token_a* ↔ *token_b* pair.

        Looks the pool up via the MetaRegistry, auto-detects its
        :class:`CurvePoolKind`, and reads its coin list (decimals/symbols from
        each ERC-20, reusing *token_a* / *token_b* where they match).  Multiple
        pools may exist for a pair; *index* selects which (registry order).

        Args:
            w3: Connected web3 instance.
            token_a, token_b: The pair to route between.
            registry_address: MetaRegistry address (mainnet default).
            index: Which matching pool to use when several exist.
            load: If ``True`` (default), also snapshot state via
                :meth:`load_state` so the pool prices locally immediately.

        Returns:
            A ready :class:`CurvePool`, or ``None`` if no pool trades the pair.
        """
        pools = await cls.find_pools(w3, token_a, token_b, registry_address)
        if index >= len(pools):
            return None
        pool_address = pools[index]
        kind = await cls.detect_kind(w3, pool_address)
        tokens = await cls._discover_tokens(w3, pool_address, [token_a, token_b])
        pool = cls(w3, pool_address, tokens, kind=kind)
        if load:
            await pool.load_state()
        return pool

    @staticmethod
    async def _discover_tokens(w3: AsyncWeb3, pool_address: str, known: list[Token]) -> list[Token]:
        """Read a pool's ``coins`` and build :class:`Token`s, reusing *known*."""
        known_by_addr = {t.address: t for t in known}
        chain_id = known[0].chain_id
        tokens: list[Token] = []
        for i in range(8):
            try:
                addr = Address(await CURVE_POOL.fns.coins(i).call(w3, to=pool_address))
            except (ContractLogicError, BadFunctionCallOutput):
                break  # past the last coin
            if addr in known_by_addr:
                tokens.append(known_by_addr[addr])
                continue
            try:
                decimals = await ERC20.fns.decimals().call(w3, to=addr)
            except (ContractLogicError, BadFunctionCallOutput):
                decimals = 18  # native ETH placeholder (0xEeee…) has no decimals()
            try:
                symbol = await ERC20.fns.symbol().call(w3, to=addr)
            except (ContractLogicError, BadFunctionCallOutput):
                symbol = ""
            tokens.append(Token(chain_id=chain_id, address=addr, symbol=symbol, decimals=decimals))
        return tokens

    # ------------------------------------------------------------------
    # Local state snapshot
    # ------------------------------------------------------------------

    async def load_state(self) -> dict:
        """Snapshot the pool's on-chain state for local pricing.

        After this call, :meth:`get_dy` (and the higher-level helpers) compute
        prices in Python with no further RPC.  Re-call to refresh the snapshot.

        Returns:
            The captured state dict (also stored on the instance).

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: If the pool does not
                expose the views required for its :class:`CurvePoolKind`.
        """
        try:
            if self._kind.is_crypto:
                self._state = await self._load_crypto_state()
            else:
                self._state = await self._load_stable_state()
        except Exception as exc:  # pragma: no cover - network/abi failures
            raise PoolDataError(f"Failed to load Curve pool state: {exc}") from exc
        return self._state

    async def _load_stable_state(self) -> dict:
        c = CURVE_POOL
        balances = [await self._call(c.fns.balances(i)) for i in range(len(self._tokens))]
        decimal_rates = [10 ** (36 - t.decimals) for t in self._tokens]

        if self._kind is CurvePoolKind.STABLE_LEGACY:
            # Original plain pools predate A_PRECISION and take the fee last.
            return {
                "balances": balances,
                "rates": decimal_rates,
                "amp": await self._call(c.fns.A()),
                "fee": await self._call(c.fns.fee()),
                "a_precision": 1,
                "ng_d_form": False,
                "offpeg_fee_multiplier": 0,
                "legacy_fee_order": True,
            }

        is_ng = self._kind is CurvePoolKind.STABLE_NG
        return {
            "balances": balances,
            # Stable-NG folds oracle rates into stored_rates; plain factory pools
            # only normalise decimals.
            "rates": list(await self._call(c.fns.stored_rates())) if is_ng else decimal_rates,
            "amp": await self._call(c.fns.A_precise()),
            "fee": await self._call(c.fns.fee()),
            "a_precision": curve_math.A_PRECISION,
            "ng_d_form": is_ng,
            "offpeg_fee_multiplier": await self._call(c.fns.offpeg_fee_multiplier()) if is_ng else 0,
            "legacy_fee_order": False,
        }

    async def _load_crypto_state(self) -> dict:
        c = CURVE_V2_POOL
        n = len(self._tokens)
        balances = [await self._call(c.fns.balances(i)) for i in range(n)]

        # price_scale[0] is the unit coin (PRECISION).  Two-coin pools expose a
        # scalar price_scale(); pools with 3+ coins expose price_scale(uint256).
        scales = (
            [await self._call(c.fns.price_scale())]
            if n == 2
            else [await self._call(c.fns.price_scale(k)) for k in range(n - 1)]
        )

        return {
            "balances": balances,
            "precisions": [10 ** (18 - t.decimals) for t in self._tokens],
            "price_scale": [curve_math.PRECISION, *scales],
            "amp": await self._call(c.fns.A()),
            "gamma": await self._call(c.fns.gamma()),
            "d": await self._call(c.fns.D()),
            "mid_fee": await self._call(c.fns.mid_fee()),
            "out_fee": await self._call(c.fns.out_fee()),
            "fee_gamma": await self._call(c.fns.fee_gamma()),
        }

    # ------------------------------------------------------------------
    # Price queries
    # ------------------------------------------------------------------

    def get_dy_local(self, token_in: Token, token_out: Token, amount_in: int) -> int:
        """Compute ``get_dy`` purely in Python from the loaded state snapshot.

        Args:
            token_in: Input token.
            token_out: Output token.
            amount_in: Raw input amount.

        Returns:
            Expected raw output amount.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: If :meth:`load_state` has
                not been called.
        """
        if self._state is None:
            raise PoolDataError("Pool state not loaded; call load_state() first")

        # The state dict keys mirror the math-function parameters (see the
        # _load_*_state methods), so it unpacks straight into the call.
        i = self._coin_index(token_in)
        j = self._coin_index(token_out)
        get_dy = curve_math.crypto_get_dy if self._kind.is_crypto else curve_math.stable_get_dy
        return get_dy(i, j, amount_in, **self._state)

    def get_dx_local(self, token_in: Token, token_out: Token, amount_out: int) -> int:
        """Exact-output inverse of :meth:`get_dy_local` from loaded state.

        Returns the raw input amount of *token_in* needed to receive
        *amount_out* of *token_out*.  Stableswap pools use the closed-form
        :func:`~pydefi.amm.curve_math.stable_get_dx`; cryptoswap pools (which
        have no closed form) binary-search the local :meth:`get_dy_local`.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: If :meth:`load_state` has
                not been called.
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: If the pool
                cannot produce *amount_out* at any input size (cryptoswap only;
                stableswap returns ``0`` for infeasible targets).
        """
        if self._state is None:
            raise PoolDataError("Pool state not loaded; call load_state() first")

        if not self._kind.is_crypto:
            i = self._coin_index(token_in)
            j = self._coin_index(token_out)
            return curve_math.stable_get_dx(i, j, amount_out, **self._state)

        # Cryptoswap: get_dy is monotone in dx, so binary-search for the input.
        lo, hi = 0, 1
        while True:
            try:
                if self.get_dy_local(token_in, token_out, hi) >= amount_out:
                    break
            except (curve_math.CurveConvergenceError, ZeroDivisionError) as exc:
                # The probe input blew past what the invariant can represent.
                raise InsufficientLiquidityError(f"pool cannot produce {amount_out} of {token_out.symbol}") from exc
            hi *= 2
            if hi > 2**256:
                raise InsufficientLiquidityError(f"pool cannot produce {amount_out} of {token_out.symbol}")
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.get_dy_local(token_in, token_out, mid) >= amount_out:
                hi = mid
            else:
                lo = mid
        return hi

    async def get_dy(self, token_in: Token, token_out: Token, amount_in: int) -> int:
        """Simulate a swap, preferring local math when state is loaded.

        When :meth:`load_state` has been called (and ``use_underlying`` is
        ``False``), this computes the result locally with no RPC.  Otherwise it
        falls back to the pool's on-chain ``get_dy`` view.

        Args:
            token_in: Input token.
            token_out: Output token.
            amount_in: Raw input amount.

        Returns:
            Expected raw output amount.

        Raises:
            :class:`~pydefi.exceptions.InsufficientLiquidityError`: On revert.
        """
        if self._state is not None and not self._use_underlying:
            return self.get_dy_local(token_in, token_out, amount_in)

        i = self._coin_index(token_in)
        j = self._coin_index(token_out)
        fn_name = "get_dy_underlying" if self._use_underlying else "get_dy"
        try:
            amount_out: int = await getattr(self._contract.fns, fn_name)(i, j, amount_in).call(
                self.w3, to=self.router_address
            )
        except Exception as exc:
            raise InsufficientLiquidityError(f"get_dy failed: {exc}") from exc
        return amount_out

    async def get_amounts_out(self, amount_in: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Simulate a swap along *path* through this pool.

        Curve pools typically contain 2-4 tokens.  Multi-hop routes require
        chaining calls across multiple pools.

        Args:
            amount_in: Input amount.
            path: Must be exactly two tokens from this pool.

        Returns:
            ``[amount_in, amount_out]``
        """
        if len(path) != 2:
            raise ValueError("CurvePool.get_amounts_out requires a 2-token path")
        amount_out_raw = await self.get_dy(path[0], path[1], amount_in.amount)
        return [amount_in, TokenAmount(token=path[1], amount=amount_out_raw)]

    async def get_amounts_in(self, amount_out: TokenAmount, path: list[Token]) -> list[TokenAmount]:
        """Estimate required input to obtain *amount_out* (approximation).

        With local state loaded this is exact and RPC-free (closed-form
        :meth:`get_dx_local` for stableswap, a local search for cryptoswap).
        Without state it falls back to a binary search over the on-chain
        ``get_dy`` view, since most pools predate the on-chain ``get_dx``.

        Args:
            amount_out: Desired output.
            path: Must be exactly two tokens.

        Returns:
            ``[amount_in, amount_out]``
        """
        if len(path) != 2:
            raise ValueError("CurvePool.get_amounts_in requires a 2-token path")

        if self._state is not None and not self._use_underlying:
            amount_in = self.get_dx_local(path[0], path[1], amount_out.amount)
            return [TokenAmount(token=path[0], amount=amount_in), amount_out]

        target = amount_out.amount
        # Start with a 1:1 estimate and double until we exceed the target
        lo, hi = 1, target * 2
        for _ in range(64):
            mid = (lo + hi) // 2
            dy = await self.get_dy(path[0], path[1], mid)
            if dy >= target:
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1:
                break

        return [TokenAmount(token=path[0], amount=hi), amount_out]

    # ------------------------------------------------------------------
    # Route builder
    # ------------------------------------------------------------------

    async def build_swap_route(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        slippage_bps: int = 50,
    ) -> SwapRoute:
        """Build a direct swap route through this Curve pool.

        Args:
            amount_in: Exact input amount.
            token_out: Desired output token (must be in this pool).
            slippage_bps: Slippage tolerance in basis points.

        Returns:
            A :class:`~pydefi.types.SwapRoute`.
        """
        amount_out_raw = await self.get_dy(amount_in.token, token_out, amount_in.amount)
        amount_out = TokenAmount(token=token_out, amount=amount_out_raw)

        # SwapStep.fee is in basis points; use the pool's real fee when state
        # is loaded (mid_fee at parity for cryptoswap), else Curve's typical 4.
        fee_bps = 4
        if self._state is not None:
            raw_fee = self._state["mid_fee" if self._kind.is_crypto else "fee"]
            fee_bps = raw_fee * 10_000 // curve_math.FEE_DENOMINATOR

        step = SwapStep(
            token_in=amount_in.token,
            token_out=token_out,
            pool_address=self.router_address,
            protocol=self.protocol_name,
            fee=fee_bps,
        )

        return SwapRoute(
            steps=[step],
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=Decimal(0),
        )

    def build_exchange_tx(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        *,
        min_amount_out: int | None = None,
        slippage_bps: int = 50,
        underlying: bool = False,
    ) -> dict:
        """Encode a direct ``exchange`` (or ``exchange_underlying``) call.

        Returns the project-wide ``{to, data, value}`` tx dict for the pool's
        own ``exchange`` — no router involved.  The caller must have approved
        the pool to spend ``amount_in`` first.

        Args:
            amount_in: Exact input token + amount.
            token_out: Desired output token (must be in this pool).
            min_amount_out: Minimum acceptable output.  If ``None``, it is
                derived from the local quote and *slippage_bps* (requires
                :meth:`load_state`).
            slippage_bps: Slippage tolerance used only when *min_amount_out* is
                ``None``.
            underlying: Use ``exchange_underlying`` (for lending base coins).
        """
        i = self._coin_index(amount_in.token)
        j = self._coin_index(token_out)
        if min_amount_out is None:
            expected = self.get_dy_local(amount_in.token, token_out, amount_in.amount)
            min_amount_out = apply_slippage(expected, slippage_bps)
        fn_name = "exchange_underlying" if underlying else "exchange"
        data = getattr(self._contract.fns, fn_name)(i, j, amount_in.amount, min_amount_out).data
        return {"to": self.router_address, "data": "0x" + data.hex(), "value": "0"}

    def get_registry_contract(self, registry_address: str) -> Contract:
        """Return a :class:`~eth_contract.Contract` bound to a Curve registry."""
        return CURVE_REGISTRY(to=registry_address)


class CurveMetaPool:
    """A Curve stableswap **meta-pool** (a coin paired with a base pool's LP token).

    Prices *underlying* swaps — between the meta-pool's primary coin and any coin
    of the base pool — locally via :func:`pydefi.amm.curve_math.meta_get_dy_underlying`.
    Underlying index ``0`` is the primary coin; ``1..`` are the base pool's coins.

    Note: Stable-NG meta-pools are priced with the flat base fee — the NG
    off-peg dynamic fee is not applied to underlying quotes.  Factory metas
    (the default) are exact.

    Args:
        w3: Connected web3 instance.
        pool_address: The meta-pool contract address.
        primary_token: The meta-pool's coin 0 (the non-LP coin).
        base_pool: A :class:`CurvePool` for the base pool (e.g. 3pool).  Its LP
            token must be the base pool contract itself (the usual Curve case).
        kind: The meta-pool's stableswap flavour (factory by default).
    """

    def __init__(
        self,
        w3: AsyncWeb3,
        pool_address: str,
        primary_token: Token,
        base_pool: CurvePool,
        kind: CurvePoolKind = CurvePoolKind.STABLE_FACTORY,
    ) -> None:
        self.w3 = w3
        self.pool_address = pool_address
        self.primary_token = primary_token
        self.base_pool = base_pool
        self._kind = kind
        #: Underlying tokens: primary coin followed by the base pool's coins.
        self.tokens = [primary_token, *base_pool.tokens]
        self._meta: dict | None = None
        self._base: dict | None = None

    @property
    def protocol_name(self) -> str:
        return "Curve"

    def _underlying_index(self, token: Token) -> int:
        for idx, t in enumerate(self.tokens):
            if t.address == token.address:
                return idx
        raise ValueError(f"Token {token.symbol} is not an underlying coin of this meta-pool")

    async def load_state(self) -> None:
        """Snapshot the meta-pool and its base pool for local underlying pricing."""
        await self.base_pool.load_state()
        base = dict(self.base_pool._state)
        # The meta-pool's coin 1 is the base LP token; read its supply there
        # (the base pool and its LP token are often separate contracts).
        base_lp = self.w3.to_checksum_address(await CURVE_POOL.fns.coins(1).call(self.w3, to=self.pool_address))
        base["total_supply"] = await ERC20.fns.totalSupply().call(self.w3, to=base_lp)
        self._base = base

        c = CURVE_POOL
        is_ng = self._kind is CurvePoolKind.STABLE_NG
        if self._kind is CurvePoolKind.STABLE_LEGACY:
            # Legacy meta-pools may not expose A_precise at all.
            amp = await c.fns.A().call(self.w3, to=self.pool_address)
            a_precision = 1
        else:
            amp = await c.fns.A_precise().call(self.w3, to=self.pool_address)
            a_precision = curve_math.A_PRECISION
        if is_ng:
            # NG metas expose stored_rates, which already folds the base pool's
            # virtual price into rates[1] (and any oracle rate for coin 0).
            rates = list(await c.fns.stored_rates().call(self.w3, to=self.pool_address))
            offpeg = await c.fns.offpeg_fee_multiplier().call(self.w3, to=self.pool_address)
        else:
            virtual_price = await c.fns.get_virtual_price().call(self.w3, to=self.base_pool.router_address)
            rates = [10 ** (36 - self.primary_token.decimals), virtual_price]
            offpeg = 0
        self._meta = {
            "balances": [await c.fns.balances(k).call(self.w3, to=self.pool_address) for k in range(2)],
            # rates[1] is the base LP token's price (its virtual price).
            "rates": rates,
            "amp": amp,
            "fee": await c.fns.fee().call(self.w3, to=self.pool_address),
            "a_precision": a_precision,
            "ng_d_form": is_ng,
            "offpeg_fee_multiplier": offpeg,
        }

    def get_dy_underlying(self, token_in: Token, token_out: Token, amount_in: int) -> int:
        """Local underlying quote between the primary coin and a base-pool coin.

        Raises:
            :class:`~pydefi.exceptions.PoolDataError`: If state is not loaded.
        """
        if self._meta is None or self._base is None:
            raise PoolDataError("Meta-pool state not loaded; call load_state() first")
        i = self._underlying_index(token_in)
        j = self._underlying_index(token_out)
        return curve_math.meta_get_dy_underlying(i, j, amount_in, self._meta, self._base)

    def build_exchange_underlying_tx(
        self,
        amount_in: TokenAmount,
        token_out: Token,
        *,
        min_amount_out: int | None = None,
        slippage_bps: int = 50,
    ) -> dict:
        """Encode an ``exchange_underlying`` call between two underlying coins.

        Returns the ``{to, data, value}`` tx dict.  If *min_amount_out* is
        ``None`` it is derived from :meth:`get_dy_underlying` and *slippage_bps*.
        """
        i = self._underlying_index(amount_in.token)
        j = self._underlying_index(token_out)
        if min_amount_out is None:
            expected = self.get_dy_underlying(amount_in.token, token_out, amount_in.amount)
            min_amount_out = apply_slippage(expected, slippage_bps)
        data = CURVE_POOL.fns.exchange_underlying(i, j, amount_in.amount, min_amount_out).data
        return {"to": self.pool_address, "data": "0x" + data.hex(), "value": "0"}
