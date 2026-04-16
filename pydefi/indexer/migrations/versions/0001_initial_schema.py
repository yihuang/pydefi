"""Initial schema: Factory, Pool, V2SyncEvent, V3SwapEvent, IndexerState tables.

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "factory",
        sa.Column("factory_address", sa.String(), nullable=False),
        sa.Column("protocol", sa.String(), nullable=False),
        sa.Column("chain_id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("factory_address"),
    )
    op.create_table(
        "pool",
        sa.Column("pool_address", sa.String(), nullable=False),
        sa.Column("protocol", sa.String(), nullable=False),
        sa.Column("chain_id", sa.Integer(), nullable=False),
        sa.Column("token0_address", sa.String(), nullable=False),
        sa.Column("token0_symbol", sa.String(), nullable=False),
        sa.Column("token0_decimals", sa.Integer(), nullable=False),
        sa.Column("token1_address", sa.String(), nullable=False),
        sa.Column("token1_symbol", sa.String(), nullable=False),
        sa.Column("token1_decimals", sa.Integer(), nullable=False),
        sa.Column("fee_bps", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("pool_address"),
    )
    op.create_table(
        "v2syncevent",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pool_address", sa.String(), nullable=False),
        sa.Column("block_number", sa.Integer(), nullable=False),
        sa.Column("block_hash", sa.String(), nullable=False),
        sa.Column("tx_hash", sa.String(), nullable=False),
        sa.Column("log_index", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Integer(), nullable=False),
        sa.Column("reserve0", sa.Text(), nullable=False),
        sa.Column("reserve1", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_v2syncevent_block_number"), "v2syncevent", ["block_number"], unique=False)
    op.create_index(op.f("ix_v2syncevent_pool_address"), "v2syncevent", ["pool_address"], unique=False)
    op.create_index("uq_v2syncevent_tx_log", "v2syncevent", ["tx_hash", "log_index"], unique=True)
    op.create_table(
        "v3swapevent",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pool_address", sa.String(), nullable=False),
        sa.Column("block_number", sa.Integer(), nullable=False),
        sa.Column("block_hash", sa.String(), nullable=False),
        sa.Column("tx_hash", sa.String(), nullable=False),
        sa.Column("log_index", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Integer(), nullable=False),
        sa.Column("sqrt_price_x96", sa.Text(), nullable=False),
        sa.Column("liquidity", sa.Text(), nullable=False),
        sa.Column("tick", sa.Text(), nullable=False),
        sa.Column("amount0", sa.Text(), nullable=False),
        sa.Column("amount1", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_v3swapevent_block_number"), "v3swapevent", ["block_number"], unique=False)
    op.create_index(op.f("ix_v3swapevent_pool_address"), "v3swapevent", ["pool_address"], unique=False)
    op.create_index("uq_v3swapevent_tx_log", "v3swapevent", ["tx_hash", "log_index"], unique=True)
    op.create_table(
        "indexerstate",
        sa.Column("address", sa.String(), nullable=False),
        sa.Column("last_indexed_block", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("address"),
    )


def downgrade() -> None:
    op.drop_table("indexerstate")
    op.drop_index("uq_v3swapevent_tx_log", table_name="v3swapevent")
    op.drop_index(op.f("ix_v3swapevent_pool_address"), table_name="v3swapevent")
    op.drop_index(op.f("ix_v3swapevent_block_number"), table_name="v3swapevent")
    op.drop_table("v3swapevent")
    op.drop_index("uq_v2syncevent_tx_log", table_name="v2syncevent")
    op.drop_index(op.f("ix_v2syncevent_pool_address"), table_name="v2syncevent")
    op.drop_index(op.f("ix_v2syncevent_block_number"), table_name="v2syncevent")
    op.drop_table("v2syncevent")
    op.drop_table("pool")
    op.drop_table("factory")
