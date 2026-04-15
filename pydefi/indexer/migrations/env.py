"""Alembic environment configuration for pydefi indexer.

The database URL is resolved in the following order:

1. ``-x db_url=<url>`` CLI argument
2. ``PYDEFI_DB_URL`` environment variable
3. ``sqlalchemy.url`` in ``alembic.ini``
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# Make sure all models are imported so that SQLModel.metadata is populated.
import pydefi.indexer.models  # noqa: F401

# Alembic Config object (gives access to values in alembic.ini).
config = context.config

# Override DB URL from Alembic CLI (-x db_url=...) or environment variable if set.
x_args = context.get_x_argument(as_dictionary=True)
db_url = x_args.get("db_url") or os.environ.get("PYDEFI_DB_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# Interpret the alembic.ini file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLModel / SQLAlchemy metadata.
target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine; calls to
    ``context.execute()`` emit the given string to the output directly.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with a real database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
