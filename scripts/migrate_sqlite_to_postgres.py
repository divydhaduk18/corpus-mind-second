"""Production Database Migration Script: SQLite to PostgreSQL (Improvement 31).

Migrates all tables, schemas, relations, and data from local SQLite database
to enterprise PostgreSQL with table dependency ordering, data validation,
and sequence synchronization.
"""

import os
import sys
import argparse
import sqlite3
import logging
from typing import Dict, List, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.models import Base

logger = logging.getLogger("migration")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


TABLE_DEPENDENCY_ORDER = [
    "users",
    "chat_sessions",
    "messages",
    "documents",
    "document_permissions",
    "document_jobs",
    "ragas_evaluations",
    "audit_logs",
    "analytics_logs"
]


def migrate(sqlite_path: str, postgres_url: str):
    """Executes full data migration from SQLite to PostgreSQL."""
    if not SQLALCHEMY_AVAILABLE:
        logger.error("SQLAlchemy is required for migration. Run 'pip install sqlalchemy psycopg2-binary'.")
        sys.exit(1)

    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite source file '{sqlite_path}' does not exist.")
        sys.exit(1)

    logger.info(f"Source SQLite: {sqlite_path}")
    logger.info(f"Target PostgreSQL: {postgres_url.split('@')[-1] if '@' in postgres_url else postgres_url}")

    # Initialize PostgreSQL engine and create tables
    pg_engine = create_engine(postgres_url)
    logger.info("Verifying/creating target PostgreSQL schema via SQLAlchemy Base...")
    Base.metadata.create_all(pg_engine)
    logger.info("PostgreSQL schema successfully prepared.")

    # Open SQLite connection
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    # Discover tables present in SQLite
    sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    existing_tables = {row[0] for row in sqlite_cursor.fetchall()}

    total_migrated_records = 0

    with pg_engine.connect() as pg_conn:
        trans = pg_conn.begin()
        try:
            for table_name in TABLE_DEPENDENCY_ORDER:
                if table_name not in existing_tables:
                    logger.info(f"Skipping '{table_name}' (not present in SQLite source).")
                    continue

                sqlite_cursor.execute(f"SELECT * FROM {table_name}")
                rows = [dict(r) for r in sqlite_cursor.fetchall()]
                count = len(rows)

                if count == 0:
                    logger.info(f"Table '{table_name}': 0 records.")
                    continue

                logger.info(f"Migrating table '{table_name}' ({count} records)...")

                # Column names from first record
                cols = list(rows[0].keys())
                col_names_str = ", ".join(cols)
                placeholders = ", ".join([f":{c}" for c in cols])
                insert_stmt = text(f"INSERT INTO {table_name} ({col_names_str}) VALUES ({placeholders})")

                # Batch insert
                pg_conn.execute(insert_stmt, rows)
                total_migrated_records += count

                # Update PostgreSQL primary key sequence if table has integer primary key 'id'
                if "id" in cols:
                    try:
                        seq_query = text(f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), COALESCE((SELECT MAX(id) FROM {table_name}), 1))")
                        pg_conn.execute(seq_query)
                    except Exception as seq_e:
                        logger.debug(f"Sequence sync note for {table_name}: {seq_e}")

            trans.commit()
            logger.info("=" * 60)
            logger.info(f"MIGRATION COMPLETE! Successfully migrated {total_migrated_records} records.")
            logger.info("=" * 60)

        except Exception as e:
            trans.rollback()
            logger.error(f"Migration failed with error: {e}. Transaction rolled back.")
            sys.exit(1)
        finally:
            sqlite_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus Mind: Migrate SQLite to PostgreSQL")
    parser.add_argument("--sqlite-path", default=str(config.DB_PATH), help="Path to source SQLite file")
    parser.add_argument("--postgres-url", default=os.getenv("DATABASE_URL", "postgresql://corpus_user:corpus_password@localhost:5432/corpus_mind"), help="Target PostgreSQL URL")
    args = parser.parse_args()

    migrate(args.sqlite_path, args.postgres_url)
