#!/usr/bin/env python3
"""
Load Credit Card Data into Training Transactions Table

This script loads the creditcard.csv dataset into the PostgreSQL training_transactions table.
"""

import os
import sys
from datetime import datetime

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text


def main():
    print("🚀 Starting credit card data loading process...")

    # Database connection parameters
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "fraud_detection")
    DB_USER = os.getenv("POSTGRES_USER", "fraud_user")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud_pass_dev_2024")

    # Create database connection string
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")

    # Create SQLAlchemy engine
    engine = create_engine(db_url)

    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

    # Load creditcard.csv file
    csv_path = "creditcard.csv"

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded {len(df):,} rows from {csv_path}")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("Please ensure creditcard.csv is in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)

    # Check if training_transactions table exists
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'training_transactions'
            );
        """
            )
        )
        table_exists = result.fetchone()[0]

        if not table_exists:
            print("❌ Table 'training_transactions' does not exist!")
            print("Please run the schema.sql first to create the database tables.")
            sys.exit(1)

        # Check current count
        result = conn.execute(text("SELECT COUNT(*) FROM training_transactions"))
        initial_count = result.fetchone()[0]
        print(f"Table currently has {initial_count:,} rows")

    # Prepare data for insertion
    column_mapping = {
        "Time": "time",
        "V1": "v1",
        "V2": "v2",
        "V3": "v3",
        "V4": "v4",
        "V5": "v5",
        "V6": "v6",
        "V7": "v7",
        "V8": "v8",
        "V9": "v9",
        "V10": "v10",
        "V11": "v11",
        "V12": "v12",
        "V13": "v13",
        "V14": "v14",
        "V15": "v15",
        "V16": "v16",
        "V17": "v17",
        "V18": "v18",
        "V19": "v19",
        "V20": "v20",
        "V21": "v21",
        "V22": "v22",
        "V23": "v23",
        "V24": "v24",
        "V25": "v25",
        "V26": "v26",
        "V27": "v27",
        "V28": "v28",
        "amount": "amount",
        "Class": "class",
    }

    # Rename columns
    df_db = df.rename(columns=column_mapping)

    # Add metadata columns
    df_db["dataset_source"] = "kaggle_creditcard"
    df_db["imported_at"] = datetime.now()

    print("✅ Data prepared for insertion")

    # Insert data in batches
    batch_size = 1000
    total_rows = len(df_db)

    print(f"Starting insertion of {total_rows:,} rows in batches of {batch_size}...")

    try:
        inserted_count = 0

        for i in range(0, total_rows, batch_size):
            batch = df_db.iloc[i : i + batch_size]

            # Insert batch
            batch.to_sql(
                "training_transactions", engine, if_exists="append", index=False
            )

            inserted_count += len(batch)
            progress = (inserted_count / total_rows) * 100

            print(f"Progress: {inserted_count:,}/{total_rows:,} rows ({progress:.1f}%)")

        print("✅ All data inserted successfully!")

    except Exception as e:
        print(f"❌ Error during insertion: {e}")
        sys.exit(1)

    # Final verification
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM training_transactions"))
        final_count = result.fetchone()[0]

        print(f"✅ Final count: {final_count:,} rows in training_transactions table")

        # Get fraud statistics
        result = conn.execute(
            text(
                """
            SELECT
                COUNT(*) as total_transactions,
                SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) as fraud_count,
                ROUND(AVG(CASE WHEN class = 1 THEN 1.0 ELSE 0.0 END), 4) as fraud_rate,
                ROUND(AVG(amount), 2) as avg_amount
            FROM training_transactions
        """
            )
        )

        stats = result.fetchone()

        print("\n📊 Final Statistics:")
        print(f"Total transactions: {stats[0]:,}")
        print(f"Fraud transactions: {stats[1]:,}")
        print(f"Fraud rate: {stats[2]:.4f}")
        print(f"Average amount: ${stats[3]:.2f}")

    print(
        "\n🎉 Data loading complete! The training_transactions table is ready for model training."
    )


if __name__ == "__main__":
    main()
