# =============================================================================
# Database connection — asn_scraped_accidents
# =============================================================================
# Fill in your credentials below, then run:
#   python tests/db_connect.py
# to verify the connection and preview the data.
# =============================================================================

import psycopg2
import pandas as pd

# ── Connection config — fill these in ────────────────────────────────────────
DB_CONFIG = {
    "host":     "172.29.98.161",      # change if not local
    "port":     5432,             # default PostgreSQL port
    "dbname":   "aviation_db",   # e.g. "aviation_safety"
    "user":     "manyara",  # e.g. "postgres"
    "password": "toormaster",
}

TABLE = "public.asn_scraped_accidents"

# ── How many rows to pull for testing ────────────────────────────────────────
SAMPLE_SIZE = 20   # increase once you've confirmed connection works


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_narratives(limit: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Pull narratives + metadata from the database.
    Returns a DataFrame with columns:
        uid, narrative, phase, category, aircraft_type, operator,
        location, date, aircraft_damage
    """
    query = f"""
        SELECT
            uid,
            narrative,
            phase,
            category,
            aircraft_type,
            operator,
            location,
            date,
            aircraft_damage
        FROM {TABLE}
        WHERE narrative IS NOT NULL
          AND TRIM(narrative) != '';
    """
    conn = get_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


if __name__ == "__main__":
    print(f"Connecting to {DB_CONFIG['dbname']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    try:
        conn = get_connection()
        conn.close()
        print("✅ Connection successful")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        exit(1)

    print(f"\nFetching {SAMPLE_SIZE} sample rows from '{TABLE}'...")
    df = fetch_narratives(SAMPLE_SIZE)

    print(f"✅ Got {len(df)} rows")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample narratives:")
    for i, row in df.head(3).iterrows():
        print(f"\n  [{row.get('uid', i)}]  phase={row.get('phase', '?')}")
        narrative = str(row['narrative'])
        print(f"  {narrative[:200]}{'...' if len(narrative) > 200 else ''}")
