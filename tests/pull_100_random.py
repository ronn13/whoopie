import os
import json
import pandas as pd
from db_connect import get_connection

def pull_random_100():
    query = """
        SELECT uid, narrative
        FROM public.asn_scraped_accidents
        WHERE narrative IS NOT NULL
          AND LENGTH(TRIM(narrative)) > 50
        ORDER BY RANDOM()
        LIMIT 100;
    """
    
    print("Connecting to DB and fetching 100 random narratives...")
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    output_path = os.path.join("tests", "100_to_annotate.json")
    
    # Save as records
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(records)} records to {output_path}")

if __name__ == "__main__":
    pull_random_100()
