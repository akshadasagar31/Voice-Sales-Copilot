"""
Re-indexing script: Migrates all documents/chunks from the existing index
'voice-sales-copilot' to the new integrated embeddings index 'voice-sales-copilot-llama'
using NVIDIA's 'llama-text-embed-v2'.
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_to_integrated")

SOURCE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-sales-copilot")
TARGET_INDEX_NAME = "voice-sales-copilot-llama"
TARGET_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "sales_playbooks")
MODEL_NAME = "llama-text-embed-v2"

def main():
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        logger.error("PINECONE_API_KEY is not set in backend/.env!")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    existing_indexes = pc.list_indexes().names()
    logger.info(f"Existing Pinecone indexes: {existing_indexes}")

    if SOURCE_INDEX_NAME not in existing_indexes:
        logger.error(f"Source index '{SOURCE_INDEX_NAME}' does not exist!")
        sys.exit(1)

    # 1. Create target integrated index if it does not exist
    if TARGET_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating new integrated index '{TARGET_INDEX_NAME}' with model '{MODEL_NAME}'...")
        pc.create_index_for_model(
            name=TARGET_INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": MODEL_NAME,
                "field_map": {"text": "text"},
                "metric": "cosine",
            },
        )
        logger.info(f"Initiated creation of '{TARGET_INDEX_NAME}'. Waiting for Ready state...")
        while not pc.describe_index(TARGET_INDEX_NAME).status.get("ready"):
            time.sleep(2)
        logger.info(f"Index '{TARGET_INDEX_NAME}' is Ready!")
    else:
        logger.info(f"Target index '{TARGET_INDEX_NAME}' already exists.")

    source_idx = pc.Index(SOURCE_INDEX_NAME)
    target_idx = pc.Index(TARGET_INDEX_NAME)

    # 2. Collect all vector IDs from source index in TARGET_NAMESPACE
    logger.info(f"Scanning vector IDs from source index '{SOURCE_INDEX_NAME}' (namespace: '{TARGET_NAMESPACE}')...")
    all_ids = []
    try:
        for page in source_idx.list(namespace=TARGET_NAMESPACE):
            for item in page:
                v_id = item if isinstance(item, str) else getattr(item, "id", str(item))
                all_ids.append(v_id)
    except Exception as e:
        logger.error(f"Failed to list IDs from source index: {e}")
        sys.exit(1)

    logger.info(f"Found {len(all_ids)} vector IDs in source namespace '{TARGET_NAMESPACE}'.")

    if not all_ids:
        logger.warning("No vectors found to migrate. Exiting.")
        return

    # 3. Fetch vector records with metadata in batches of 100
    batch_size = 50
    total_migrated = 0
    start_time = time.perf_counter()

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        logger.info(f"Fetching batch {i//batch_size + 1}/{(len(all_ids) + batch_size - 1)//batch_size} ({len(batch_ids)} records)...")
        
        fetch_res = source_idx.fetch(ids=batch_ids, namespace=TARGET_NAMESPACE)
        vectors_dict = fetch_res.get("vectors") if isinstance(fetch_res, dict) else getattr(fetch_res, "vectors", {})

        records_to_upsert = []
        for vid, vdata in vectors_dict.items():
            if isinstance(vdata, dict):
                meta = vdata.get("metadata") or {}
            else:
                meta = getattr(vdata, "metadata", {}) or {}
                if hasattr(meta, "to_dict"):
                    meta = meta.to_dict()
                elif not isinstance(meta, dict):
                    try:
                        meta = dict(meta)
                    except Exception:
                        meta = {}

            text_content = meta.get("text", "").strip()
            if not text_content:
                logger.warning(f"Record '{vid}' has empty text in metadata; skipping.")
                continue

            record = {
                "_id": str(vid),
                "text": text_content,
                "source": str(meta.get("source", "")),
                "page": int(meta.get("page", 1)) if meta.get("page") is not None else 1,
                "chunk_index": int(meta.get("chunk_index", 0)) if meta.get("chunk_index") is not None else 0,
                "character_count": int(meta.get("character_count", len(text_content))),
            }
            records_to_upsert.append(record)

        if records_to_upsert:
            target_idx.upsert_records(namespace=TARGET_NAMESPACE, records=records_to_upsert)
            total_migrated += len(records_to_upsert)
            logger.info(f"  -> Upserted {len(records_to_upsert)} records to '{TARGET_INDEX_NAME}' (Total: {total_migrated}/{len(all_ids)})")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Successfully migrated {total_migrated} records in {elapsed:.2f} seconds.")

    # 4. Verify target index stats
    time.sleep(2)
    stats = target_idx.describe_index_stats()
    logger.info(f"Target index stats: {stats.to_dict()}")

if __name__ == "__main__":
    main()
