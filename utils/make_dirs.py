import os


def make_dirs() -> None:
    """
    Creates target dirs as needed.
    Throws if any of the target destinations cannot be written.
    """
    from const import (
        DATA_DIR_PATH,
        CACHE_DIR_PATH,
        HASH_DEDUPE_CACHE_DIR,
        DEDUPE_EMBEDDING_CACHE_DIR,
        ANALYSIS_OUTPUT_DIR,
        PEEK_OUTPUT_DIR,
    )

    os.makedirs(DATA_DIR_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR_PATH, exist_ok=True)
    os.makedirs(HASH_DEDUPE_CACHE_DIR, exist_ok=True)
    os.makedirs(DEDUPE_EMBEDDING_CACHE_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PEEK_OUTPUT_DIR, exist_ok=True)
