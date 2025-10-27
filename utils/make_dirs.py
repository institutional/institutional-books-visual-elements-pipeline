import os


def make_dirs() -> None:
    """
    Creates target dirs as needed.
    Throws if any of the target destinations cannot be written.
    """
    from const import DATA_DIR_PATH, CACHE_DIR_PATH

    os.makedirs(DATA_DIR_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR_PATH, exist_ok=True)
