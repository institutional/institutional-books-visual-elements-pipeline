import utils


# Columns added to existing tables after initial creation.
# Each entry: (table_name, column_definition_sql)
# These run as "ADD COLUMN IF NOT EXISTS" before Peewee's create_tables
# to avoid index-creation errors on missing columns.
_PENDING_COLUMN_MIGRATIONS = [
    ("caption", "lang_detected VARCHAR(10)"),
    ("caption", "linear_prob DOUBLE PRECISION"),
    ("caption", "thesaurus_matches JSONB"),
]


def create_tables() -> bool:
    """
    Lists models and automatically creates database tables if needed.
    """
    import models

    available_models = [model_name for model_name in dir(models) if model_name[0].isupper()]

    db = utils.get_db()
    db.execute_sql("SET lock_timeout = '5s'")

    for table_name, col_def in _PENDING_COLUMN_MIGRATIONS:
        try:
            db.execute_sql(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_def}")
        except Exception:
            pass

    try:
        db.create_tables([models.__getattribute__(model_name) for model_name in available_models])
    except Exception:
        pass

    return True
