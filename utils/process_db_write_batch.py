import peewee
from peewee import OperationalError
from psycopg2 import OperationalError as PsycopgOperationalError, InterfaceError

from utils.get_db import get_db

TRANSIENT_DB_ERRORS = (OperationalError, PsycopgOperationalError, InterfaceError)


def _with_db_retry(fn, retries: int = 3, base_delay: float = 0.5):
    import time

    last_exc = None
    for attempt in range(retries):
        try:
            # Ensure DB proxy is initialized for this process
            db = get_db()
            with db.connection_context():
                return fn()
        except TRANSIENT_DB_ERRORS as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    raise last_exc  # should be unreachable


def process_db_write_batch(
    model: peewee.Model,
    entries_to_create: list[peewee.Model] = [],
    entries_to_update: list[peewee.Model] = [],
    fields_to_update: list[peewee.Field] = [],
) -> bool:
    """
    Processes a batch of database create/update operations.

    Notes:
    - `entries_to_create` an `entries_to_update` are emptied in place
    """
    if entries_to_create is None:
        entries_to_create = []
    if entries_to_update is None:
        entries_to_update = []
    if fields_to_update is None:
        fields_to_update = []

    batch_size = int(65535 / 20)

    if entries_to_create:

        def _create():
            model.bulk_create(entries_to_create, batch_size=batch_size)

        _with_db_retry(_create)
        entries_to_create.clear()

    if entries_to_update:

        def _update():
            model.bulk_update(
                entries_to_update,
                fields=fields_to_update,
                batch_size=batch_size,
            )

        _with_db_retry(_update)
        entries_to_update.clear()

    return True
