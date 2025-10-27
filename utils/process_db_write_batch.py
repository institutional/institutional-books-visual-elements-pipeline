import peewee


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
    batch_size = int(65535 / 20)

    if entries_to_create:
        model.bulk_create(entries_to_create, batch_size=batch_size)
        entries_to_create.clear()

    if entries_to_update:
        model.bulk_update(
            entries_to_update,
            fields=fields_to_update,
            batch_size=batch_size,
        )
        entries_to_update.clear()

    return True
