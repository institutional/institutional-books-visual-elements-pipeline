import os
import peewee

database_proxy = peewee.DatabaseProxy()

_init_pid = None
""" Keeps track of the pid from which the connection was initialized. """


def get_db() -> peewee.DatabaseProxy:
    """
    Process-safe access to the datababse.

    Returns an active database proxy.

    Automatically recreates connection:
    - Not available
    - `get_db()` is called from a subprocess/fork.
    """
    global _init_pid
    current_pid = os.getpid()

    # Check if we need to initialize or reinitialize due to process fork
    if database_proxy.obj is None or _init_pid != current_pid or database_proxy.obj.is_closed():

        db = peewee.PostgresqlDatabase(
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            autorollback=True,
            sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
        )

        try:
            db.connect()
            database_proxy.initialize(db)
            _init_pid = current_pid
        except Exception as err:
            raise ConnectionError(f"Could not connect to PostgreSQL.") from err

    return database_proxy
