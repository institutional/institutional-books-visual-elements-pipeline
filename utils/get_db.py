import os
import os
import peewee
from playhouse.shortcuts import ReconnectMixin

database_proxy = peewee.DatabaseProxy()

_init_pid = None
""" Keeps track of the pid from which the connection was initialized. """


class ReconnectPostgresqlDatabase(ReconnectMixin, peewee.PostgresqlDatabase):
    """
    Postgres DB that transparently reconnects on common connection errors
    (including EOF / server-closed connections).
    """

    pass


def get_db() -> peewee.DatabaseProxy:
    """
    Process-safe access to the database.

    Returns an active database proxy.

    Automatically (re)initializes the underlying DB object when:
    - Not initialized yet
    - `get_db()` is called from a different process (after fork)
    """
    global _init_pid
    current_pid = os.getpid()

    # (Re)initialize for this PID
    if database_proxy.obj is None or _init_pid != current_pid:
        db = ReconnectPostgresqlDatabase(
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
            raise ConnectionError("Could not connect to PostgreSQL.") from err

    # We deliberately do *not* call db.is_closed() here; ReconnectMixin will
    # reconnect on demand when a query is issued and a connection error occurs.
    return database_proxy
