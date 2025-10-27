from .build import build
from .clear_cache import clear_cache
from .status import status

import click


@click.group("system")
def system():
    pass


system.add_command(build)
system.add_command(clear_cache)
system.add_command(status)
