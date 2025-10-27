from .prepare import prepare
from .execute import execute
from .status import status

import click


@click.group("orchestration")
def orchestration():
    pass


orchestration.add_command(prepare)
orchestration.add_command(execute)
orchestration.add_command(status)
