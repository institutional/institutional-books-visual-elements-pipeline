from dotenv import load_dotenv

load_dotenv()

import sys

import click
from loguru import logger

import utils

from commands.system import system
from commands.orchestration import orchestration
from commands.steps import steps
from commands.export import export

utils.check_env()
utils.make_dirs()


@click.group()
@click.option(
    "--verbose",
    is_flag=True,
    help="If set, includes DEBUG-level statements in log output.",
)
def cli(verbose: bool):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)

    # Always try to create tables
    utils.create_tables()


cli.add_command(system)
cli.add_command(orchestration)
cli.add_command(steps)
cli.add_command(export)

if __name__ == "__main__":
    cli()
