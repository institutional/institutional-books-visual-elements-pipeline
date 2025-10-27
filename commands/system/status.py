from pathlib import Path
import multiprocessing
import os

import click
import peewee
from humanize import naturalsize, intcomma

import utils
import models
from models import PipelineRun


@click.command("status")
def status():
    """
    Reports on the pipeline's status (database and cache size, etc ...)
    """

    def _print_section_heading(heading: str):
        click.echo(80 * "-")
        click.echo(heading)
        click.echo(80 * "-")

    #
    # Database
    #
    _print_section_heading("Database status")

    available_models = [model_name for model_name in dir(models) if model_name[0].isupper()]

    for model_name in available_models:
        model: peewee.Model = models.__getattribute__(model_name)
        table_name = model._meta.table_name
        click.echo(f"Table {table_name}: {intcomma(model.select().count())} record(s)")

    #
    # Resources
    #
    _print_section_heading("Resources")
    click.echo(f"Total CPU cores/threads: {multiprocessing.cpu_count()}")
    click.echo(f"Torch devices: {", ".join(utils.get_torch_devices())}")

    #
    # Cache
    #
    _print_section_heading("Cache status")

    cache_size_max = int(os.getenv("CACHE_MAX_SIZE_IN_GB", 1)) * 1_000_000_000
    cache_size_current = utils.get_cache().volume()

    click.echo(
        f"Cache size: " + f"{naturalsize(cache_size_current)} / {naturalsize(cache_size_max)}"
    )
