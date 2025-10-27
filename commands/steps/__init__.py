import click
from .step01_detect import step01_detect


@click.group("steps")
def steps():
    pass


steps.add_command(step01_detect)
