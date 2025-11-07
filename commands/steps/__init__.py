import click
from .step01_detect import step01_detect
from .step02_classify import step02_classify


@click.group("steps")
def steps():
    pass


steps.add_command(step01_detect)
steps.add_command(step02_classify)
