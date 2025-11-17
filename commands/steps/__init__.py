import click
from .step01_detect import step01_detect
from .step02_classify import step02_classify
from .step03_generate_dedupe_embeddings import step03_generate_dedupe_embeddings
from .step04_generate_caption_requests import step04_generate_caption_requests


@click.group("steps")
def steps():
    pass


steps.add_command(step01_detect)
steps.add_command(step02_classify)
steps.add_command(step03_generate_dedupe_embeddings)
steps.add_command(step04_generate_caption_requests)
