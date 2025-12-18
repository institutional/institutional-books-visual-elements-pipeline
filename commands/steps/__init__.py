import click
from .step01_detect import step01_detect
from .step02_classify import step02_classify
from .step03_generate_dedupe_data import step03_generate_dedupe_data
from .step04_process_caption_requests import step04_process_caption_requests
from .step05_store import step05_store
from .step06_dedupe_by_image_hash import step06_dedupe_by_image_hash
from .step07_dedupe_by_image_embedding import step07_dedupe_by_image_embedding


@click.group("steps")
def steps():
    pass


steps.add_command(step01_detect)
steps.add_command(step02_classify)
steps.add_command(step03_generate_dedupe_data)
steps.add_command(step04_process_caption_requests)
steps.add_command(step05_store)
steps.add_command(step06_dedupe_by_image_hash)
steps.add_command(step07_dedupe_by_image_embedding)
