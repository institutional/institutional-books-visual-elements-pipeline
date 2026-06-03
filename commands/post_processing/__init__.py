import click
from .backfill import backfill
from .count_tokens import count_tokens
from .embedding_atlas import embedding_atlas


@click.group("post-processing")
def post_processing():
    pass


post_processing.add_command(backfill)
post_processing.add_command(count_tokens)
post_processing.add_command(embedding_atlas)
