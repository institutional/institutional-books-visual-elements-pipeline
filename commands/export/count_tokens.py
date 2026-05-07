import click
import tiktoken
from loguru import logger

from models import Caption


@click.command("count-tokens")
@click.option(
    "--encoding",
    "encoding_name",
    type=str,
    default="cl100k_base",
    help="Tiktoken encoding to use (default: cl100k_base)",
)
def count_tokens(encoding_name):
    """Count the total number of tokens in the caption table's text column."""
    enc = tiktoken.get_encoding(encoding_name)

    total_tokens = 0
    total_rows = 0

    for cap in Caption.select(Caption.text).iterator():
        if cap.text:
            total_tokens += len(enc.encode(cap.text))
            total_rows += 1

    logger.info(f"Encoding: {encoding_name}")
    logger.info(f"Captions counted: {total_rows:,}")
    logger.info(f"Total tokens: {total_tokens:,}")
