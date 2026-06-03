import click
from loguru import logger


@click.command("run-all")
@click.option(
    "--skip-thesaurus",
    is_flag=True,
    help="Skip ChronAm thesaurus matching during backfill",
)
@click.option(
    "--cpus-limit",
    type=int,
    default=4,
    help="Number of parallel worker processes for backfill (default: 4)",
)
@click.option(
    "--count-tokens",
    is_flag=True,
    help="Also run count-tokens after backfill",
)
@click.option(
    "--embedding-atlas",
    is_flag=True,
    help="Also run embedding-atlas after backfill",
)
@click.option(
    "--drop-existing-view",
    is_flag=True,
    help="Drop and recreate the filtered_dataset view",
)
@click.pass_context
def run_all(ctx, skip_thesaurus, cpus_limit, count_tokens, embedding_atlas, drop_existing_view):
    """
    Run the full post-processing pipeline in sequence.

    Executes: create-view -> backfill -> [count-tokens] -> [embedding-atlas]

    By default only runs create-view and backfill. Use --count-tokens and
    --embedding-atlas to include the optional steps.
    """
    from .create_view import create_view as create_view_cmd
    from .backfill import backfill as backfill_cmd
    from .count_tokens import count_tokens as count_tokens_cmd
    from .embedding_atlas import embedding_atlas as embedding_atlas_cmd

    logger.info("Starting post-processing pipeline...")

    logger.info("[1/{}] Creating filtered_dataset view...".format(
        2 + int(count_tokens) + int(embedding_atlas)
    ))
    ctx.invoke(create_view_cmd, drop_existing=drop_existing_view)

    logger.info("[2/{}] Running backfill...".format(
        2 + int(count_tokens) + int(embedding_atlas)
    ))
    ctx.invoke(backfill_cmd, skip_thesaurus=skip_thesaurus, cpus_limit=cpus_limit)

    step = 3
    total = 2 + int(count_tokens) + int(embedding_atlas)

    if count_tokens:
        logger.info(f"[{step}/{total}] Running count-tokens...")
        ctx.invoke(count_tokens_cmd)
        step += 1

    if embedding_atlas:
        logger.info(f"[{step}/{total}] Running embedding-atlas...")
        ctx.invoke(embedding_atlas_cmd, no_serve=True)

    logger.success("Post-processing pipeline complete.")
