from typing import Any
import json
import openai
import click


client = openai.OpenAI()

click.command("retrieve-captions")


def retrieve_captions():
    pass


def check_batch_status(client: Any, batch_name: str) -> Any:
    """
    Retrieves status information for a batch by name.

    Args:
        client: The API client.
        batch_name: Identifier of the batch to query.

    Returns:
        The batch object containing status information.
    """
    batch = client.batches.retrieve(batch_name)
    return batch


def retrieve_results(client: Any, batch_name: str) -> str:
    """
    Retrieves result file content for the given batch.

    Args:
        client: The API client.
        batch_name: Identifier of the batch.

    Returns:
        The response text of the result file.
    """
    file_response = client.files.content(batch_name)
    return file_response.text


def cancel_batch(client: Any, batch_name: str) -> None:
    """
    Cancels an existing batch.

    Args:
        client: The API client.
        batch_name: Identifier of the batch to cancel.
    """
    client.batches.cancel(batch_name)


def list_batches(client: Any, limit: int = 10) -> Any:
    """
    Lists batches up to a specified limit.

    Args:
        client: The API client.
        limit: Number of batches to retrieve (default 10).

    Returns:
        The result of client.batches.list().
    """
    return client.batches.list(limit=limit)


def get_responses(batch_file: str) -> None:
    """
    Prints response messages from lines in a batch results file (newline-delimited JSON).

    Args:
        batch_file: Path to the results file.
    """
    with open(batch_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Defensive chaining to safely get nested keys
                response = (
                    data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message")
                )
                print(response)
                print()  # Blank line for readability
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                print(f"Could not parse line: {line.strip()} ({e})")
