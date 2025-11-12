import json
from typing import List, Dict, Tuple, Any, Optional
import click
import os
from glob import glob
import openai
import base64
import io
from PIL import Image

client = openai.OpenAI()


click.command("request-captions")


def request_captions():
    # grab the text by using the institutional-books dataset
    # grab language using the institutional-books dataset
    # get batch of crops from cache using crop function
    # create batch files using batch of image crops
    # run process_batch() for each jsonl file
    pass


def create_batch_file(
    jsonl_filename: str,
    image_urls: List[str],
    page_texts: List[str],
    language: str,
    max_tokens: int,
) -> int:
    """
    Creates a JSONL batch file given lists of images and page texts, composing
    messages with the provided system/user messages and model parameters.

    Args:
        jsonl_filename: Path to output file (.jsonl).
        image_urls: File with list of base64-encoded image strings (one per request).
        page_texts: File with list of text content (one per request; must align with image_urls).
        system_message: Dictionary configuring the system message.
        user_messages: List of user message dicts (contextual messages).
        max_tokens: Maximum tokens for each model request.
    Returns:
        The number of requests written to the file.
    Raises:
        ValueError: If input lists are not aligned.
    """

    system_message, user_messages = create_prompt(language)

    image_urls = load_text_lines(image_urls)
    page_texts = load_text_lines(page_texts)

    if len(image_urls) != len(page_texts):
        raise ValueError(
            f"Length mismatch: {len(image_urls)} image_urls vs {len(page_texts)} page_texts"
        )

    model_name = "gpt-4.1-nano"
    logprobs_value = True
    top_logprobs_value = 2
    temperature_value = 0

    with open(jsonl_filename, "w", encoding="utf-8") as outfile:
        for idx, (img, text) in enumerate(zip(image_urls, page_texts), start=1):
            custom_id = f"request-{idx}"

            image_and_text_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    },
                ],
            }

            messages = [system_message] + user_messages + [image_and_text_message]

            body = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "logprobs": logprobs_value,
                "top_logprobs": top_logprobs_value,
                "temperature": temperature_value,
            }

            obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }

            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

    click.echo(f"Wrote {len(image_urls)} requests to {jsonl_filename}")


def load_text_lines(path):
    """Loads a file as lines or as a JSON list."""
    with open(path, "r", encoding="utf-8") as f:
        f.seek(0)
        f.seek(0)
        return [line.rstrip("\n") for line in f]


def create_prompt(language: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Constructs a system message and user prompt messages for image captioning in a given language.

    Args:
        language: The target language for the caption (e.g., "English", "Spanish").

    Returns:
        A tuple containing:
            - system_message: Dictionary with the system's context.
            - user_messages: List of user message dictionaries, including
              the caption instruction and the specified language.
    """
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a librarian that captions images in precise and concise language."
                ),
            }
        ],
    }

    user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Create a caption in 50 words or less for the image given the context by the page text."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Reply only with the caption."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Write the caption in {language}"}],
        },
    ]

    return system_message, user_messages


def preprocess(input_dir, output_dir):
    """Resize all images in INPUT_DIR and save to OUTPUT_DIR"""
    os.makedirs(output_dir, exist_ok=True)
    # Accept jpg and png images
    patterns = [
        os.path.join(input_dir, "*.jpg"),
        os.path.join(input_dir, "*.jpeg"),
        os.path.join(input_dir, "*.png"),
    ]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob(pattern))
    if not image_paths:
        click.echo(f"No images found in {input_dir}")
        return
    for img_path in image_paths:
        # Assuming process_image returns a PIL Image and new dimensions
        new_image, new_dims = process_image(img_path, 1248)
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        new_image.save(out_path)
        click.echo(f"Processed: {img_path} -> {out_path} ({new_dims[0]}x{new_dims[1]})")
    click.echo("Processing complete.")


def process_batch(batch_file, metadata):
    """Upload BATCH_FILE and create a batch job with optional --metadata."""
    # Handle metadata
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            try:
                with open(metadata, "r") as f:
                    metadata_dict = json.load(f)
            except Exception as e:
                click.echo(f"Error with metadata: {e}")
                return
    # Upload the file
    click.echo(f"Uploading {batch_file}...")
    file = upload_batch(client, batch_file)
    # Create the batch
    click.echo("Creating batch...")
    batch = create_batch(client, file, metadata=metadata_dict)
    click.echo(f"Batch created: {batch}")


def upload_batch(client: Any, batch_filename: str) -> Any:
    """
    Uploads a batch file to the API client.

    Args:
        client: The API client with file upload capability.
        batch_filename: Path to the batch file to upload.

    Returns:
        The uploaded file object as returned by the client.
    """
    # Use context manager to ensure file is closed properly
    with open(batch_filename, "rb") as file_obj:
        batch_input_file = client.files.create(file=file_obj, purpose="batch")
    return batch_input_file


def create_batch(client: Any, batch_file: Any, metadata: Optional[Dict[str, str]] = None) -> Any:
    """
    Creates a new batch using the uploaded file's ID.

    Args:
        client: The API client.
        batch_file: The uploaded batch file object (should have 'id').
        metadata: Optional metadata dictionary.

    Returns:
        The created batch object.
    """
    if metadata is None:
        metadata = {}
    batch_input_file_id = batch_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )
    print(f"Created batch: {batch}")
    return batch


def resize_image(image: Image.Image, max_dimension: int) -> Image.Image:
    """
    Resizes the given image so that its largest dimension is at most max_dimension,
    maintaining the aspect ratio. Uses LANCZOS filter for resizing.

    Args:
        image: A PIL Image object.
        max_dimension: The maximum width or height of the output image.

    Returns:
        A resized PIL Image object, or the original image if resizing is not needed.
    """
    width, height = image.size

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image


def convert_to_png(image: Image.Image) -> bytes:
    """
    Converts the given PIL Image to PNG bytes.

    Args:
        image: A PIL Image object.

    Returns:
        PNG-encoded image bytes.
    """
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def process_image(path: str, max_size: int) -> Tuple[str, int]:
    """
    Opens an image file, checks if it's already a PNG and under the max size.
    If not, resizes and converts it to PNG, then base64 encodes the image.

    Args:
        path: Path to the source image on disk.
        max_size: Maximum allowed width or height of the image in pixels.

    Returns:
        A tuple of (base64-encoded PNG image string, original largest dimension).
    """
    with Image.open(path) as image:
        width, height = image.size

        # Attempt to determine mime type: get_format_mimetype is available in Pillow>=7.0.0
        try:
            mimetype = image.get_format_mimetype()
        except AttributeError:
            # Fallback: infer mimetype from format if needed
            mimetype = Image.MIME.get(image.format, "application/octet-stream")

        if mimetype == "image/png" and width <= max_size and height <= max_size:
            # Image is already a PNG and fits size requirements; just read as is
            with open(path, "rb") as file:
                encoded_image = base64.b64encode(file.read()).decode("utf-8")
            return encoded_image, max(width, height)
        else:
            # Resize and convert to PNG as necessary
            resized_image = resize_image(image, max_size)
            png_image = convert_to_png(resized_image)
            encoded_image = base64.b64encode(png_image).decode("utf-8")
            return encoded_image, max(width, height)
