import boto3
from botocore.config import Config
import os

VALID_TARGETS = ["GRIN_DATA", "OUTPUT", "FILTER"]


def get_s3_client(target: str = VALID_TARGETS[0]) -> object:
    """
    boto3 helper for connecting to S3-compatible, remote storage.

    target can be:
    - "GRIN_DATA": Remote storage for Google Books tar.gz archives
    - "OUTPUT": Remote storage for the pipeline's output
    - "FILTER": Remote storage for filtered dataset exports
    """
    assert target in VALID_TARGETS

    return boto3.client(
        "s3",
        endpoint_url=os.getenv(f"{target}_ENDPOINT"),
        aws_access_key_id=os.getenv(f"{target}_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv(f"{target}_SECRET_ACCESS_KEY"),
        config=Config(
            region_name=os.getenv(f"{target}_REGION", "auto"),
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
