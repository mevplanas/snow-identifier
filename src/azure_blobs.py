# Import OS packages
import os

# Import configuration
import yaml

# Import Azure API
from azure.storage.blob import BlobServiceClient, __version__

# Import time
from datetime import datetime, timedelta

from tqdm import tqdm

IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def get_blob_names_by_days(config: yaml, days_ago: int = 7) -> list:
    """

    This function gets names (paths) of blobs that were uploaded today (by metadata).

    Arguments
    config: yaml
        Configuration file as dictionary.
    days_ago: int
        days before today

    Outputs
    List of blob names (paths) that were uploaded today.

    """

    # Define the connection string and container name
    connect_str = config["AZURE_INPUT_VASA"]["conn_string"]
    container_name = config["AZURE_INPUT_VASA"]["container_name"]

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get a BlobClient object for the container
    container_client = blob_service_client.get_container_client(container_name)

    # Get the date range to filter blobs
    query_date_start = datetime.utcnow().date() - timedelta(days=days_ago)
    query_date_end = datetime.utcnow().date()
    date_sequence = [
        query_date_start + timedelta(days=x)
        for x in range((query_date_end - query_date_start).days + 1)
    ]

    # Converting date to string of format %YYYY%mm%dd
    date_sequence = [x.strftime("%Y%m%d") for x in date_sequence]

    # Filter blobs by creation time
    image_blobs_to_use = []

    for date in date_sequence:
        current_blobs = container_client.list_blobs(
            name_starts_with=date, include=["metadata"]
        )
        for blob in tqdm(current_blobs, desc="Getting current blobs"):
            if blob.name.endswith(IMAGE_FORMATS):
                image_blobs_to_use.append(blob.name)

    return image_blobs_to_use


def download_image(blob_name: str, config: yaml, local_file_dir: str) -> None:
    """

    This function saves blob locally as image.

    Arguments
    blob_name: str
        Blob name as path which to download.
    config: yaml
        Configuration file as dictionary.
    local_file_dir: str
        Directory where to save blob.

    Outputs
    None

    """

    connect_str = config["AZURE_INPUT_VASA"]["conn_string"]
    container_name = config["AZURE_INPUT_VASA"]["container_name"]

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get a BlobClient object for the blob to download
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )

    # Extracting the last part of the blob name
    blob_img_name = blob_name.split("/")[-1]

    # Extracting the path to blob
    blob_path = "/".join(blob_name.split("/")[:-1])

    # Check if path exists
    blob_local_dir = os.path.join(local_file_dir, blob_path)
    if not os.path.exists(blob_local_dir):
        os.makedirs(blob_local_dir)

    # Download the blob to a local file
    with open(os.path.join(blob_local_dir, blob_img_name), "wb") as blob:
        try:
            download_stream = blob_client.download_blob()
            blob.write(download_stream.readall())
        except Exception as e:
            print(f"Cannot download blob: {e}")


def get_blobs_by_folder_name(config: yaml) -> list:
    """

    This function gets names (paths) of blobs that were uploaded today (by metadata).

    Arguments
    config: yaml
        Configuration file as dictionary.
    days_ago: int
        days before today

    Outputs
    List of blob names (paths) that were uploaded today.

    """

    # Define the connection string and container name
    connect_str = config["AZURE_INPUT_VASA"]["conn_string"]
    container_name = config["AZURE_INPUT_VASA"]["container_name"]

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get a BlobClient object for the container
    container_client = blob_service_client.get_container_client(container_name)

    # Filter blobs by creation time
    image_blobs_to_use = []

    current_blobs = container_client.list_blobs(
        name_starts_with="RnD/Gatvių valymas/2023-2024/20231130 Ateieties-Jaruzales-saligatviai/",
        include=["metadata"],
    )
    for blob in tqdm(current_blobs, desc="Getting current blobs"):
        if blob.name.endswith(IMAGE_FORMATS):
            image_blobs_to_use.append(blob.name)

    return image_blobs_to_use


def upload_image(config: yaml, colored_img_name: str, img_bytes: bytes) -> None:
    # Define the connection string and container name
    connect_str = config["AZURE_INPUT_VASA"]["conn_string"]
    container_name = config["AZURE_INPUT_VASA"]["output_container_name"]

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get a BlobClient object for the container
    container_client = blob_service_client.get_container_client(container_name)

    container_client.upload_blob(colored_img_name, img_bytes)
