# Importing OS
import os

# SQLIte database for infered images
import sqlite3

# JSON file handling
import json
from textwrap import fill

# Computer vision
import cv2

# Datetime wrangling
from datetime import datetime

# YAML reading
import yaml

# Iteration tracking
from tqdm import tqdm

# Import computer vision
from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np

_TAGS_r = dict(((v, k) for k, v in TAGS.items()))

# Import Azure Blob clients
from src.azure_blobs import download_image, get_blobs_by_folder_name, upload_image

# Import system
from sys import platform

# Import database models
from db.models import ImagePredictions

# Import string wrangling
import re

# Import geometry
from shapely.geometry import Point

from azure.storage.blob import BlobServiceClient, __version__

IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def image_link_generator(file_path: str) -> str:
    storage_url = "https://vp3ddata.blob.core.windows.net/raw-data-bo-nuotraukos/"

    if platform == "win32":
        # Creating the target name
        _target_name = file_path.split("images\\")[-1]
        target_name = re.sub("\\\\", "/", _target_name)
    else:
        _target_name = file_path.split("images/")[-1]
        target_name = re.sub("//|/", "/", _target_name)

    blob_url = f"{storage_url}{target_name}"

    return blob_url


def get_meta(file_path):
    """
    The functions extracts exif metadata from image

    Arguments
    ---------
    img_bytes: bytes
        bytes image

    Output
    ------
    long : float
        image center longitude in WGS
    lat : float
        image center lattitude in WGS
    """

    img = Image.open(file_path)
    # Get EXIF data
    exifd = img._getexif()

    if exifd == None:
        long = 0.00
        lat = 0.00

        return long, lat

    # Getting keys of EXIF data
    keys = list(exifd.keys())

    # Remove MakerNote tag because these can be too long
    keys.remove(_TAGS_r["MakerNote"])

    # Get key values from img exif data
    keys = [k for k in keys if k in TAGS]

    # GPS
    gpsinfo = exifd[_TAGS_r["GPSInfo"]]

    # Decimal part of longitude
    decimal_long = (float(gpsinfo[2][1]) * 60 + float(gpsinfo[2][2])) / 3600

    # Longitude
    long = float(gpsinfo[2][0]) + decimal_long

    # Decimal part of latitude
    decimal_lat = (float(gpsinfo[4][1]) * 60 + float(gpsinfo[4][2])) / 3600

    # Latitude
    lat = float(gpsinfo[4][0]) + decimal_lat

    return long, lat


# Defining the function for prediction
def infer_snow(image_path: str, output_path: str, box_padding: int = 25) -> float:
    """
    Function to infer whether the image has snow or not

    Parameters
    ----------
    image_path : str
        Path to the image
    box_padding : int
        Padding for the bounding box

    Returns
    -------
    float
        Probability of the image having snow
    """
    # Reading the image
    img = Image.open(image_path)
    open_cv_image = np.array(img)

    # Converting to grayscale
    image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Converting from 255 to 1
    image = image / 255

    # Getting box around the center point of the image
    image_height, image_width = image.shape

    # Getting the center point
    center_point = (image_width // 2, image_height // 2)

    # Getting the box coordinates
    x_min = center_point[0] - box_padding
    x_max = center_point[0] + box_padding
    y_min = center_point[1] - box_padding
    y_max = center_point[1] + box_padding

    # Getting the box
    box = image[y_min:y_max, x_min:x_max]

    # Calculating the mean pixel value
    mean_pixel_value = box.mean()

    # Calculating top left and bottom right corners
    top_left = x_min, y_max
    bottom_right = x_max, y_min

    # Opening image with cv2
    # colored_image = cv2.imread(open_cv_image)
    colored_image = cv2.rectangle(
        open_cv_image, top_left, bottom_right, color=(255, 0, 0), thickness=3
    )

    # Saving colored image
    cv2.imwrite(output_path, colored_image)

    # Returning the probability
    return mean_pixel_value


def infer_label(prob: float) -> str:
    if prob < 0.33:
        return "no_snow"
    elif prob < 0.66:
        return "maybe_snow"
    else:
        return "snow"


# Defining the pipeline
def pipeline(env: str = "dev") -> None:
    """
    Pipeline that reads the input images and creates the output json files
    """
    # Infering the current file dir
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Reading the configuration.yml file
    with open(os.path.join(current_dir, "configuration.yml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Extracting the padding for the snow box
    box_padding = config["PADDING"]

    connect_str = config["AZURE_INPUT_VASA"]["conn_string"]
    container_name = config["AZURE_INPUT_VASA"]["output_container_name"]

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Get a BlobClient object for the container
    container_client = blob_service_client.get_container_client(container_name)

    # # Initializing the database
    # conn = sqlite3.connect(os.path.join(current_dir, "database.db"))

    if env == "dev":
        # Listing the input images
        images = os.listdir(os.path.join(current_dir, "input"))
    else:
        # Define direcotry where image will be stored
        images_local_dir = os.path.join(current_dir, "images")
        # os.makedirs(images_local_dir, exist_ok=True)
        # # Get images blob name form Azure Storage
        # storage_images = get_blobs_by_folder_name(config=config)
        # # Downloading images from Azure storage to local dir
        # for img in tqdm(storage_images, desc="Downloading images from Azure Storage:"):
        #     download_image(
        #         blob_name=img, config=config, local_file_dir=images_local_dir
        #     )

        images = []
        for root, dirs, files in os.walk(images_local_dir):
            for file in files:
                # Infering whether the file ends with .jpg, .JPG, .jpeg, .png, .PNG
                if file.endswith(IMAGE_FORMATS):
                    # Adding the full path to the file
                    file = os.path.join(root, file)
                    # Appending to the list of images to infer
                    images.append(file)

    db_images = ImagePredictions.read_distinct()
    result = db_images["image_name"].tolist()

    # Infering the images that are not in the database
    images_to_process = [image for image in images if image not in result]

    # Creating the output dir
    output_path = os.path.join(current_dir, "output")
    os.makedirs(output_path, exist_ok=True)

    # Get last object id in database
    obj_id = ImagePredictions.select_max_id()
    if obj_id == None:
        obj_id = 0
    else:
        obj_id += 1

    # Defining directory for colored images
    output_dir = os.path.join(current_dir, "colored_images")
    os.makedirs(output_dir, exist_ok=True)

    # Iterating over the images
    for image in tqdm(images_to_process):
        # Defining colored image path
        image_output_path = os.path.basename(image)

        # Getting the mean pixel value
        mean_pixel_value = infer_snow(
            image, box_padding=box_padding, output_path=image_output_path
        )

        with open(file=image_output_path, mode="rb") as data:
            container_client.upload_blob(name=image, data=data, overwrite=True)

        # Infering the label
        label = infer_label(mean_pixel_value)

        image_link = image_link_generator(image)

        x, y = get_meta(image)
        point_geom = str(Point(x, y))

        record = [
            {
                "OBJECTID": obj_id,
                "image_name": image,
                "prediction_prob": mean_pixel_value,
                "prediction_class": label,
                "image_link_original": image_link,
                "image_link_processed": image_link,
                "datetime_processed": str(datetime.now()),
                "Shape": point_geom,
            }
        ]

        # upload_image(config=config, colored_img_name=)
        ImagePredictions.insert_records(record)

        # Saving the image to json to output path
        with open(os.path.join(output_path, image.replace(".jpg", ".json")), "w") as f:
            json.dump(
                {
                    "image_name": image,
                    "datetime_processed": str(datetime.now()),
                    "prediction_prob": mean_pixel_value,
                    "prediction_class": label,
                },
                f,
            )
        obj_id += 1


if __name__ == "__main__":
    pipeline(env="prod")
