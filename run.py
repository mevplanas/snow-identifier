# Importing OS
import os

# SQLIte database for infered images
import sqlite3

# JSON file handling
import json

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

_TAGS_r = dict(((v, k) for k, v in TAGS.items()))


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
def infer_snow(image_path: str, box_padding: int = 25) -> float:
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
    image = cv2.imread(image_path)

    # Converting to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
def pipeline() -> None:
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

    # Initializing the database
    conn = sqlite3.connect(os.path.join(current_dir, "database.db"))

    # Listing the input images
    images = os.listdir(os.path.join(current_dir, "input"))

    # Infering which images are in the database
    cursor = conn.cursor()
    cursor.execute("SELECT distinct image_name FROM processed_images")
    result = cursor.fetchall()
    result = [r[0] for r in result]

    # Infering the images that are not in the database
    images_to_process = [image for image in images if image not in result]

    # Creating the output dir
    output_path = os.path.join(current_dir, "output")
    os.makedirs(output_path, exist_ok=True)

    # Iterating over the images
    for image in tqdm(images_to_process):
        # Getting the mean pixel value
        mean_pixel_value = infer_snow(
            os.path.join(current_dir, "input", image), box_padding=box_padding
        )

        # Infering the label
        label = infer_label(mean_pixel_value)

        # Uploading to database
        cursor.execute(
            """
            INSERT INTO processed_images (image_name, datetime_processed, prediction_prob, prediction_class)
            VALUES (?, datetime('now'), ?, ?)
        """,
            (image, mean_pixel_value, label),
        )

        # Committing the changes
        conn.commit()

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


if __name__ == "__main__":
    pipeline()
