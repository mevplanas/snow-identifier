# Importing OS
import os

# Datetime wrangling
from datetime import datetime

# YAML reading
import yaml

# Iteration tracking
from tqdm import tqdm

#  Math
import numpy as np

# Computer vision
import cv2

# Azure Storage Blobs
from src.azure_blobs import download_image, get_blobs_by_folder_name, upload_image

# Utilis
from src.custom_utils import image_link_generator, infer_label, blob_renamer

# Image processing
from src.image_processing import (
    get_closest_mask,
    get_closest,
    get_meta,
    get_poly_coords,
    get_img_datetime,
    get_intersection_geom,
    infer_snow,
    COLOR_DICT,
)

# Import database models
from db.models import ImagePredictions, InspectionPoints

# Import geometry
from shapely.geometry import Point

# Ultralytics YOLO model
from ultralytics import YOLO

IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


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
    box_padding = config["BBOX_PADDING"]
    infer_box_padding = config["INFER_BBOX_PADDING"]

    # # Initializing the database
    # conn = sqlite3.connect(os.path.join(current_dir, "database.db"))
    images = []
    if env == "dev":
        # Listing the input images
        images = os.listdir(os.path.join(current_dir, "input"))
    else:
        # Define direcotry where image will be stored
        images_local_dir = os.path.join(current_dir, "images")
        os.makedirs(images_local_dir, exist_ok=True)
        # Blob Prefix
        prefix = config["AZURE_INPUT"]["blob_prefix"]
        # Get images blob name form Azure Storage
        storage_images = get_blobs_by_folder_name(
            config=config, name_starts_with=prefix
        )
        # 20231204 Fabijoniskes/DJI_0764
        #
        # Downloading images from Azure storage to local dir
        for img in tqdm(storage_images, desc="Downloading images from Azure Storage:"):
            download_image(
                blob_name=img, config=config, local_file_dir=images_local_dir
            )

        for root, dirs, files in os.walk(images_local_dir):
            for file in files:
                # Infering whether the file ends with .jpg, .JPG, .jpeg, .png, .PNG
                if file.endswith(IMAGE_FORMATS):
                    # Adding the full path to the file
                    file = os.path.join(root, file)
                    blob = blob_renamer(file)
                    # Appending to the list of images to infer
                    images.append(blob)

    # Get inspection points from MSS database
    inspection_points = InspectionPoints.get_all()

    # Defining the default obj_id
    obj_id = 0
    images_to_process = images.copy()

    try:
        # Reading the database
        db_images = ImagePredictions.read_distinct()
        result = db_images["image_name"].tolist()

        # Infering the images that are not in the database
        images_to_process_blobs = [image for image in images if image not in result]

        # Create image path from blob name
        images_to_process = [
            os.path.join(images_local_dir, *blob.split("/"))
            for blob in images_to_process_blobs
        ]

        # Creating the output dir
        output_path = os.path.join(current_dir, "output")
        os.makedirs(output_path, exist_ok=True)

        # Get last object id in database
        obj_id = ImagePredictions.select_max_id()
        if obj_id == None:
            obj_id = 0
        else:
            obj_id += 1
    except Exception as e:
        print(e)
        obj_id = 0

    # Defining directory for colored images
    output_dir = os.path.join(current_dir, "colored_images")
    os.makedirs(output_dir, exist_ok=True)

    # Creating a placeholder to store the results
    records = []

    # Defining the model to use
    path_to_model = os.path.join("machine_learning", "models", "path_identifier.pt")

    # Loading the model
    model = YOLO(path_to_model)

    # Threshold
    threshold = config["PATH_DETECTION_MODEL"]["threshold"]

    # Probabilities dictionary
    propbalities_dict = config["LABELS_DICT"]

    for image in tqdm(images_to_process):
        # Defining colored image path
        image_output_path = os.path.join(output_dir, os.path.basename(image))
        try:
            # Reading the image
            img = cv2.imread(image)

            # Going to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Applying the model
            results = model.predict(image, conf=threshold)

            # Extracting all the masks
            # If mask empty apply infer_snow function
            if results[0].masks == None:
                mean_val = infer_snow(image, image_output_path, infer_box_padding)
            # Else apply closest mask
            else:
                masks = results[0].masks.xy

                # Calculating the center point of the image
                center_point = (img.shape[1] / 2, img.shape[0] / 2)

                # Getting closest mask
                closest_mask = get_closest_mask(center_point=center_point, masks=masks)

                # Get intersection polygon
                intersection_geom = get_intersection_geom(
                    closest_mask=closest_mask,
                    center_point=center_point,
                    padding=box_padding,
                )

                poly_coords = get_poly_coords(intersection_geom)

                if intersection_geom.area == 0:
                    mean_val = infer_snow(image, image_output_path, infer_box_padding)
                else:
                    # Draw poly on image
                    cv2.polylines(
                        img,
                        [poly_coords],
                        isClosed=False,
                        color=COLOR_DICT.get("red"),
                        thickness=5,
                    )

                    # Converting the image to grayscale
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    # Create a mask
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)

                    # Set mean value to 1 if no intersection between center bbox
                    # and closest mask was found
                    # Fill the polygon on the mask
                    cv2.fillPoly(mask, [poly_coords], 255)

                    # Apply the mask to the image
                    masked_image = cv2.bitwise_and(img_gray, img_gray, mask=mask)

                    # Calculate the mean pixel value
                    # Use mask to ignore zero pixels in the mean calculation

                    mean_val = cv2.mean(masked_image, mask=mask)

                    # Limiting the mean value to 0 - 1
                    mean_val = np.clip(mean_val[0] / 255, 0, 1)

                    # Rounding to 2 decimals
                    mean_val = round(mean_val, 2)

                    # Saving colored image
                    cv2.imwrite(image_output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Generating the image link
            image_link_colored = image_link_generator(
                image, storage_url=config["AZURE_INPUT"]["output_container_url"]
            )
            try:
                with open(file=image_output_path, mode="rb") as data:
                    _img = image.split(os.sep)
                    img_index = _img.index("images") + 1
                    blob_img_path = os.path.join(*_img[img_index:])
                    # Uploading
                    upload_image(
                        config=config,
                        colored_img_name=blob_img_path,
                        img_bytes=data,
                    )
            except Exception as e:
                print(e)

            # Generating the image link (original)
            image_link = image_link_generator(image)

            # Getting the metadata
            x, y = get_meta(image)

            # Crete Point geometry
            point_geom = str(Point(y, x))

            # Get closest inpection point from inspection_points list
            inspection_point_dict = get_closest(
                (x, y), intrest_points=inspection_points
            )

            img_datetime = get_img_datetime(image)

            img_datetime_format = f"{img_datetime[0:4]}-{img_datetime[5:7]}-{img_datetime[8:10]} {img_datetime[11:]}"

            # Label string value
            label = infer_label(mean_val, propbalities_dict)

            records.append(
                {
                    "OBJECTID": obj_id,
                    "image_name": blob_renamer(image),
                    "prediction_prob": mean_val,
                    "prediction_class": label,
                    "image_link_original": image_link,
                    "image_link_processed": image_link_colored,
                    "datetime_processed": str(datetime.now()),
                    "image_datetime": img_datetime_format,
                    "inspection_object": inspection_point_dict.get("object"),
                    "inspection_object_name": inspection_point_dict.get("object_name"),
                    "manager": inspection_point_dict.get("manager"),
                    "Shape": point_geom,
                }
            )

            # Incrementing the object id
            obj_id += 1

            # Removing the image from the local dir
            os.remove(image)

        except Exception as e:
            print(e)

    # Uploading the MSS database
    ImagePredictions.insert_records(records)


if __name__ == "__main__":
    pipeline(env="prod")
