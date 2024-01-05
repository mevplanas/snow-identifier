# Importing OS
import os

# Datetime wrangling
from datetime import datetime

# YAML reading
import yaml

# Iteration tracking
from tqdm import tqdm

# Azure Storage Blobs
from src.azure_blobs import download_image, get_blobs_by_folder_name, upload_image

# Utilis
from src.custom_utils import image_link_generator, infer_label, blob_renamer

# Image processing
from src.image_processing import (
    get_closest,
    get_meta,
    get_img_datetime,
    predict_snow,
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
            config=config,
            name_starts_with=prefix,
        )

        # Downloading images from Azure storage to local dir
        for img in tqdm(storage_images, desc="Downloading images from Azure Storage:"):
            download_image(
                blob_name=img, config=config, local_file_dir=images_local_dir
            )

        for root, _, files in os.walk(images_local_dir):
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

    # Iterating over the images 
    for image in tqdm(images_to_process):
        # Defining the full output path 
        image_output_path = os.path.join(output_dir, os.path.basename(image))

        # Applying the infer snow function 
        mean_val, intersection_found = predict_snow(
            img_path=image, 
            output_path=image_output_path, 
            yolo_model=model, 
            yolo_threshold=threshold, 
            center_padding=box_padding,
            )

        # Generating the image link
        image_link_colored = image_link_generator(
            image, storage_url=config["AZURE_INPUT"]["output_container_url"]
        )
        try:
            # Uploading image to Azure Storage
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

        # Get image timestamp
        img_datetime = get_img_datetime(image)

        # Format timestamp
        img_datetime_format = f"{img_datetime[0:4]}-{img_datetime[5:7]}-{img_datetime[8:10]} {img_datetime[11:]}"

        # Label string value
        label = infer_label(mean_val, propbalities_dict)

        # If the label is 'snow' but the intersection was not found
        # we rename the label to 'maybe_snow'
        if label == "snow" and not intersection_found:
            label = "maybe_snow"

        # Append prediction and other information to list
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

    # Uploading the MSS database
    ImagePredictions.insert_records(records)

if __name__ == "__main__":
    pipeline(env="prod")
