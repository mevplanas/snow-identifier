# Geometry
from datetime import date
from shapely.geometry import Polygon
from shapely import wkt

# YOLO modeling 
from ultralytics import YOLO

# Array math
import numpy as np

# Computer vision
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

# Define TAGS
_TAGS_r = dict(((v, k) for k, v in TAGS.items()))

# Define image formats
IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Creating a dictionary of BGR colors
COLOR_DICT_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# Creating a dictionary of RGB colors
COLOR_DICT_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def calculate_centroid(polygon):
    """Calculate the centroid of a polygon."""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    centroid_x = sum(x_coords) / len(polygon)
    centroid_y = sum(y_coords) / len(polygon)
    return (centroid_x, centroid_y)


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_closest_mask(center_point, masks):
    # Calculating the centroids of the masks
    centroids = []
    for mask in masks:
        centroid = calculate_centroid(mask)
        centroids.append(centroid)

    # Getting the closest mask
    closest_mask = None
    closest_distance = None
    for mask, centroid in zip(masks, centroids):
        distance = calculate_distance(center_point, centroid)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_mask = mask
    closest_mask = closest_mask.reshape((-1, 1, 2))
    closest_mask = closest_mask.astype(int)
    return closest_mask


def get_intersection_geom(closest_mask, center_point, padding: int):
    # Get width and height coordinates from image center
    w, h = center_point

    # Create bbox around image center
    center_box = [w - padding, h - padding, w + padding, h + padding]
    center_box = [int(x) for x in center_box]

    # Convert bounding box into polygon
    center_bbox_shape = Polygon(
        [
            (center_box[0], center_box[1]),
            (center_box[2], center_box[1]),
            (center_box[2], center_box[3]),
            (center_box[0], center_box[3]),
        ]
    )

    # Convert numpy array to list
    closest_mask_list = closest_mask.tolist()
    # Wrangling list for polygon coversion
    line_struct = [x[0] for x in closest_mask_list]
    # Create closest mask polygon
    segment = Polygon(line_struct)

    # Get intersected polygon boundries between bbox and closest mask polygons
    intersection_geom = segment.intersection(center_bbox_shape)

    return intersection_geom


def get_poly(geom):
    x, y = geom.exterior.coords.xy
    # Create list of tuples with coordinates
    coord_pairs = list(zip(x, y))
    # Convert each tuple to list element
    coord_pairs = [list([int(y) for y in x]) for x in coord_pairs]
    # Convert list to numpy array
    intersection_poly = np.array(coord_pairs)
    return intersection_poly


def get_poly_coords(intersection):
    # Check geometry type
    if (
        intersection.geom_type == "GeometryCollection"
        or intersection.geom_type == "MultiPolygon"
    ):
        # If type geometry collection iterate until Polygon detected
        area = 0
        geom_holder = None
        for geom in intersection.geoms:
            if geom.geom_type == "Polygon":
                if area < geom.area:
                    area = geom.area
                    geom_holder = geom
                    continue
                else:
                    poly = get_poly(geom_holder)
                    return poly
            poly = get_poly(geom_holder)
            return poly

    elif intersection.geom_type == "LineString":
        return Polygon()
    else:
        poly = get_poly(intersection)
        return poly


def get_closest(input_point: tuple, intrest_points: list[dict]) -> dict:
    """
    The function gets closest point from inspection points.

    Arguments
    ---------
    input_point: tuple
        latitude and longitude coordinate of image center
    intrest_points: list[dict]
        inspection points

    Output
    ------
    closest_point: dict
        closest inspection point between image coordinates
    """

    # List where distance between image coordinates and inspection point be storded
    distances = []

    # Iterate through inspection points
    for _point in intrest_points:
        # Geometry value from dictionary
        str_geom = _point.get("Shape")
        # Convert string geometry into WKT (well know binary) geometry
        geom = wkt.loads(str_geom)
        # Get distance between inspection point and image coordinates
        distance = ((geom.y - input_point[0]) ** 2) + ((geom.x - input_point[1]) ** 2)
        # Append distance to list
        distances.append(distance)

    # Get minimum value from distance list
    min_distance = min(distances)

    # What index value with minimum distance
    index_value = distances.index(min_distance)

    # By minimum distance index value get inspection point dictionary
    closest_point = intrest_points[index_value]

    return closest_point


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

    # Converting the colored img to RGB
    colored_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Drawubg a rectangle on top
    colored_image = cv2.rectangle(
        colored_image,
        top_left,
        bottom_right,
        color=COLOR_DICT_BGR.get("red"),
        thickness=20,
    )

    # Saving colored image
    cv2.imwrite(output_path, colored_image)

    # Returning the probability
    return mean_pixel_value

def get_img_datetime(img_path: str):
    img = Image.open(img_path)

    # Get EXIF data
    exifd = img._getexif()

    # Getting keys of EXIF data
    keys = list(exifd.keys())

    # Remove MakerNote tag because these can be too long
    keys.remove(_TAGS_r["MakerNote"])

    # Get key values from img exif data
    keys = [k for k in keys if k in TAGS]

    # Timestamp
    date_time = exifd[_TAGS_r["DateTimeOriginal"]]

    return date_time

def predict_snow(
        img_path: str, 
        output_path: str,
        yolo_model: YOLO, 
        yolo_threshold: float = 0.5,
        center_padding: int = 25,
        ):
    try:
        # Reading the image
        img = cv2.imread(img_path)

        # Going to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Applying the model
        results = yolo_model.predict(img_path, conf=yolo_threshold, verbose=False)

        # Defining the initial mean value 
        mean_val = 0

        # Defining the initial boolean indicating the ML found an intersection
        intersection_found = False

        # Extracting all the masks
        # If mask empty apply infer_snow function
        if results[0].masks == None:
            mean_val = infer_snow(img_path, output_path)
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
                padding=center_padding,
            )

            poly_coords = get_poly_coords(intersection_geom)

            # Check if any intersection return
            if intersection_geom.area == 0:
                # If not calculate snow avg value with infer function
                mean_val = infer_snow(img_path, output_path)
            else:
                # If intersection found set boolean to True
                intersection_found = True

                # Draw poly on image
                cv2.polylines(
                    img,
                    [poly_coords],
                    isClosed=False,
                    color=COLOR_DICT_RGB.get("red"),
                    thickness=20,
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
                cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Returning the mean value 
        return mean_val, intersection_found
            
    except Exception as e:
        print(e)
