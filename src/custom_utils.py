# Regex
import re

# System
from sys import platform


def image_link_generator(
    file_path: str,
    image_dir_prefix: str = "images",
    storage_url: str = "https://vp3ddata.blob.core.windows.net/snow-identifier-input/",
) -> str:
    if platform == "win32":
        # Creating the target name
        _target_name = file_path.split(f"{image_dir_prefix}\\")[-1]
        target_name = re.sub("\\\\", "/", _target_name)
    else:
        _target_name = file_path.split(f"{image_dir_prefix}/")[-1]
        target_name = re.sub("//|/", "/", _target_name)

    blob_url = f"{storage_url}{target_name}"

    return blob_url


def infer_label(prob: float, prob_dict: dict) -> str:
    label = "maybe_snow"

    if not isinstance(prob, float):
        return label

    if prob < prob_dict.get("no_snow", 0.5):
        label = "no snow"
    elif prob < prob_dict.get("maybe_snow", 0.75):
        label = "maybe_snow"
    elif prob < prob_dict.get("snow", 1.0):
        label = "snow"
    else:
        label = "maybe_snow"

    return label


def blob_renamer(full_path: str) -> str:
    """
    The blob
    """

    if platform == "win32":
        # Creating the target name
        target_name = full_path.split("images\\")[-1]
        target_name = re.sub("\\\\", "/", target_name)
    else:
        target_name = full_path.split("images/")[-1]
        target_name = re.sub("//|/", "/", target_name)

    return target_name
