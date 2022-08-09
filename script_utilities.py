import glob
import json
import os

from docrecjson import decoder
from docrecjson.elements import Document

from loguru import logger


def load_document(filepath: str) -> Document:
    with open(filepath) as json_data:
        json_annotation = json.load(json_data)

    return decoder.loads(json.dumps(json_annotation))


def get_ground_truth(filename: str, ground_truth_directory: str) -> Document:
    # todo add handling with getting ground truth from earlier versions
    # remove .png extension from conversion if present
    filename = filename.replace(".png", "")
    filename_without_extension = os.path.basename(os.path.splitext(filename)[0])
    glob_searchstring: str = os.path.join(ground_truth_directory, filename_without_extension) + ".*"
    matching_gt = []
    for filepath in glob.glob(glob_searchstring):
        matching_gt.append(filepath)

    if len(matching_gt) != 1:
        raise RuntimeError(
            "Expected number of matching annotation file for image file to be 1, actual number was: " + str(
                len(matching_gt)))
    else:
        logger.debug("Found matching annotation file for [" + filename + "]: [" + matching_gt[0] + "]")

    gt_annotation_path: str = matching_gt[0]
    return load_document(gt_annotation_path)


def get_image_file(filename: str, image_file_directory: str) -> str:
    filepath = os.path.join(image_file_directory, filename)
    if os.path.exists(filepath):
        return filepath
    else:
        raise RuntimeError("Expected " + filepath + " to be present.")
