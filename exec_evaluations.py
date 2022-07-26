#!/usr/bin/env python3
"""
This script is intended to be an easy entrypoint for executing all metrics written in the evaluations package.
It's intended to be executed over a whole directory of annotations together with their ground truth
"""
import argparse
import glob
import json
import os.path
import sys

from typing import List

import python.evaluations.iou as iou

from docrecjson import decoder
from docrecjson.elements import Document, Revision, Table

from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="INFO")

IOU: str = "iou"


def _load_document(filepath: str) -> Document:
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
    return _load_document(gt_annotation_path)


def _process_prediction_directory(prediction_directory: str, ground_truth_directory: str):
    files_considered: int = 0
    # this list is intended for average iou computation. Each index represents the summed revision.
    metrics: dict = {IOU: {}}

    for filepath in tqdm(glob.glob(os.path.join(prediction_directory, "*"))):

        filename: str = os.path.basename(filepath)
        if filename.startswith('.'):
            logger.info("Ignoring [" + str(filename) + "] because it's hidden.")
            continue
        logger.debug("[" + filename + "]")

        prediction: Document = _load_document(filepath)
        ground_truth: Document = get_ground_truth(filename, ground_truth_directory)

        if prediction.revisions is not None:
            for revision_index in range(len(prediction.revisions)):
                _process_revision(ground_truth, metrics, prediction, revision_index)

            if len(prediction.revisions) != len(metrics[IOU]):
                raise RuntimeError("Mismatching revision sum and total revision dictionary.")
        else:
            tables_prediction: List[Table] = [x for x in prediction.objects() if isinstance(x, Table)]
            tables_gt: List[Table] = [x for x in ground_truth.objects() if isinstance(x, Table)]
            iou_value: float = iou.intersection_over_union(tables_gt=tables_gt,
                                                           tables_prediction=tables_prediction)
            logger.debug("IoU  value: " + str(iou_value))
            metrics["no revision"] = float(
                metrics["no revision"]) + iou_value if metrics.get(
                "no revision") is not None else iou_value

        files_considered += 1

    logger.info("Computed evaluations for [" + str(
        files_considered) + "] files in [" + prediction_directory + "] with gt: [" + ground_truth_directory + "]")
    logger.info("Found results for multiple revisions in the prediction files: ")

    logger.info("Found the following IoU values for the revisions:")
    for key, value in metrics[IOU].items():
        logger.info(key + ": " + str(float(value) / files_considered))


def _process_revision(ground_truth: Document, metrics: dict, prediction: Document, revision_index: int) -> dict:
    prediction.select_revision(revision_index)
    revision: Revision = prediction.revisions[revision_index]
    revision_name: str = 'revision:' + str(revision_index) + ':' + revision.name if revision.name is not None else ""
    tables_prediction: List[Table] = [x for x in prediction.objects() if isinstance(x, Table)]
    tables_gt: List[Table] = [x for x in ground_truth.objects() if isinstance(x, Table)]

    iou_value: float = iou.intersection_over_union(tables_gt=tables_gt, tables_prediction=tables_prediction)

    metrics[IOU][revision_name] = float(
        metrics[IOU][revision_name]) + iou_value if metrics[IOU].get(
        revision_name) is not None else iou_value
    logger.debug("IoU value of " + 'revision:' + str(revision_index) + ':' + revision_name + ": " + str(iou_value))

    return metrics


def process_prediction_file(prediction_file: str, ground_truth_directory: str):
    filename: str = os.path.basename(prediction_file)
    logger.info("[" + filename + "]")

    prediction: Document = _load_document(prediction_file)
    ground_truth: Document = get_ground_truth(filename, ground_truth_directory)

    intersection_over_union: float = iou.intersection_over_union(ground_truth, prediction)

    logger.info("Intersection over Union: " + str(intersection_over_union))


def main(prediction_file: str, prediction_directory: str, ground_truth_directory: str):
    if not prediction_directory == "":
        logger.info("Prediction directory: " + prediction_directory)
        logger.info("Ground Truth directory: " + ground_truth_directory)
        _process_prediction_directory(prediction_directory, ground_truth_directory)
    elif not prediction_file == "":
        process_prediction_file(prediction_file, ground_truth_directory)
    else:
        raise RuntimeError("No prediction_file or prediction_directory was specified!")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--prediction_file", type=str, required=False,
                        help="Specify a prediction file. "
                             "You have to specify a prediction file or an prediction directory.",
                        default="")
    parser.add_argument("-p", "--prediction_directory", type=str,
                        required=False,
                        help="Specify the directory where the prediction json files are in. "
                             "You have to specify a prediction file or a prediction directory.",
                        default="")
    parser.add_argument("-g", "--ground_truth_directory", type=str, required=False,
                        help="This is the directory with the ground_truth files. "
                             "It's necessary to have ground truth information to compute the evaluation metrics. "
                             "If you leave this empty, "
                             "this application will take the earliest version from the predictions as ground truth.",
                        default="")
    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args.prediction_file, args.prediction_directory, args.ground_truth_directory)
