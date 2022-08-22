#!/usr/bin/env python3
"""
This script is intended to be an easy entrypoint for executing all metrics written in the evaluations package.
It's intended to be executed over a whole directory of annotations together with their ground truth
"""
import argparse
import glob
import os.path
import sys

from typing import List, Optional

import python.evaluations.iou as iou
import python.evaluations.foreground_pixel_accuracy as foreground_pixel_accuracy
import python.evaluations.levenshtein_distance as ld

from docrecjson.elements import Document, Revision, Table

from loguru import logger
from tqdm import tqdm

import script_utilities

logger.remove()
logger.add(sys.stderr, level="INFO")

IOU: str = "iou"
IOU_F1_60 = "iou_f1_60"
IOU_F1_70 = "iou_f1_70"
IOU_F1_80 = "iou_f1_80"
IOU_F1_90 = "iou_f1_90"
IOU_F1_THRESHOLDS = [IOU_F1_60, IOU_F1_70, IOU_F1_80, IOU_F1_90]

FOREGROUND_PIXEL_ACCURACY: str = "foreground_pixel_accuracy"
FPA_F1_60 = "fpa_f1_60"
FPA_F1_70 = "fpa_f1_70"
FPA_F1_80 = "fpa_f1_80"
FPA_F1_90 = "fpa_f1_90"
FPA_F1_THRESHOLDS = [FPA_F1_60, FPA_F1_70, FPA_F1_80, FPA_F1_90]

LEVENSHTEIN_DISTANCE: str = "levenshtein_distance"
LD_F1_60 = "ld_f1_60"
LD_F1_70 = "ld_f1_70"
LD_F1_80 = "ld_f1_80"
LD_F1_90 = "ld_f1_90"
LD_F1_THRESHOLDS = [LD_F1_60, LD_F1_70, LD_F1_80, LD_F1_90]


def _output_metrics(files_considered: int, metrics: dict):
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Found the following average IoU values for the revisions:")
    for key, value in metrics[IOU].items():
        logger.info(key + ": " + str(float(value) / files_considered))
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Found the following IoU F1 Scores with their respective Thresholds.")
    for threshold_key in IOU_F1_THRESHOLDS:
        for key, value in metrics[threshold_key].items():
            logger.info(str(threshold_key) + ": " + key + ": " + str(float(value) / files_considered))
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Found the following Foreground Pixel Accuracy Scores:")
    for key, value in metrics[FOREGROUND_PIXEL_ACCURACY].items():
        logger.info(key + ": " + str(float(value) / files_considered))
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Found the following FPA F1 Scores with their respective Thresholds.")
    for threshold_key in FPA_F1_THRESHOLDS:
        for key, value in metrics[threshold_key].items():
            logger.info(str(threshold_key) + ": " + key + ": " + str(float(value) / files_considered))
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Found the following Levenshtein Distance Scores:")
    for key, value in metrics[LEVENSHTEIN_DISTANCE].items():
        logger.info(key + ": " + str(float(value) / files_considered))
    logger.info("-----------------------------------------------------------------------------------------------------")
    logger.info("Found the following Levenshtein Distance F1 Scores")
    for threshold_key in LD_F1_THRESHOLDS:
        for key, value in metrics[threshold_key].items():
            logger.info(str(threshold_key) + ": " + key + ": " + str(float(value) / files_considered))


def _process_revision(ground_truth: Document, metrics: dict, prediction: Document, revision_index: int,
                      image_directory: Optional[str]) -> dict:
    prediction.select_revision(revision_index)
    revision: Revision = prediction.revisions[revision_index]
    revision_name: str = 'revision:' + str(revision_index) + ':' + revision.name if revision.name is not None else ""
    tables_prediction: List[Table] = [x for x in prediction.objects() if isinstance(x, Table)]
    tables_gt: List[Table] = [x for x in ground_truth.objects() if isinstance(x, Table)]

    # add fpa metrics
    if image_directory is not None:
        image_filepath: str = script_utilities.get_image_file(prediction.filename, image_directory)
        fpa = foreground_pixel_accuracy.foreground_pixel_accuracy(ground_truth, revision, image_filepath)

        fpa_f1_60: float = foreground_pixel_accuracy.fpa_f1_score(0.6, ground_truth, revision, image_filepath)
        fpa_f1_70: float = foreground_pixel_accuracy.fpa_f1_score(0.7, ground_truth, revision, image_filepath)
        fpa_f1_80: float = foreground_pixel_accuracy.fpa_f1_score(0.8, ground_truth, revision, image_filepath)
        fpa_f1_90: float = foreground_pixel_accuracy.fpa_f1_score(0.9, ground_truth, revision, image_filepath)
        fpa_f1_values: List[float] = [fpa_f1_60, fpa_f1_70, fpa_f1_80, fpa_f1_90]

        metrics[FOREGROUND_PIXEL_ACCURACY][revision_name] = float(
            metrics[FOREGROUND_PIXEL_ACCURACY][revision_name]) + fpa if metrics[FOREGROUND_PIXEL_ACCURACY].get(
            revision_name) is not None else fpa

        for threshold_key, value in zip(FPA_F1_THRESHOLDS, fpa_f1_values):
            metrics[threshold_key][revision_name] = \
                float(metrics[threshold_key][revision_name]) + value if metrics[threshold_key].get(
                    revision_name) is not None else value

    # add iou metrics
    iou_value: float = iou.intersection_over_union(tables_gt=tables_gt, tables_prediction=tables_prediction)
    metrics[IOU][revision_name] = float(metrics[IOU][revision_name]) + iou_value if metrics[IOU].get(
        revision_name) is not None else iou_value

    iou_f1_60: float = iou.iou_f1_score_with_threshold(0.6, tables_gt, tables_prediction)
    iou_f1_70: float = iou.iou_f1_score_with_threshold(0.7, tables_gt, tables_prediction)
    iou_f1_80: float = iou.iou_f1_score_with_threshold(0.8, tables_gt, tables_prediction)
    iou_f1_90: float = iou.iou_f1_score_with_threshold(0.9, tables_gt, tables_prediction)
    iou_f1_values: List[float] = [iou_f1_60, iou_f1_70, iou_f1_80, iou_f1_90]

    for threshold_key, value in zip(IOU_F1_THRESHOLDS, iou_f1_values):
        metrics[threshold_key][revision_name] = \
            float(metrics[threshold_key][revision_name]) + value if metrics[threshold_key].get(
                revision_name) is not None else value

    # add levenshtein metrics
    levenshtein_distance: float = ld.levenshtein_distance(ground_truth, revision)
    metrics[LEVENSHTEIN_DISTANCE][revision_name] = float(
        metrics[LEVENSHTEIN_DISTANCE][revision_name]) + levenshtein_distance if metrics[LEVENSHTEIN_DISTANCE].get(
        revision_name) is not None else levenshtein_distance

    ld_f1_60: float = ld.ld_f1(0.6, ground_truth, revision)
    ld_f1_70: float = ld.ld_f1(0.7, ground_truth, revision)
    ld_f1_80: float = ld.ld_f1(0.8, ground_truth, revision)
    ld_f1_90: float = ld.ld_f1(0.9, ground_truth, revision)
    ld_f1_values: List[float] = [ld_f1_60, ld_f1_70, ld_f1_80, ld_f1_90]

    for threshold_key, value in zip(LD_F1_THRESHOLDS, ld_f1_values):
        metrics[threshold_key][revision_name] = \
            float(metrics[threshold_key][revision_name]) + value if metrics[threshold_key].get(
                revision_name) is not None else value

    return metrics


def _process_prediction_file(ground_truth: Document, metrics: dict, prediction: Document,
                             image_directory: Optional[str]):
    if prediction.revisions is not None:
        for revision_index in range(len(prediction.revisions)):
            _process_revision(ground_truth, metrics, prediction, revision_index, image_directory)

        if len(prediction.revisions) != len(metrics[IOU]):
            raise RuntimeError("Mismatching revision sum and total revision dictionary.")
    else:
        tables_prediction: List[Table] = [x for x in prediction.objects() if isinstance(x, Table)]
        tables_gt: List[Table] = [x for x in ground_truth.objects() if isinstance(x, Table)]
        iou_value: float = iou.intersection_over_union(tables_gt=tables_gt,
                                                       tables_prediction=tables_prediction)
        logger.debug("IoU  value: " + str(iou_value))
        metrics[IOU] = float(
            metrics["no revision"]) + iou_value if metrics.get(
            "no revision") is not None else iou_value

    return metrics


def _handle_prediction_directory(prediction_directory: str, ground_truth_directory: str,
                                 image_directory: Optional[str]):
    files_considered: int = 0
    # this list is intended for average iou computation. Each index represents the summed revision.
    metrics: dict = {IOU: {}, IOU_F1_60: {}, IOU_F1_70: {}, IOU_F1_80: {}, IOU_F1_90: {},
                     FOREGROUND_PIXEL_ACCURACY: {}, FPA_F1_60: {}, FPA_F1_70: {}, FPA_F1_80: {}, FPA_F1_90: {},
                     LEVENSHTEIN_DISTANCE: {}, LD_F1_60: {}, LD_F1_70: {}, LD_F1_80: {}, LD_F1_90: {}}

    for filepath in tqdm(glob.glob(os.path.join(prediction_directory, "*"))):

        filename: str = os.path.basename(filepath)
        if filename.startswith('.'):
            logger.info("Ignoring [" + str(filename) + "] because it's hidden.")
            continue

        prediction: Document = script_utilities.load_document(filepath)
        ground_truth: Document = script_utilities.get_ground_truth(filename, ground_truth_directory)

        metrics = _process_prediction_file(ground_truth, metrics, prediction, image_directory)

        files_considered += 1
        if files_considered % 100 == 0:
            logger.info(
                "Outputting temporary average metrics after processing of " + str(files_considered) + " files: ")
            _output_metrics(files_considered, metrics)

    logger.info("Computed evaluations for [" + str(
        files_considered) + "] files in [" + prediction_directory + "] with gt: [" + ground_truth_directory + "]")
    logger.info("Found results for multiple revisions in the prediction files: ")

    logger.success("Computed average metrics for [" + str(files_considered) + "] files.")
    _output_metrics(files_considered, metrics)


def _handle_prediction_file(prediction_file: str, ground_truth_directory: str, image_directory: Optional[str]):
    filename: str = os.path.basename(prediction_file)
    logger.info("[" + filename + "]")

    prediction: Document = script_utilities.load_document(prediction_file)
    ground_truth: Document = script_utilities.get_ground_truth(filename, ground_truth_directory)

    metrics: dict = {IOU: {}, IOU_F1_60: {}, IOU_F1_70: {}, IOU_F1_80: {}, IOU_F1_90: {}, FOREGROUND_PIXEL_ACCURACY: {}}
    metrics = _process_prediction_file(ground_truth, metrics, prediction, image_directory)
    _output_metrics(1, metrics)


def main(prediction_file: str, prediction_directory: str, ground_truth_directory: str, image_directory: Optional[str]):
    if image_directory is not None:
        logger.info("Image directory: " + image_directory)
    if not prediction_directory == "":
        logger.info("Prediction directory: " + prediction_directory)
        logger.info("Ground Truth directory: " + ground_truth_directory)
        _handle_prediction_directory(prediction_directory, ground_truth_directory, image_directory)
    elif not prediction_file == "":
        _handle_prediction_file(prediction_file, ground_truth_directory, image_directory)
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
    parser.add_argument("-g", "--ground_truth_directory", type=str, required=True,
                        help="This is the directory with the ground_truth files. "
                             "It's necessary to have ground truth information to compute the evaluation metrics. "
                             "If you leave this empty, "
                             "this application will take the earliest version from the predictions as ground truth.")
    parser.add_argument("-i", "--image_directory", type=str, required=False,
                        help="Specify a directory which contains the images used for prediction and evaluation."
                             "This enables for additional metrics to be computed.",
                        default=None)
    # todo remove default "" values and replace with none
    # todo add check_args method, check that either prediction_file or prediction_directory is supplied
    # todo add log level as arguments like with the docrecJSON-converter
    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args.prediction_file, args.prediction_directory, args.ground_truth_directory, args.image_directory)

# todo rename to docrecJSON-evaluations
