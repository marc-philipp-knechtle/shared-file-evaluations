import shapely
from docrecjson.elements import Cell, Document, Revision, Table
from typing import List, Dict

from shapely.geometry import Polygon, mapping, MultiPoint

import python.evaluations.utility as utility

import numpy as np

from PIL import Image

from loguru import logger


def _foreground_pixel_accuracy_for_single_cell(gt_cell: Cell, cells_to_search: List[Cell], image: Image) -> float:
    """

    :param: gt_cell: ground truth cell
    :param: cells_to_search: prediction cells
    :param: image: image file for the annotation and ground truth
    :return: the share of the black pixels which are identical in gt and prediction
    """

    prediction_cell: Cell = utility.find_cell_with_highest_intersection_area(gt_cell, cells_to_search)
    if prediction_cell is None:
        return 0.0
    gt_cell_area: Polygon = Polygon(gt_cell.bounding_box.polygon)
    prediction_cell_area: Polygon = Polygon(prediction_cell.bounding_box.polygon)

    # min = upper left coordinate
    # max = lower right coordinate
    x_min, y_min, x_max, y_max = gt_cell_area.bounds
    x = np.arange(np.floor(x_min), np.ceil(x_max), 1)  # returns all values between min and max spaced with 1
    y = np.arange(np.floor(y_min), np.ceil(y_max), 1)

    # create matrix with bounds of the polygon element
    point_matrix = [np.tile(x, len(y)), np.repeat(y, len(x))]
    # create a shapely multipoint object
    points_matrix = MultiPoint(np.transpose(point_matrix))
    points_inside_cell: MultiPoint = points_matrix.intersection(gt_cell_area)

    gt_pixels: int = 0
    prediction_pixels: int = 0
    pixels = image.load()
    coord: shapely.geometry.Point
    logger.enable("python.evaluations.foreground_pixel_accuracy")
    for coord in points_inside_cell.geoms:
        try:
            rgb = pixels[coord.x, coord.y]
        except IndexError:
            # This is because of the structure of the SciTSR dataset
            # For some reason, some indexes are out of the original image file.
            # You can view those files easily by viewing the dataset in fiftyone
            logger.warning(
                "Got coordinates (" + str(coord.x) + ", " + str(coord.y) + ") for Foreground Pixel Accuracy "
                                                                           "which are out of the bounds of the image.")
            logger.info("Further log messages for this cell are disabled.")
            logger.disable("python.evaluations.foreground_pixel_accuracy")
            continue
        if rgb != (0, 0, 0):
            # print(str(coord) + ":" + str(pixels[coord.x, coord.y]))  # todo does pixels[x,y] work from the top or the bottom?
            point_area: shapely.geometry.point = shapely.geometry.Point(coord.x, coord.y)
            gt_pixels += 1
            if point_area.intersects(prediction_cell_area):
                prediction_pixels += 1

    return prediction_pixels / gt_pixels if gt_pixels > 0 else 0


def foreground_pixel_accuracy(doc_gt: Document, revision_prediction: Revision, image_filepath: str) -> float:
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in revision_prediction.objects if isinstance(x, Table)]
    image: Image = Image.open(image_filepath)

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)

    foreground_pixel_accuracy_total_value: float = 0
    total_cells_viewed: int = 0
    for table_gt, table_prediction in matched_tables.items():
        cells_prediction: List[Cell] = table_prediction.cells if table_prediction is not None else []
        cell_gt: Cell
        for cell_gt in table_gt.cells:
            foreground_pixel_accuracy_total_value += _foreground_pixel_accuracy_for_single_cell(cell_gt,
                                                                                                cells_prediction, image)
            total_cells_viewed += 1

    return foreground_pixel_accuracy_total_value / total_cells_viewed


def fpa_precision(threshold: float, doc_gt: Document, revision_prediction: Revision, image_filepath: str) -> float:
    """
    precision = true positives / (true positives + false positives)
    precision = (number of cells with fpa bigger than threshold) /
    [("") + (cells from prediction which don't find cells from the gt with fpa > threshold)]

    :param threshold:
    :param doc_gt:
    :param revision_prediction:
    :param image_filepath:
    :return:
    """

    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in revision_prediction.objects if isinstance(x, Table)]
    image: Image = Image.open(image_filepath)

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)

    true_positives: int = 0
    false_positives: int = 0

    for table_gt, table_prediction in matched_tables.items():
        cells_prediction: List[Cell] = table_prediction.cells if table_prediction is not None else []
        cells_gt: List[Cell] = table_gt.cells

        cell_gt: Cell
        for cell_gt in cells_gt:
            fpa: float = _foreground_pixel_accuracy_for_single_cell(cell_gt, cells_prediction, image)
            if fpa >= threshold:
                true_positives += 1

        for cell_prediction in cells_prediction:
            fpa: float = _foreground_pixel_accuracy_for_single_cell(cell_prediction, cells_gt, image)
            if fpa < threshold:
                false_positives += 1

    return true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0


def fpa_recall(threshold: float, doc_gt: Document, revision_prediction: Revision, image_filepath: str):
    """
    recall = true positives / (true positives + false negatives)
    recall = (number of cells with fpa bigger than threshold) /
    [("") + (cells from gt which don't find cell from prediction with fpa > threshold)]

    :param threshold:
    :param doc_gt:
    :param revision_prediction:
    :param image_filepath:
    :return:
    """

    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in revision_prediction.objects if isinstance(x, Table)]
    image: Image = Image.open(image_filepath)

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)

    true_positives: int = 0
    false_negatives: int = 0

    for table_gt, table_prediction in matched_tables.items():
        cells_prediction: List[Cell] = table_prediction.cells if table_prediction is not None else []
        cells_gt: List[Cell] = table_gt.cells

        cell_gt: Cell
        for cell_gt in cells_gt:
            fpa: float = _foreground_pixel_accuracy_for_single_cell(cell_gt, cells_prediction, image)
            if fpa >= threshold:
                true_positives += 1
            else:
                false_negatives += 1

    return true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0


def fpa_f1_score(threshold: float, doc_gt: Document, revision_prediction: Revision, image_filepath: str) -> float:
    """
    f1_score = 2*[(precision*recall)/(precision + recall)]

    :param threshold:
    :param doc_gt:
    :param revision_prediction:
    :param image_filepath:
    :return:
    """

    precision: float = fpa_precision(threshold, doc_gt, revision_prediction, image_filepath)
    recall: float = fpa_recall(threshold, doc_gt, revision_prediction, image_filepath)

    return float(2 * ((precision * recall) / (precision + recall))) if precision + recall > 0 else 0
