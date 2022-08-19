from docrecjson.elements import Document, PolygonRegion, Cell, Table
import python.evaluations.utility as utility

from shapely.geometry import Polygon
from loguru import logger

# todo check issues with shapely-speed ->
#  https://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely
from typing import List, Dict


def iou_f1_score_with_threshold(threshold: float, tables_gt: List[Table],
                                tables_prediction: List[Table]) -> float:
    # Go through each Ground Truth Cell -> find if IOU of this gt cell with some prediction cell is bigger than threshold
    # if len(tables_gt) > 1 or len(tables_prediction) > 1:
    # todo resolve this issue
    # create list of polygon regions for each table -> match them against each other?
    # -> how to match both tables against each other?
    # -> maybe with a certainity on how much they resemble?
    # raise RuntimeError("Specified table list of length longer than 1. This is currently not supported.")

    if len(tables_prediction) == 0:
        # no table was detected
        return 0

    table_gt: Table = tables_gt[0]
    table_prediction: Table = tables_prediction[0]

    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    for cell in table_gt.cells:
        iogt: float = _iogt_for_single_cell(cell, table_prediction.cells)
        if iogt >= threshold:
            true_positives += 1
        else:
            false_negatives += 1

    for cell in table_prediction.cells:
        iou: float = _iou_for_single_cell(cell, table_gt.cells)
        if iou < threshold:
            false_positives += 1

    if true_positives > 0 and len(table_gt.cells) > 0:
        precision: float = true_positives / len(table_gt.cells)
        logger.debug("precision: " + str(precision))
        recall: float = true_positives / (true_positives + false_negatives)
        logger.debug("recall: " + str(recall))
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        precision: float = 0
        recall: float = 0
        f1_score: float = 0

    return f1_score


def intersection_over_union(doc_gt: Document = None, doc_prediction: Document = None,
                            cells_gt: List[Cell] = None, cells_prediction: List[Cell] = None,
                            tables_gt: List[Table] = None, tables_prediction: List[Table] = None) -> float:
    if doc_gt is not None and doc_prediction is not None:
        polygon_content_gt = [x for x in doc_gt.content if isinstance(x, PolygonRegion)]
        polygon_content_prediction = [x for x in doc_prediction.content if isinstance(x, PolygonRegion)]
        return _intersection_over_union_polygon_region(polygon_content_gt, polygon_content_prediction)
    elif cells_gt is not None and cells_prediction is not None:
        polygon_content_gt = [x.bounding_box for x in cells_gt]
        polygon_content_prediction = [x.bounding_box for x in cells_prediction]
        return _intersection_over_union_polygon_region(polygon_content_gt, polygon_content_prediction)
    elif tables_gt is not None and tables_prediction is not None:
        if len(tables_gt) == 1 and len(tables_prediction) == 0:
            # special case where the table from the ground truth was not detected by the prediction
            return 0

        matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)
        iou_total_value: int = 0
        total_tables_viewed: int = 0
        for table_gt, table_prediction in matched_tables.items():
            cells_gt: List[Cell] = table_gt.cells
            cells_prediction: List[Cell] = table_prediction.cells
            polygon_content_gt = [x.bounding_box for x in cells_gt]
            polygon_content_prediction = [x.bounding_box for x in cells_prediction]

            iou_total_value += _intersection_over_union_polygon_region(polygon_content_gt, polygon_content_prediction)

        return iou_total_value / total_tables_viewed
    else:
        raise RuntimeError("Wrong Arguments for intersection over union calculation. Please review the required args.")


def _iou_for_single_cell(cell: Cell, cells_to_search: List[Cell]) -> float:
    for search_cell in cells_to_search:
        search_cell_area: Polygon = Polygon(search_cell.bounding_box.polygon)
        cell_area: Polygon = Polygon(cell.bounding_box.polygon)

        if search_cell_area.intersects(cell_area):
            intersection = search_cell_area.intersection(cell_area)
            return intersection.area / (search_cell_area.area + cell_area.area - intersection.area)

    return 0


def _iogt_for_single_cell(cell: Cell, cells_to_search: List[Cell]) -> float:
    for search_cell in cells_to_search:
        search_cell_area: Polygon = Polygon(search_cell.bounding_box.polygon)
        cell_area: Polygon = Polygon(cell.bounding_box.polygon)

        if search_cell_area.intersects(cell_area):
            intersection = search_cell_area.intersection(cell_area)
            return intersection.area / cell_area.area

    return 0


def _intersection_over_union_polygon_region(polygon_content_gt: List[PolygonRegion],
                                            polygon_content_prediction: List[PolygonRegion]):
    elements_iou_considered: int = 0
    prediction_elements_viewed: set = set()
    total_iou: float = 0
    for area_gt in polygon_content_gt:
        area_gt_polygon: Polygon = Polygon(area_gt.polygon)
        intersecting_element_found: bool = False

        for area_prediction in polygon_content_prediction:
            area_prediction_polygon: Polygon = Polygon(area_prediction.polygon)

            if area_gt_polygon.intersects(area_prediction_polygon):
                intersection = area_gt_polygon.intersection(area_prediction_polygon)

                logger.debug("Size of the ground truth: " + str(area_gt_polygon.area))
                logger.debug("Size of the prediction element: " + str(area_prediction_polygon.area))
                logger.debug("Size of the intersection of those polygons: " + str(intersection.area))

                elements_iou_considered += 1
                intersecting_element_found = True
                prediction_elements_viewed.add(area_prediction.oid)

                if not intersection.area == area_gt_polygon.area:
                    total_iou += intersection.area / (
                            area_gt_polygon.area + area_prediction_polygon.area - intersection.area)
                else:
                    total_iou += 1

        if not intersecting_element_found:
            total_iou += 0
            elements_iou_considered += 1
    for area_prediction in polygon_content_prediction:
        if area_prediction.oid not in prediction_elements_viewed:
            total_iou += 0
            elements_iou_considered += 1
    return total_iou / elements_iou_considered


def cell_intersection_over_union(cell_1: PolygonRegion, cell_2: PolygonRegion):
    polygon_1: Polygon = Polygon(cell_1.polygon)
    polygon_2: Polygon = Polygon(cell_2.polygon)

    intersection = polygon_1.intersection(polygon_2)

    if not intersection.area == polygon_1.area:
        return intersection.area / (polygon_1.area + polygon_2.area - intersection.area)
    else:
        return 0
