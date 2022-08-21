from docrecjson.elements import Cell, Table
from typing import List, Optional, Dict

from shapely.geometry import Polygon

from loguru import logger
from shapely.validation import make_valid


def find_cell_with_highest_intersection_area(cell: Cell, cells_to_search: List[Cell]) -> Optional[Cell]:
    """
    returns cell from cells_to_search with the highest intersection to cell
    :param cell: base cell
    :param cells_to_search: list of cells to search a matching candidate
    :return: the matching cell or none if there is no intersecting cell in cells_to_search
    """
    cell_area: Polygon = Polygon(cell.bounding_box.polygon)

    cell_with_highest_intersection: Optional[Cell] = None
    highest_intersection: float = 0

    search_cell: Cell
    for search_cell in cells_to_search:
        search_cell_area: Polygon = Polygon(search_cell.bounding_box.polygon)
        if search_cell_area.intersects(cell_area):
            intersection = search_cell_area.intersection(cell_area)
            if intersection.area > highest_intersection:
                highest_intersection = intersection.area
                cell_with_highest_intersection = search_cell

    return cell_with_highest_intersection


def match_tables(tables_gt: List[Table], tables_prediction: List[Table]) -> Dict[Table, Table]:
    """
    :param: tables_gt: ground truth tables
    :param: tables_prediction: prediction tables
    :return: a dictionary with each gt table matched to the suiting prediction table
    """
    table_gt: Table
    matched_tables: Dict[Table, Table] = {}
    for table_gt in tables_gt:
        table_gt_area: Polygon = Polygon(table_gt.get_table_coordinates())
        prediction_table_with_highest_intersection: Optional[Table] = None
        highest_intersection: float = 0
        for table_prediction in tables_prediction:
            # todo remove table after if was matched to a gt table
            table_prediction_area: Polygon = Polygon(table_prediction.get_table_coordinates())
            if not table_prediction_area.is_valid:
                logger.warning("Getting invalid prediction table! Please review the order of the Polygon coordinates.")
                logger.warning("Trying to resolve the invalid polygon: ")
                table_prediction_area = make_valid(table_prediction_area)
            if table_gt_area.intersects(table_prediction_area):
                intersection = table_gt_area.intersection(table_prediction_area)
                if intersection.area > highest_intersection:
                    highest_intersection = intersection.area
                    prediction_table_with_highest_intersection = table_prediction

        matched_tables[table_gt] = prediction_table_with_highest_intersection

    return matched_tables
