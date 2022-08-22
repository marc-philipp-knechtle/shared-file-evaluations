from copy import copy

import nltk
from docrecjson.elements import Cell, Table, Revision, Document
from typing import overload, List, Dict

import python.evaluations.utility as utility


def _levenshtein_distance_total(cell_a: Cell, cell_b: Cell) -> int:
    distance = nltk.edit_distance(cell_a.text_content.text, cell_b.text_content.text)
    return distance


def _levenshtein_distance_cell(cell_a: Cell, cell_b: Cell) -> float:
    if cell_a.text_content is None or cell_b.text_content is None:
        return 0
    normalized_distance = 1 - (_levenshtein_distance_total(cell_a, cell_b) / max(len(cell_a.text_content.text),
                                                                                 len(cell_b.text_content.text)))
    return normalized_distance


def _levenshtein_distance_cell_list(cell_a: Cell, cells: List[Cell]) -> float:
    matching_cell = utility.find_cell_with_highest_intersection_area(cell_a, cells)
    if matching_cell is None:
        return 0
    return _levenshtein_distance_cell(cell_a, matching_cell)


def _levenshtein_distance_table(table_a: Table, table_b: Table) -> float:
    cells_viewed: int = 0
    total_levenshtein_distance_relative: float = 0
    search_cells: List[Cell] = copy(table_b.cells)
    for cell in table_a.cells:
        matching_cell = utility.find_cell_with_highest_intersection_area(cell, search_cells)
        if matching_cell is not None:
            total_levenshtein_distance_relative += levenshtein_distance(cell, matching_cell)
            search_cells.remove(matching_cell)
        cells_viewed += 1

    # add relative value for each cell which was not matched
    total_levenshtein_distance_relative += len(search_cells)
    cells_viewed += len(search_cells)

    return total_levenshtein_distance_relative / cells_viewed if cells_viewed > 0 else 0


def _levenshtein_distance_document(doc_gt: Document, revision_prediction: Revision) -> float:
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in copy(revision_prediction.objects) if isinstance(x, Table)]

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)
    tables_viewed: int = 0
    total_levenshtein_distance: float = 0

    for table_gt, table_prediction in matched_tables.items():
        if table_prediction is not None:
            total_levenshtein_distance += levenshtein_distance(table_gt, table_prediction)
            tables_prediction.remove(table_prediction)
        tables_viewed += 1

    tables_viewed += len(tables_prediction)
    total_levenshtein_distance += len(tables_prediction)

    return total_levenshtein_distance / tables_viewed if tables_viewed != 0 else 0


def ld_precision(threshold: float, doc_gt: Document, revision_prediction: Revision) -> float:
    """
    precision = true positives / (true positives + false positives)
    precision = (number of cells from gt matched with cell from prediction with ld > threshold)/
    [("")  + (cells from prediction which don't find cells from the gt with fpa > threshold)]

    :param threshold:
    :param doc_gt:
    :param revision_prediction:
    :return:
    """
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in revision_prediction.objects if isinstance(x, Table)]
    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)

    true_positives: int = 0
    false_positives: int = 0

    for table_gt, table_prediction in matched_tables.items():
        cells_prediction: List[Cell] = table_prediction.cells if table_prediction is not None else []
        cells_gt: List[Cell] = table_gt.cells

        cell_gt: Cell
        for cell_gt in cells_gt:
            ld: float = _levenshtein_distance_cell_list(cell_gt, cells_prediction)
            if ld >= threshold:
                true_positives += 1

        for cell_prediction in cells_prediction:
            ld: float = _levenshtein_distance_cell_list(cell_prediction, cells_gt)
            if ld < threshold:
                false_positives += 1
    return true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0


def ld_recall(threshold: float, doc_gt: Document, revision_prediction: Revision):
    """
    recall = true positives / (true positives + false negatives)
    recall = (number of cells with fpa bigger than threshold) /
    [("") + (cells from gt which don't find cell from prediction with ld > threshold)]

    :param threshold:
    :param doc_gt:
    :param revision_prediction:
    :return:
    """
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in revision_prediction.objects if isinstance(x, Table)]

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)

    true_positives: int = 0
    false_negatives: int = 0

    for table_gt, table_prediction in matched_tables.items():
        cells_prediction: List[Cell] = table_prediction.cells if table_prediction is not None else []
        cells_gt: List[Cell] = table_gt.cells

        cell_gt: Cell
        for cell_gt in cells_gt:
            fpa: float = _levenshtein_distance_cell_list(cell_gt, cells_prediction)
            if fpa >= threshold:
                true_positives += 1
            else:
                false_negatives += 1

    return true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0


def ld_f1(threshold: float, doc_gt: Document, revision_prediction: Revision):
    precision: float = ld_precision(threshold, doc_gt, revision_prediction)
    recall: float = ld_recall(threshold, doc_gt, revision_prediction)
    return float(2 * ((precision * recall) / (precision + recall))) if precision + recall > 0 else 0


@overload
def levenshtein_distance(cell_a: Cell, cell_b: Cell):
    ...


@overload
def levenshtein_distance(cell_a: Cell, cells: List[Cell]):
    ...


@overload
def levenshtein_distance(table_a: Table, table_b: Table):
    ...


@overload
def levenshtein_distance(doc_gt: Document, revision_prediction: Revision):
    ...


def levenshtein_distance(param_a, param_b) -> float:
    if isinstance(param_a, Document):
        return _levenshtein_distance_document(param_a, param_b)
    elif isinstance(param_b, Table):
        return _levenshtein_distance_table(param_a, param_b)
    elif isinstance(param_a, Cell) and isinstance(param_b, Cell):
        return _levenshtein_distance_cell(param_a, param_b)
    elif isinstance(param_a, Cell) and isinstance(param_b, List):
        return _levenshtein_distance_cell_list(param_a, param_b)
