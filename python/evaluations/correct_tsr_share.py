"""
This metric is very simple. It matches the table structure coordinates and checks whether they are correct.
It returns the share of the correct coordinates.
Each cell has four coordinates.
start_column, end_column, start_row, end_row
"""
from copy import copy

from docrecjson.elements import Revision, Document, Table, Cell
from typing import List, Dict, overload

import python.evaluations.utility as utility


def _correct_tsr_share_cell(cell_gt: Cell, cell_prediction: Cell) -> float:
    correct_identified_table_coordinates: int = 0
    if cell_prediction.start_column_index == cell_gt.start_column_index:
        correct_identified_table_coordinates += 1
    if cell_prediction.end_column_index == cell_gt.end_column_index:
        correct_identified_table_coordinates += 1
    if cell_prediction.start_row_index == cell_gt.start_row_index:
        correct_identified_table_coordinates += 1
    if cell_prediction.end_row_index == cell_gt.end_row_index:
        correct_identified_table_coordinates += 1
    return correct_identified_table_coordinates / 4


def _correct_tsr_share_cell_list(cell_a, cells: List[Cell]):
    matching_cell: Cell = utility.find_cell_with_highest_intersection_area(cell_a, cells)
    if matching_cell is None:
        return 0
    return _correct_tsr_share_cell(cell_a, matching_cell)


def _correct_tsr_share_table(table_a: Table, table_b: Table):
    total_correct_tsr_share: float = 0
    search_cells: List[Cell] = copy(table_b.cells)
    for cell in table_a.cells:
        total_correct_tsr_share += _correct_tsr_share_cell_list(cell, search_cells)

    return total_correct_tsr_share / len(table_a.cells)


def _correct_tsr_share_document(doc_gt: Document, revision_prediction: Revision) -> float:
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in copy(revision_prediction.objects) if isinstance(x, Table)]

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)
    total_correct_tsr_share: float = 0
    tables_viewed: int = 0

    for table_gt, table_prediction in matched_tables.items():
        if table_prediction is not None:
            total_correct_tsr_share += _correct_tsr_share_table(table_gt, table_prediction)
            tables_prediction.remove(table_prediction)
        tables_viewed += 1

    tables_viewed += len(tables_prediction)

    return total_correct_tsr_share / tables_viewed


@overload
def correct_tsr_share(cell_a: Cell, cell_b: Cell):
    ...


@overload
def correct_tsr_share(cell_a: Cell, cells: List[Cell]):
    ...


@overload
def correct_tsr_share(table_a: Table, table_b: Table):
    ...


@overload
def correct_tsr_share(doc_gt: Document, revision_prediction: Revision):
    ...


def correct_tsr_share(param_a, param_b) -> float:
    if isinstance(param_a, Document) and isinstance(param_b, Revision):
        return _correct_tsr_share_document(param_a, param_b)
    elif isinstance(param_a, Table) and isinstance(param_b, Table):
        return _correct_tsr_share_table(param_a, param_b)
    elif isinstance(param_a, Cell) and isinstance(param_b, List):
        return _correct_tsr_share_cell_list(param_a, param_b)
    elif isinstance(param_a, Cell) and isinstance(param_b, Cell):
        return _correct_tsr_share_cell(param_a, param_b)
