import nltk
from docrecjson.elements import Cell, Table, Revision, Document
from typing import overload, List, Dict

import python.evaluations.utility as utility


def _levenshtein_distance_total(cell_a: Cell, cell_b: Cell) -> int:
    return nltk.edit_distance(cell_a.text_content.text, cell_b.text_content.text)


def _levenshtein_distance_cell(cell_a: Cell, cell_b: Cell) -> float:
    if cell_a.text_content is None or cell_b.text_content is None:
        return 1
    return _levenshtein_distance_total(cell_a, cell_b) / max(len(cell_a.text_content.text),
                                                             len(cell_b.text_content.text))


def _levenshtein_distance_table(table_a: Table, table_b: Table) -> float:
    cells_viewed: int = 0
    total_levenshtein_distance_relative: float = 0
    search_cells: List[Cell] = table_b.cells
    for cell in table_a.cells:
        matching_cell = utility.find_cell_with_highest_intersection_area(cell, search_cells)
        if matching_cell is not None:
            total_levenshtein_distance_relative += levenshtein_distance(cell, matching_cell)
            search_cells.remove(matching_cell)
        else:
            total_levenshtein_distance_relative += 1
        cells_viewed += 1

    # add relative value for each cell which was not matched
    total_levenshtein_distance_relative += len(search_cells)
    cells_viewed += len(search_cells)

    return total_levenshtein_distance_relative / cells_viewed if cells_viewed > 0 else 0


def _levenshtein_distance_document(doc_gt: Document, revision_prediction: Revision) -> float:
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in revision_prediction.objects if isinstance(x, Table)]

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)
    tables_viewed: int = 0
    total_levenshtein_distance: float = 0

    for table_gt, table_prediction in matched_tables.items():
        if table_prediction is not None:
            total_levenshtein_distance += levenshtein_distance(table_gt, table_prediction)
            tables_prediction.remove(table_prediction)
        else:
            total_levenshtein_distance += 1
        tables_viewed += 1

    tables_viewed += len(tables_prediction)
    total_levenshtein_distance += len(tables_prediction)

    return total_levenshtein_distance / tables_viewed if tables_viewed != 0 else 0


@overload
def levenshtein_distance(cell_a: Cell, cell_b: Cell):
    ...


@overload
def levenshtein_distance(table_a: Table, table_b: Table):
    ...


@overload
def levenshtein_distance(doc_gt: Document, revision_prediction: Revision):
    ...


def levenshtein_distance(param_a, param_b):
    if isinstance(param_a, Document):
        return _levenshtein_distance_document(param_a, param_b)
    elif isinstance(param_b, Table):
        return _levenshtein_distance_table(param_a, param_b)
    elif isinstance(param_a, Cell):
        return _levenshtein_distance_cell(param_a, param_b)
