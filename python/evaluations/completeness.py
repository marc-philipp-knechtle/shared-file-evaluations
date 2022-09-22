# anteil der korrekt identiffizierten elements, beachten: die anzahl der komplett vorhandenen elemente
# bsp. column enthält alle cells


# besipiel completeness: cells in columns einordnen
# 16 rows, 6 columns
# 16 rows identifiziert, 9 columns idetifiziert
# komplett richtig identifiziert: alle rows, column 1 und 5
# 9-2 = 7 columns die übrig sind stimmen nicht mit den eigtl. 5 überein

# completeness = (correct rows + correct collumns)/(total rows and columns)
from copy import copy
from typing import overload, Dict, List

from docrecjson.elements import Table, Revision, Document, Cell
from evaluations import utility


def _completeness_table(table_gt: Table, table_prediction: Table) -> float:
    rows_gt, columns_gt = table_gt.get_table_structure()
    rows_prediction, columns_prediction = table_prediction.get_table_structure()

    complete_elements: int = 0

    row: List[Cell]
    for i, row in enumerate(rows_gt):
        if len(row) == (len(rows_prediction[i]) if i < len(rows_prediction) else -1):
            complete_elements += 1

    column: List[Cell]
    for i, column in enumerate(columns_gt):
        if len(column) == (len(columns_prediction[i]) if i < len(columns_prediction) else -1):
            complete_elements += 1

    return complete_elements / (len(rows_gt) + len(columns_gt))


def _completeness_document(doc_gt: Document, revision_prediction: Revision) -> float:
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in copy(revision_prediction.objects) if isinstance(x, Table)]

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)
    total_completeness: float = 0
    tables_viewed: int = 0

    for table_gt, table_prediction in matched_tables.items():
        if table_prediction is not None:
            total_completeness += _completeness_table(table_gt, table_prediction)
            tables_prediction.remove(table_prediction)
        tables_viewed += 1

    # todo does this count to the metric? this makes the metric worse... but td is not part of this work
    tables_viewed += len(tables_prediction)

    return total_completeness / tables_viewed


@overload
def completeness(table_gt: Table, table_prediction: Table) -> float:
    ...


@overload
def completeness(doc_gt: Document, revision_prediction: Revision) -> float:
    ...


def completeness(param_a, param_b) -> float:
    if isinstance(param_a, Document) and isinstance(param_b, Revision):
        return _completeness_document(param_a, param_b)
    elif isinstance(param_a, Table) and isinstance(param_b, Table):
        return _completeness_table(param_a, param_b)
