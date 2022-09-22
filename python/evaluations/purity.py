from copy import copy
from typing import overload, List, Dict

from docrecjson.elements import Table, Document, Revision
from evaluations import utility


def _purity_table(table_gt: Table, table_prediction: Table) -> float:
    rows_gt, columns_gt = table_gt.get_table_structure()
    rows_prediction, columns_prediction = table_prediction.get_table_structure()

    columns_identified: int = 0
    rows_identified: int = 0

    if len(columns_prediction) >= len(columns_gt):
        columns_identified += len(columns_gt)
    else:
        columns_identified += len(columns_prediction) if len(columns_prediction) > 1 else 0

    if len(rows_prediction) >= len(rows_gt):
        rows_identified += len(rows_gt)
    else:
        rows_identified += len(rows_prediction) if len(rows_prediction) > 1 else 0

    return (columns_identified + rows_identified) / (len(columns_gt) + len(rows_gt))


def _purity_document(doc_gt: Document, revision_prediction: Revision) -> float:
    tables_gt: List[Table] = [x for x in doc_gt.objects() if isinstance(x, Table)]
    tables_prediction: List[Table] = [x for x in copy(revision_prediction.objects) if isinstance(x, Table)]

    matched_tables: Dict[Table, Table] = utility.match_tables(tables_gt, tables_prediction)
    total_purity: float = 0
    tables_viewed: int = 0

    for table_gt, table_prediction in matched_tables.items():
        if table_prediction is not None:
            total_purity += purity(table_gt, table_prediction)
            tables_prediction.remove(table_prediction)
        tables_viewed += 1

    # todo does this count to the metric? this makes the metric worse... but td is not part of this work
    tables_viewed += len(tables_prediction)

    return total_purity / tables_viewed


@overload
def purity(table_gt: Table, table_prediction: Table) -> float:
    ...


@overload
def purity(doc_gt: Document, revision_prediction: Revision) -> float:
    ...


def purity(param_a, param_b):
    if isinstance(param_a, Document) and isinstance(param_b, Revision):
        return _purity_document(param_a, param_b)
    elif isinstance(param_a, Table) and isinstance(param_b, Table):
        return _purity_table(param_a, param_b)
