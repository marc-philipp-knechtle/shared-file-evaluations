import json

from docrecjson import decoder
from docrecjson.elements import Document

import python.evaluations.iou as iou

example_shared_file_format_simple_1_path: str = "../../resources/example-shared-file-format-simple-1.json"
example_shared_file_format_simple_2_path: str = "../../resources/example-shared-file-format-simple-2.json"


def _load_document(filepath: str) -> Document:
    with open(filepath) as json_data:
        json_annotation = json.load(json_data)

    return decoder.loads(json.dumps(json_annotation))


def test_equal_files():
    example_document: Document = _load_document(example_shared_file_format_simple_1_path)
    assert iou.intersection_over_union(example_document, example_document) == 1


def test_small_iou():
    """
    coordinates1:   [296,775],[388,775],[388,765],[296,765]
                    [bottom left][bottom right][upper right][upper left]
                    width = 91, height = 10, area = 910
    coordinates2:   [296,774],[388,774],[388,765],[296,765]
                    width = 91, height = 9, area = 819

    -> area from coordinates2 is the common region
    -> iou = 819/(910 - 819) + (819 - 819) + 819 = 0.9

    """
    document_1: Document = _load_document(example_shared_file_format_simple_1_path)
    document_2: Document = _load_document(example_shared_file_format_simple_2_path)
    assert iou.intersection_over_union(document_1, document_2) == 0.9
