from docrecjson.elements import Document, PolygonRegion
from shapely.geometry import Polygon


# todo check issues with shapely-speed ->
#  https://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely


def intersection_over_union(ground_truth: Document, prediction: Document) -> float:
    polygon_content_gt = [x for x in ground_truth.content if type(x) == PolygonRegion]
    polygon_content_prediction = [x for x in prediction.content if type(x) == PolygonRegion]

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

                print("Size of the ground truth: " + str(area_gt_polygon.area))
                print("Size of the prediction element: " + str(area_prediction_polygon.area))
                print("Size of the intersection of those polygons: " + str(intersection.area))

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

# todo make version of this with single shared-file-format document where the ground truth is the index 0 element
# the other elements will be predictions -> it returns a list of prediction - for each version in comparison to the
# ground truth
