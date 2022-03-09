import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    # Boxes intersect if they do not fulfill the conditions where they _don't_ intersect
    do_boxes_intersect = not (prediction_box[0] > gt_box[2]
                              or prediction_box[2] < gt_box[0]
                              or prediction_box[1] > gt_box[3]
                              or prediction_box[3] < gt_box[1])

    if do_boxes_intersect:
        top_left_corner = np.max(
            np.array([prediction_box, gt_box]), axis=0)[:2]
        bottom_right_corner = np.min(
            np.array([prediction_box, gt_box]), axis=0)[2:]

        diff = bottom_right_corner - top_left_corner

        intersection = diff[0] * diff[1]  # Area of intersection
        assert intersection > 0, f"Intersection {intersection:0.2f} is negative!"
    else:
        # No intersection
        intersection = 0

    # Compute union

    def _box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    union = _box_area(prediction_box) + _box_area(gt_box) - intersection
    iou = intersection / union  # Intersection over Union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    precision = num_tp / (num_tp + num_fp)
    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    recall = num_tp / (num_tp + num_fn)
    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    matches = []

    for i_pred, pred_box in enumerate(prediction_boxes):
        for i_gt, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(prediction_box=pred_box,
                                gt_box=gt_box)
            if iou > iou_threshold:
                matches.append((i_pred, i_gt, iou))

    # Sort all matches on IoU in descending order
    sorted_matches = sorted(matches, key=lambda tup: tup[2], reverse=True)

    # Filter out too low

    # Keep track of the indices already used
    used_pred_indices = set()
    used_gt_indices = set()

    match_pred_indices = []
    match_gt_indices = []

    # Find all matches with the highest IoU threshold
    # Use greedy approach, but keep track of boxes already in matches
    for match in sorted_matches:
        i_pred, i_gt, iou = match
        if i_pred in used_pred_indices or i_gt in used_gt_indices:
            # Match cannot be in multiple pairs!
            continue

        # Keep track of used indices
        used_pred_indices.add(i_pred)
        used_gt_indices.add(i_gt)

        match_pred_indices.append(i_pred)
        match_gt_indices.append(i_gt)

    prediction_box_matches = np.take(
        a=prediction_boxes, indices=match_pred_indices, axis=0)
    gt_box_matches = np.take(a=gt_boxes, indices=match_gt_indices, axis=0)

    assert prediction_box_matches.shape == gt_box_matches.shape, "Prediction and gt boxes different shapes!"

    return prediction_box_matches, gt_box_matches


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """


    pred_box_matches, gt_box_matches = get_all_box_matches(
        prediction_boxes=prediction_boxes,
        gt_boxes=gt_boxes,
        iou_threshold=iou_threshold)

    # True positive: A pred matched to gt
    true_pos = pred_box_matches.shape[0]

    # False positive: A pred not matched to a gt
    false_pos = prediction_boxes.shape[0] - pred_box_matches.shape[0]

    # False negative: A gt not matched to a pred
    false_neg = gt_boxes.shape[0] - gt_box_matches.shape[0]

    result = {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}
    return result


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    results = []

    for pred_boxes, gt_boxes in zip (all_prediction_boxes, all_gt_boxes):
        result = calculate_individual_image_result(
            prediction_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=iou_threshold
        )
        results.append(result)




def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = []
    recalls = []
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(
        precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
