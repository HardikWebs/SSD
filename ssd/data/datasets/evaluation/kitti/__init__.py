# import os
# import numpy as np
# import torch
# from collections import defaultdict
# from sklearn.metrics import average_precision_score, precision_recall_curve
# import matplotlib.pyplot as plt
# from datetime import datetime
# import logging


# def kitti_evaluation(dataset, predictions, output_dir, iteration=None):
#     """
#     Evaluate the SSD model on the KITTI dataset.

#     Args:
#         dataset (MyDataset): The KITTI dataset object.
#         predictions (list): List of model predictions (BoxList objects).
#         output_dir (str): Directory to save evaluation results and visualizations.
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Convert ground truth annotations to a format suitable for evaluation
#     ground_truths = []
#     for i in range(len(dataset)):
#         _, targets, _ = dataset[i]
#         boxes = targets["boxes"]
#         labels = targets["labels"]
#         ground_truths.append((boxes, labels))

#     # Convert model predictions to a format suitable for evaluation
#     detections = []
#     for prediction in predictions:
#         boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
#         # boxes = prediction.bbox.numpy()  # Predicted boxes (M, 4)
#         # labels = prediction.get_field("labels").numpy()  # Predicted labels (M,)
#         # scores = prediction.get_field("scores").numpy()  # Confidence scores (M,)
#         detections.append((boxes, labels, scores))

#     # Evaluate detections using mean Average Precision (mAP)
#     results = evaluate_detections(ground_truths, detections, dataset.class_names)

#     # Format results
#     result_str = ""
#     metrics = {"mAP": results["map"]}
#     for i, ap in enumerate(results["ap"]):
#         if i == 0:  # Skip background
#             continue
#         class_name = dataset.class_names[i]
#         metrics[class_name] = ap
#         result_str += "{:<16}: {:.4f}\n".format(class_name, ap)

#     # Log results
#     logger = logging.getLogger("SSD.inference")
#     logger.info(result_str)

#     # Save results to a file
#     if iteration is not None:
#         result_path = os.path.join(output_dir, "result_{:07d}.txt".format(iteration))
#     else:
#         result_path = os.path.join(output_dir, "result_{}.txt".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
#     with open(result_path, "w") as f:
#         f.write(result_str)

#     # Save visualization of detections for a few samples
#     visualize_dir = os.path.join(output_dir, "visualizations")
#     os.makedirs(visualize_dir, exist_ok=True)
#     for i in range(min(10, len(dataset))):  # Visualize first 10 samples
#         image, targets, image_id = dataset[i]
#         if image_id in predictions:
#             pred_boxes, pred_labels, pred_scores = detections[i]
#         else:
#             pred_boxes, pred_labels, pred_scores = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
#         save_detection_results(
#             image, targets.boxes, targets.labels, pred_boxes, pred_labels, pred_scores,
#             dataset.class_names, os.path.join(visualize_dir, f"sample_{i}.png")
#         )

#     print(f"Evaluation results saved to {output_dir}")
#     return dict(metrics=metrics)

# def compute_iou(boxes1, boxes2):
#     """
#     Compute IoU between two sets of boxes.
#     Args:
#         boxes1 (np.array): Shape (N, 4).
#         boxes2 (np.array): Shape (M, 4).
#     Returns:
#         np.array: IoU matrix of shape (N, M).
#     """
#     # Ensure boxes are valid
#     if len(boxes1) == 0 or len(boxes2) == 0:
#         return np.zeros((len(boxes1), len(boxes2)))

#     # Get coordinates of boxes
#     x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
#     y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
#     x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
#     y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

#     # Compute intersection area
#     intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

#     # Compute areas of boxes
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

#     # Compute union area
#     union = area1[:, None] + area2 - intersection

#     # Compute IoU
#     iou = intersection / union
#     return iou

# def bbox_iou(bbox_a, bbox_b):
#     """Calculate the Intersection of Unions (IoUs) between bounding boxes.
#     IoU is calculated as a ratio of area of the intersection
#     and area of the union.
#     This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
#     inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
#     same type.
#     The output is same type as the type of the inputs.
#     Args:
#         bbox_a (array): An array whose shape is :math:`(N, 4)`.
#             :math:`N` is the number of bounding boxes.
#             The dtype should be :obj:`numpy.float32`.
#         bbox_b (array): An array similar to :obj:`bbox_a`,
#             whose shape is :math:`(K, 4)`.
#             The dtype should be :obj:`numpy.float32`.
#     Returns:
#         array:
#         An array whose shape is :math:`(N, K)`. \
#         An element at index :math:`(n, k)` contains IoUs between \
#         :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
#         box in :obj:`bbox_b`.
#     """
#     # Ensure boxes are valid
#     if len(bbox_a) == 0 or len(bbox_b) == 0:
#         return np.zeros((len(bbox_a), len(bbox_b)))

#     if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
#         raise IndexError

#     # top left
#     tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
#     # bottom right
#     br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

#     area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
#     area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
#     area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
#     return area_i / (area_a[:, None] + area_b - area_i)

# def evaluate_detections(ground_truths, detections, class_names, iou_threshold=0.5):
#     """
#     Evaluate detections using mean Average Precision (mAP).

#     Args:
#         ground_truths (list): List of ground truth annotations (boxes, labels).
#         detections (list): List of detections (boxes, labels, scores).
#         class_names (list): List of class names.
#         iou_threshold (float): IoU threshold for matching predictions to ground truth.

#     Returns:
#         dict: Evaluation results containing mAP, AP per class, precision, and recall.
#     """
#     results = defaultdict(lambda: {"precision": 0.0, "recall": 0.0, "ap": 0.0})
#     aps = []

#     for class_idx, class_name in enumerate(class_names):
#         if class_name == "__background__":
#             continue

#         y_true = []
#         y_scores = []

#         for gt, det in zip(ground_truths, detections):
#             gt_boxes, gt_labels = gt
#             det_boxes, det_labels, det_scores = det

#             # Filter ground truth and predictions for the current class
#             gt_mask = (gt_labels == class_idx)
#             det_mask = (det_labels == class_idx)

#             gt_boxes_class = gt_boxes[gt_mask]
#             det_boxes_class = det_boxes[det_mask]
#             det_scores_class = det_scores[det_mask]

#             # Initialize true positives and false positives
#             tp = np.zeros(len(det_scores_class))
#             fp = np.zeros(len(det_scores_class))

#             if len(gt_boxes_class) == 0:
#                 # No ground truth for this class, all predictions are false positives
#                 fp[:] = 1
#             else:
#                 # Compute IoU between predicted and ground truth boxes
#                 iou = bbox_iou(det_boxes_class, gt_boxes_class)

#                 # Match predictions to ground truth
#                 for i in range(len(det_scores_class)):
#                     max_iou = np.max(iou[i])
#                     if max_iou >= iou_threshold:
#                         tp[i] = 1
#                     else:
#                         fp[i] = 1

#             # Accumulate true positives and scores
#             y_true.extend(tp)
#             y_scores.extend(det_scores_class)

#         if len(y_true) > 0 and len(y_scores) > 0:
#             # Compute precision, recall, and AP
#             precision, recall, _ = precision_recall_curve(y_true, y_scores)
#             ap = average_precision_score(y_true, y_scores)

#             # Store results
#             results[class_name]["precision"] = precision.mean()
#             results[class_name]["recall"] = recall.mean()
#             results[class_name]["ap"] = ap
#             aps.append(ap)
#         else:
#             # If no predictions or ground truth for this class, set AP to 0
#             results[class_name]["ap"] = 0.0
#             aps.append(0.0)

#     # Compute mean Average Precision (mAP)
#     results["map"] = np.mean(aps) if aps else 0.0
#     results["ap"] = aps  # Store APs for each class

#     return results

# def save_detection_results(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, class_names, output_path):
#     """
#     Visualize ground truth and predicted bounding boxes on the image and save the result.

#     Args:
#         image (np.array): Input image (H, W, 3).
#         gt_boxes (np.array): Ground truth boxes (N, 4).
#         gt_labels (np.array): Ground truth labels (N,).
#         pred_boxes (np.array): Predicted boxes (M, 4).
#         pred_labels (np.array): Predicted labels (M,).
#         pred_scores (np.array): Confidence scores (M,).
#         class_names (list): List of class names.
#         output_path (str): Path to save the visualization.
#     """
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image)

#     # Plot ground truth boxes
#     for box, label in zip(gt_boxes, gt_labels):
#         x1, y1, x2, y2 = box
#         plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="green", linewidth=2))
#         plt.text(x1, y1, class_names[label], color="green", fontsize=12, bbox=dict(facecolor="white", alpha=0.7))

#     # Plot predicted boxes
#     for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
#         x1, y1, x2, y2 = box
#         plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
#         plt.text(x1, y1, f"{class_names[label]} {score:.2f}", color="red", fontsize=12, bbox=dict(facecolor="white", alpha=0.7))

#     plt.axis("off")
#     plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
#     plt.close()

import logging
import os
from datetime import datetime

import numpy as np

from .eval_detection_voc import eval_detection_voc  # Assuming this function is available

def kitti_evaluation(dataset, predictions, output_dir, iteration=None):
    """
    Evaluate the SSD model on the KITTI dataset.

    Args:
        dataset (MyDataset): The KITTI dataset object.
        predictions (list): List of model predictions (BoxList objects).
        output_dir (str): Directory to save evaluation results.
        iteration (int, optional): Iteration number for saving results. Defaults to None.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []

    for i in range(len(dataset)):
        # Get ground truth annotations
        image, targets, _ = dataset[i]
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)

        # Get predictions
        prediction = predictions[i]
        width, height = dataset.get_image_dimensions(i)
        prediction = prediction.resize((width, height)).numpy()  # Resize to image dimensions
        boxes, labels, scores = prediction["boxes"], prediction["labels"], prediction["scores"]

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)

    # Evaluate detections using VOC-style evaluation
    result = eval_detection_voc(
        pred_bboxes=pred_boxes_list,
        pred_labels=pred_labels_list,
        pred_scores=pred_scores_list,
        gt_bboxes=gt_boxes_list,
        gt_labels=gt_labels_list,
        gt_difficults=None,  # KITTI dataset does not have a 'difficult' flag
        iou_thresh=0.5,
        use_07_metric=False
    )

    # Log results
    logger = logging.getLogger("SSD.inference")
    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # Skip background
            continue
        metrics[class_names[i]] = ap
        result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    logger.info(result_str)

    # Save results to a file
    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return dict(metrics=metrics)