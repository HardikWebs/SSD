import os
import numpy as np
import torch
from collections import defaultdict
from ssd.structures.boxlist import BoxList
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def kitti_evaluation(dataset, predictions, output_dir):
    """
    Evaluate the SSD model on the KITTI dataset.

    Args:
        dataset (MyDataset): The KITTI dataset object.
        predictions (list): List of model predictions (BoxList objects).
        output_dir (str): Directory to save evaluation results and visualizations.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert ground truth annotations to a format suitable for evaluation
    ground_truths = []
    for i in range(len(dataset)):
        _, targets, _ = dataset[i]
        boxes = targets.boxes  # Ground truth boxes (N, 4)
        labels = targets.labels  # Ground truth labels (N,)
        ground_truths.append((boxes, labels))

    # Convert model predictions to a format suitable for evaluation
    detections = []
    for prediction in predictions:
        boxes = prediction.bbox.numpy()  # Predicted boxes (M, 4)
        labels = prediction.get_field("labels").numpy()  # Predicted labels (M,)
        scores = prediction.get_field("scores").numpy()  # Confidence scores (M,)
        detections.append((boxes, labels, scores))

    # Evaluate detections using mean Average Precision (mAP)
    results = evaluate_detections(ground_truths, detections, dataset.class_names)

    # Save evaluation results to a file
    result_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(result_file, "w") as f:
        for class_name, metrics in results.items():
            f.write(f"Class: {class_name}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"AP: {metrics['ap']:.4f}\n")
            f.write("\n")

    # Save visualization of detections for a few samples
    visualize_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualize_dir, exist_ok=True)
    for i in range(min(10, len(dataset))):  # Visualize first 10 samples
        image, targets, _ = dataset[i]
        pred_boxes, pred_labels, pred_scores = detections[i]
        save_detection_results(
            image, targets.boxes, targets.labels, pred_boxes, pred_labels, pred_scores,
            dataset.class_names, os.path.join(visualize_dir, f"sample_{i}.png")
        )

    print(f"Evaluation results saved to {output_dir}")

def evaluate_detections(ground_truths, detections, class_names):
    """
    Evaluate detections using mean Average Precision (mAP).

    Args:
        ground_truths (list): List of ground truth annotations (boxes, labels).
        detections (list): List of detections (boxes, labels, scores).
        class_names (list): List of class names.

    Returns:
        dict: Evaluation results for each class.
    """
    results = defaultdict(lambda: {"precision": 0.0, "recall": 0.0, "ap": 0.0})

    for class_idx, class_name in enumerate(class_names):
        if class_name == "__background__":
            continue

        y_true = []
        y_scores = []

        for gt, det in zip(ground_truths, detections):
            gt_boxes, gt_labels = gt
            det_boxes, det_labels, det_scores = det

            # Filter predictions and ground truth for the current class
            gt_mask = (gt_labels == class_idx)
            det_mask = (det_labels == class_idx)

            y_true.extend(gt_mask.astype(int))
            y_scores.extend(det_scores[det_mask])

        if len(y_true) > 0 and len(y_scores) > 0:
            ap = average_precision_score(y_true, y_scores)
            results[class_name]["ap"] = ap

    return results

def save_detection_results(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, class_names, output_path):
    """
    Visualize ground truth and predicted bounding boxes on the image and save the result.

    Args:
        image (np.array): Input image (H, W, 3).
        gt_boxes (np.array): Ground truth boxes (N, 4).
        gt_labels (np.array): Ground truth labels (N,).
        pred_boxes (np.array): Predicted boxes (M, 4).
        pred_labels (np.array): Predicted labels (M,).
        pred_scores (np.array): Confidence scores (M,).
        class_names (list): List of class names.
        output_path (str): Path to save the visualization.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # Plot ground truth boxes
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="green", linewidth=2))
        plt.text(x1, y1, class_names[label], color="green", fontsize=12, bbox=dict(facecolor="white", alpha=0.7))

    # Plot predicted boxes
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
        plt.text(x1, y1, f"{class_names[label]} {score:.2f}", color="red", fontsize=12, bbox=dict(facecolor="white", alpha=0.7))

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()