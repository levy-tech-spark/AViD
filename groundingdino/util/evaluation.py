import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import torch.nn.functional as F
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from torchvision.ops import box_iou

class DetectionEvaluator:
    """Class for evaluating object detection models with standard metrics."""
    
    def __init__(self, 
                 iou_thresholds: List[float] = [0.5, 0.75],
                 score_threshold: float = 0.25,
                 max_detections: int = 100):
        """
        Initialize the detection evaluator.
        
        Args:
            iou_thresholds: List of IoU thresholds for evaluation
            score_threshold: Confidence score threshold for filtering predictions
            max_detections: Maximum number of detections to consider per image
        """
        self.iou_thresholds = iou_thresholds
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.reset()
        
    def reset(self):
        """Reset evaluation metrics."""
        self.metrics = {
            'ap': defaultdict(list),  # Average Precision per class
            'precision': defaultdict(list),  # Precision per class
            'recall': defaultdict(list),  # Recall per class
            'f1': defaultdict(list),  # F1 per class
        }
        self.total_gt = defaultdict(int)  # Total ground truth per class
        self.eval_data = []  # To store evaluation data per image
        
    def process_batch(self, 
                      predictions: Dict[str, torch.Tensor],
                      targets: List[Dict[str, torch.Tensor]],
                      captions: List[str]) -> None:
        """
        Process a batch of predictions and targets for evaluation.
        
        Args:
            predictions: Model predictions
                - pred_boxes: [B, N, 4] in [cx, cy, w, h] format
                - pred_logits: [B, N, num_classes] class logits
            targets: List of target dicts
                - boxes: [M, 4] in [cx, cy, w, h] format 
                - labels: [M] class indices
            captions: Text captions used for prediction
        """
        batch_size = len(targets)
        
        for i in range(batch_size):
            # Get predictions for this sample
            pred_logits = predictions["pred_logits"][i]  # [N, num_classes]
            pred_boxes = predictions["pred_boxes"][i]    # [N, 4]
            
            # Get targets for this sample
            gt_boxes = targets[i]["boxes"]    # [M, 4]
            gt_labels = targets[i]["labels"]  # [M]
            
            # Filter predictions by confidence
            scores, pred_classes = torch.max(pred_logits.sigmoid(), dim=1)
            keep = scores > self.score_threshold
            
            pred_boxes = pred_boxes[keep]
            pred_classes = pred_classes[keep]
            scores = scores[keep]
            
            # Sort by confidence score and keep top max_detections
            if len(scores) > 0:
                sorted_idxs = torch.argsort(scores, descending=True)
                if len(sorted_idxs) > self.max_detections:
                    sorted_idxs = sorted_idxs[:self.max_detections]
                    
                pred_boxes = pred_boxes[sorted_idxs]
                pred_classes = pred_classes[sorted_idxs]
                scores = scores[sorted_idxs]
            
            # Store evaluation data for this image
            self.eval_data.append({
                'pred_boxes': pred_boxes.detach().cpu(),
                'pred_classes': pred_classes.detach().cpu(),
                'scores': scores.detach().cpu(),
                'gt_boxes': gt_boxes.detach().cpu(),
                'gt_labels': gt_labels.detach().cpu(),
            })
            
            # Update total ground truth count per class
            for label in gt_labels:
                self.total_gt[label.item()] += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics including mAP, precision, recall, and F1.
        
        Returns:
            Dict of metrics with mean values
        """
        all_metrics = {}
        
        # Process each image's predictions
        for img_data in self.eval_data:
            pred_boxes = img_data['pred_boxes']
            pred_classes = img_data['pred_classes']
            scores = img_data['scores']
            gt_boxes = img_data['gt_boxes']
            gt_labels = img_data['gt_labels']
            
            # Skip if no predictions or ground truth
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
                
            # Convert boxes to [x1, y1, x2, y2] format for IoU calculation
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            
            # Compute IoU between prediction and ground truth boxes
            iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
            
            # Evaluate for each IoU threshold
            for iou_threshold in self.iou_thresholds:
                self._evaluate_image(
                    iou_matrix, 
                    pred_classes, 
                    gt_labels, 
                    scores, 
                    iou_threshold
                )
        
        # Calculate mean metrics across all classes and IoU thresholds
        for metric_name, class_values in self.metrics.items():
            # Mean across IoU thresholds per class
            class_means = {}
            for class_id, values in class_values.items():
                if values:  # Skip empty lists
                    class_means[class_id] = np.mean(values)
            
            # Mean across classes
            if class_means:
                all_metrics[f'mean_{metric_name}'] = np.mean(list(class_means.values()))
                # Also include per-class metrics
                for class_id, value in class_means.items():
                    all_metrics[f'{metric_name}_class_{class_id}'] = value
        
        # Add detection statistics
        all_metrics['detected_boxes'] = sum(len(img_data['pred_boxes']) for img_data in self.eval_data)
        all_metrics['gt_boxes'] = sum(len(img_data['gt_boxes']) for img_data in self.eval_data)
        
        return all_metrics
    
    def _evaluate_image(self, 
                        iou_matrix: torch.Tensor,
                        pred_classes: torch.Tensor,
                        gt_labels: torch.Tensor,
                        scores: torch.Tensor,
                        iou_threshold: float) -> None:
        """
        Evaluate predictions for a single image at a specific IoU threshold.
        
        Args:
            iou_matrix: IoU values between prediction and ground truth boxes [P, G]
            pred_classes: Predicted class indices [P]
            gt_labels: Ground truth class indices [G]
            scores: Prediction confidence scores [P]
            iou_threshold: IoU threshold for evaluation
        """
        # Process each class separately
        unique_classes = torch.cat([pred_classes, gt_labels]).unique()
        
        for class_id in unique_classes:
            class_id = class_id.item()
            
            # Find predictions and ground truths for this class
            pred_mask = pred_classes == class_id
            gt_mask = gt_labels == class_id
            
            if not pred_mask.any() or not gt_mask.any():
                continue
                
            # Filter IoU matrix for this class
            class_iou = iou_matrix[pred_mask][:, gt_mask]
            
            if class_iou.numel() == 0:
                continue
                
            # Get prediction scores for this class
            class_scores = scores[pred_mask]
            
            # Calculate precision and recall for this class and IoU threshold
            ap, precision, recall, f1 = self._calculate_pr_metrics(
                class_iou, 
                class_scores, 
                iou_threshold
            )
            
            # Store metrics
            self.metrics['ap'][class_id].append(ap)
            self.metrics['precision'][class_id].append(precision)
            self.metrics['recall'][class_id].append(recall)
            self.metrics['f1'][class_id].append(f1)
    
    def _calculate_pr_metrics(self, 
                             iou_matrix: torch.Tensor,
                             scores: torch.Tensor,
                             iou_threshold: float) -> Tuple[float, float, float, float]:
        """
        Calculate precision, recall, F1, and average precision for a single class at a specific IoU threshold.
        
        Args:
            iou_matrix: IoU values between prediction and ground truth boxes for a specific class [P, G]
            scores: Prediction confidence scores for a specific class [P]
            iou_threshold: IoU threshold for evaluation
            
        Returns:
            Tuple of (average precision, precision, recall, f1)
        """
        num_pred = len(scores)
        num_gt = iou_matrix.shape[1]
        
        if num_pred == 0 or num_gt == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Sort predictions by confidence score
        sorted_indices = torch.argsort(scores, descending=True)
        iou_matrix = iou_matrix[sorted_indices]
        
        # Initialize arrays to track matches
        gt_matched = torch.zeros(num_gt, dtype=torch.bool)
        pred_matched = torch.zeros(num_pred, dtype=torch.bool)
        
        # Track true/false positives
        tp = torch.zeros(num_pred, dtype=torch.bool)
        fp = torch.zeros(num_pred, dtype=torch.bool)
        
        # Match predictions to ground truths using greedy matching
        for pred_idx in range(num_pred):
            # Find ground truths that match this prediction above the IoU threshold
            above_threshold = iou_matrix[pred_idx] >= iou_threshold
            
            # Only consider unmatched ground truths
            valid_matches = above_threshold & ~gt_matched
            
            if valid_matches.any():
                # Get best matching ground truth
                best_gt_idx = torch.argmax(iou_matrix[pred_idx] * valid_matches.float())
                
                if valid_matches[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    pred_matched[pred_idx] = True
                    tp[pred_idx] = True
            else:
                fp[pred_idx] = True
        
        # Calculate precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / num_gt
        
        # Calculate average precision using all points interpolation
        precisions = torch.cat([torch.tensor([1.0]), precisions])
        recalls = torch.cat([torch.tensor([0.0]), recalls])
        
        # Compute AP using trapezoidal rule
        ap = torch.trapz(precisions, recalls).item()
        
        # Calculate final precision and recall
        if tp.sum() > 0:
            precision = tp.sum().float() / (tp.sum() + fp.sum())
            recall = tp.sum().float() / num_gt
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        else:
            precision = torch.tensor(0.0)
            recall = torch.tensor(0.0)
            f1 = torch.tensor(0.0)
        
        return ap, precision.item(), recall.item(), f1.item()


def evaluate_model(model, data_loader, prepare_batch_fn, device="cuda"):
    """
    Evaluate a model on a dataset and compute detection metrics.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation/test data
        prepare_batch_fn: Function to prepare batch data for the model
        device: Device to run evaluation on
        
    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    evaluator = DetectionEvaluator()
    
    with torch.no_grad():
        for batch in data_loader:
            images, targets, captions = prepare_batch_fn(batch)
            
            # Forward pass
            outputs = model(images, captions=captions)
            
            # Process batch for evaluation
            evaluator.process_batch(outputs, targets, captions)
    
    # Compute and return metrics
    metrics = evaluator.compute_metrics()
    return metrics


def visualize_predictions(model, data_loader, prepare_batch_fn, num_samples=5, 
                          score_threshold=0.25, device="cuda"):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation/test data
        prepare_batch_fn: Function to prepare batch data for the model
        num_samples: Number of samples to visualize
        score_threshold: Confidence score threshold
        device: Device to run evaluation on
        
    Returns:
        List of dictionaries with original images, predictions and ground truth
    """
    from groundingdino.util.inference import GroundingDINOVisualizer
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy
    import cv2
    
    model.eval()
    visualizer = GroundingDINOVisualizer(save_dir="eval_visualizations")
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            images, targets, captions = prepare_batch_fn(batch)
            
            # Forward pass
            outputs = model(images, captions=captions)
            
            # Process the first image in batch for visualization
            img = targets[0]["orig_img"] if "orig_img" in targets[0] else None
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Get predictions
            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]
            
            # Filter by confidence
            scores, pred_classes = torch.max(pred_logits.sigmoid(), dim=1)
            keep = scores > score_threshold
            
            filtered_boxes = pred_boxes[keep]
            filtered_scores = scores[keep]
            
            # Get ground truth
            gt_boxes = targets[0]["boxes"]
            
            # Convert to absolute coordinates
            if len(filtered_boxes) > 0:
                pred_boxes_xyxy = box_cxcywh_to_xyxy(filtered_boxes) * torch.tensor([w, h, w, h])
            else:
                pred_boxes_xyxy = torch.zeros((0, 4))
                
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes) * torch.tensor([w, h, w, h])
            
            # Add to results
            results.append({
                "image": img,
                "pred_boxes": pred_boxes_xyxy.cpu().numpy(),
                "pred_scores": filtered_scores.cpu().numpy(),
                "gt_boxes": gt_boxes_xyxy.cpu().numpy(),
                "caption": captions[0]
            })
    
    return results


def print_evaluation_report(metrics):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print(" "*15 + "MODEL EVALUATION REPORT")
    print("="*50)
    
    # Print overall metrics
    print("\nOVERALL METRICS:")
    print(f"Mean Average Precision (mAP): {metrics.get('mean_ap', 0):.4f}")
    print(f"Mean Precision: {metrics.get('mean_precision', 0):.4f}")
    print(f"Mean Recall: {metrics.get('mean_recall', 0):.4f}")
    print(f"Mean F1 Score: {metrics.get('mean_f1', 0):.4f}")
    
    # Print per-class metrics if available
    class_metrics = {k: v for k, v in metrics.items() if "class_" in k}
    if class_metrics:
        print("\nPER-CLASS METRICS:")
        
        # Group metrics by class
        class_groups = defaultdict(dict)
        for k, v in class_metrics.items():
            metric_name, class_id = k.split('class_')
            metric_name = metric_name[:-1]  # Remove trailing underscore
            class_id = int(class_id)
            class_groups[class_id][metric_name] = v
        
        # Print metrics for each class
        for class_id, class_dict in sorted(class_groups.items()):
            print(f"\nClass {class_id}:")
            for metric_name, value in class_dict.items():
                print(f"  {metric_name.capitalize()}: {value:.4f}")
    
    # Print detection statistics
    print("\nDETECTION STATISTICS:")
    print(f"Total predicted boxes: {metrics.get('detected_boxes', 0)}")
    print(f"Total ground truth boxes: {metrics.get('gt_boxes', 0)}")
    print("="*50) 