import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
from pathlib import Path

from groundingdino.util.inference import load_model
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from groundingdino.util.evaluation import evaluate_model, visualize_predictions, print_evaluation_report
from groundingdino.datasets.dataset import GroundingDINODataset
from config import ConfigurationManager


def setup_data_loader(data_config, batch_size=1):
    """
    Set up evaluation data loader.
    
    Args:
        data_config: Data configuration
        batch_size: Batch size for evaluation
        
    Returns:
        DataLoader for evaluation
    """
    eval_dataset = GroundingDINODataset(
        data_config.val_dir,
        data_config.val_ann
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return eval_loader


def prepare_batch(batch, device="cuda"):
    """
    Prepare batch data for model evaluation.
    
    Args:
        batch: Data batch from DataLoader
        device: Device to run inference on
        
    Returns:
        Tuple of (images, targets, captions)
    """
    from groundingdino.util.misc import nested_tensor_from_tensor_list
    
    images, targets = batch
    
    # Convert list of images to NestedTensor and move to device
    if isinstance(images, (list, tuple)):
        images = nested_tensor_from_tensor_list(images)
    images = images.to(device)
    
    # Process targets
    captions = []
    for target in targets:
        target['boxes'] = target['boxes'].to(device)
        target['size'] = target['size'].to(device)
        target['labels'] = target['labels'].to(device)
        
        # Save original image for visualization
        if 'orig_size' in target:
            h, w = target['orig_size']
            if 'orig_img' not in target and 'img' in target:
                target['orig_img'] = target['img']
        
        captions.append(target['caption'])
    
    return images, targets, captions


def visualize_results(results, save_dir='evaluation_results/visualizations'):
    """
    Visualize and save evaluation results.
    
    Args:
        results: List of dictionaries with images and predictions
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        img = result['image'].copy()
        pred_boxes = result['pred_boxes']
        pred_scores = result['pred_scores']
        gt_boxes = result['gt_boxes']
        caption = result['caption']
        
        # Draw ground truth boxes in green
        for box in gt_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw prediction boxes in red with confidence scores
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Add caption to the image
        cv2.putText(img, f"Caption: {caption}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Save the image
        output_path = os.path.join(save_dir, f"sample_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Saved {len(results)} visualizations to {save_dir}")


def plot_metrics(metrics, save_dir='evaluation_results'):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Metrics dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract class metrics
    class_metrics = {k: v for k, v in metrics.items() if "class_" in k}
    
    if class_metrics:
        # Group by class
        class_groups = {}
        for k, v in class_metrics.items():
            metric_name, class_id = k.split('class_')
            metric_name = metric_name[:-1]  # Remove trailing underscore
            class_id = int(class_id)
            
            if class_id not in class_groups:
                class_groups[class_id] = {}
            class_groups[class_id][metric_name] = v
        
        # Plot metrics per class
        metric_names = ['ap', 'precision', 'recall', 'f1']
        
        # Extract class metrics
        class_ids = sorted(class_groups.keys())
        metric_values = {metric: [class_groups[class_id].get(metric, 0) for class_id in class_ids] 
                         for metric in metric_names}
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.2
        x = np.arange(len(class_ids))
        
        for i, metric in enumerate(metric_names):
            ax.bar(x + i*bar_width, metric_values[metric], bar_width, label=metric.capitalize())
        
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics per Class')
        ax.set_xticks(x + bar_width * (len(metric_names) - 1) / 2)
        ax.set_xticklabels(class_ids)
        ax.legend()
        
        plt.savefig(os.path.join(save_dir, 'class_metrics.png'))
        plt.close()
    
    # Plot overall metrics
    overall_metrics = {
        'Mean AP': metrics.get('mean_ap', 0),
        'Mean Precision': metrics.get('mean_precision', 0),
        'Mean Recall': metrics.get('mean_recall', 0),
        'Mean F1': metrics.get('mean_f1', 0)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(overall_metrics.keys(), overall_metrics.values())
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Metrics')
    ax.set_ylim(0, 1)
    
    for i, v in enumerate(overall_metrics.values()):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.savefig(os.path.join(save_dir, 'overall_metrics.png'))
    plt.close()


def save_metrics_to_file(metrics, save_dir='evaluation_results', filename='metrics.txt'):
    """
    Save metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        save_dir: Directory to save the file
        filename: Name of the file
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, filename)
    
    with open(output_path, 'w') as f:
        f.write("MODEL EVALUATION METRICS\n")
        f.write("========================\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"Mean Average Precision (mAP): {metrics.get('mean_ap', 0):.4f}\n")
        f.write(f"Mean Precision: {metrics.get('mean_precision', 0):.4f}\n")
        f.write(f"Mean Recall: {metrics.get('mean_recall', 0):.4f}\n")
        f.write(f"Mean F1 Score: {metrics.get('mean_f1', 0):.4f}\n\n")
        
        # Per-class metrics
        class_metrics = {k: v for k, v in metrics.items() if "class_" in k}
        if class_metrics:
            f.write("PER-CLASS METRICS:\n")
            
            # Group metrics by class
            class_groups = {}
            for k, v in class_metrics.items():
                metric_name, class_id = k.split('class_')
                metric_name = metric_name[:-1]  # Remove trailing underscore
                class_id = int(class_id)
                
                if class_id not in class_groups:
                    class_groups[class_id] = {}
                class_groups[class_id][metric_name] = v
            
            # Write metrics for each class
            for class_id, class_dict in sorted(class_groups.items()):
                f.write(f"\nClass {class_id}:\n")
                for metric_name, value in class_dict.items():
                    f.write(f"  {metric_name.capitalize()}: {value:.4f}\n")
        
        # Detection statistics
        f.write("\nDETECTION STATISTICS:\n")
        f.write(f"Total predicted boxes: {metrics.get('detected_boxes', 0)}\n")
        f.write(f"Total ground truth boxes: {metrics.get('gt_boxes', 0)}\n")
    
    print(f"Saved metrics to {output_path}")


def main(args):
    # Load configuration
    data_config, model_config, training_config = ConfigurationManager.load_config(args.config)
    
    # Create results directory
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = args.device
    
    # Load model
    print(f"Loading model from {model_config.weights_path}")
    model = load_model(model_config, training_config.use_lora, device=device)
    model.to(device)
    
    # Setup data
    eval_loader = setup_data_loader(data_config, batch_size=args.batch_size)
    print(f"Loaded evaluation dataset with {len(eval_loader)} batches")
    
    # Prepare batch function that handles device transfer
    prepare_batch_fn = lambda batch: prepare_batch(batch, device=device)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, eval_loader, prepare_batch_fn, device=device)
    
    # Print evaluation report
    print_evaluation_report(metrics)
    
    # Save metrics to file
    save_metrics_to_file(metrics, save_dir=results_dir)
    
    # Plot metrics
    plot_metrics(metrics, save_dir=results_dir)
    
    # Visualize predictions if requested
    if args.visualize:
        print("Generating visualizations...")
        # Reset data loader
        eval_loader = setup_data_loader(data_config, batch_size=1)
        
        # Visualize predictions
        results = visualize_predictions(
            model, 
            eval_loader, 
            prepare_batch_fn, 
            num_samples=args.num_vis,
            score_threshold=args.score_threshold,
            device=device
        )
        
        # Save visualizations
        vis_dir = os.path.join(results_dir, 'visualizations')
        visualize_results(results, save_dir=vis_dir)
    
    print(f"Evaluation complete. Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GroundingDINO model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--num-vis", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Score threshold for visualizations")
    
    args = parser.parse_args()
    main(args) 