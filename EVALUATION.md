# Model Evaluation

This document provides an overview of the model evaluation framework added to the AViD project.

## Overview

The evaluation framework allows you to:

1. Evaluate trained models using standard object detection metrics
2. Generate visualizations of model predictions
3. Track evaluation metrics during training
4. Run standalone evaluation on saved models

## Key Metrics

The framework calculates the following metrics:

- **Mean Average Precision (mAP)**: The primary metric for object detection performance
- **Precision**: Measures how accurate the positive predictions are
- **Recall**: Measures how well the model finds all the positive samples  
- **F1 Score**: The harmonic mean of precision and recall

These metrics are calculated for each class and IoU threshold, and then averaged.

## Using Evaluation During Training

To enable evaluation during training, update your config file with evaluation settings:

```yaml
training:
  # Enable evaluation during training
  evaluate_during_training: true
  evaluation_frequency: 5  # Evaluate every 5 epochs
  
  # Detailed evaluation configuration
  evaluation:
    iou_thresholds: [0.5, 0.75]  # IoU thresholds for evaluation
    score_threshold: 0.3  # Confidence score threshold
    max_detections: 100  # Maximum detections per image
    metrics_output_dir: "evaluation_results"  # Directory to save results
    generate_visualizations: true  # Generate visualization images
    num_visualization_samples: 10  # Number of samples to visualize
```

Then run training as normal:

```bash
python train.py --config configs/your_config.yaml
```

## Standalone Evaluation

You can evaluate a trained model separately using the evaluation script:

```bash
python evaluate.py --config configs/evaluation_config.yaml --output-dir results/evaluation
```

### Command Line Arguments

- `--config`: Path to configuration file
- `--output-dir`: Directory to save evaluation results
- `--batch-size`: Batch size for evaluation
- `--device`: Device to run evaluation on (cuda/cpu)
- `--visualize`: Enable visualization generation
- `--num-vis`: Number of images to visualize
- `--score-threshold`: Confidence score threshold for predictions

## Evaluation Outputs

The evaluation generates the following outputs:

1. **Metrics Report**: Text file with detailed metrics per class
2. **Visualizations**: Images showing predictions vs. ground truth
3. **Plots**: Graphs of key metrics for easier interpretation
4. **JSON Data**: Machine-readable metrics for further analysis

## Example: Reading Evaluation Results

After evaluation, you can find the results in the specified output directory:

```
evaluation_results/
├── metrics.txt               # Text report of all metrics
├── evaluation_epoch_5.json   # Metrics in JSON format
├── overall_metrics.png       # Plot of overall metrics
├── class_metrics.png         # Plot of per-class metrics
└── visualizations/           # Prediction visualizations
    ├── sample_0.jpg
    ├── sample_1.jpg
    └── ...
```

## Customizing Evaluation

You can customize evaluation by adjusting the parameters in the config file:

- **IoU Thresholds**: Set different thresholds for IoU calculation
- **Score Threshold**: Adjust the confidence threshold for predictions
- **Visualization Settings**: Control the number and type of visualizations

## Implementation Details

The evaluation framework consists of:

1. `groundingdino/util/evaluation.py`: Core evaluation utilities
2. `evaluate.py`: Standalone evaluation script
3. Updates to `train.py`: Integration with training pipeline
4. Updates to `config.py`: Configuration support for evaluation 