import os
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from groundingdino.util.train import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from ema_pytorch import EMA
from typing import Dict, NamedTuple
from groundingdino.util.model_utils import freeze_model_layers,print_frozen_status
from torch.optim.lr_scheduler import OneCycleLR
from groundingdino.util.matchers import build_matcher
from groundingdino.util.inference import GroundingDINOVisualizer
from groundingdino.util.model_utils import freeze_model_layers, print_frozen_status
from groundingdino.util.lora import get_lora_optimizer_params, verify_only_lora_trainable
from datetime import datetime
import yaml
from typing import Dict, Optional, Any
from groundingdino.datasets.dataset import GroundingDINODataset
from groundingdino.util.losses import SetCriterion
from config import ConfigurationManager, DataConfig, ModelConfig
from peft import get_peft_model_state_dict
from groundingdino.util.evaluation import evaluate_model, print_evaluation_report, visualize_predictions
import json
import cv2

# Ignore tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
    

def setup_model(model_config: ModelConfig, use_lora: bool=False) -> torch.nn.Module:
    return load_model(
        model_config.config_path,
        model_config.weights_path,
        use_lora=use_lora,
    )

def setup_data_loaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:

    train_dataset = GroundingDINODataset(
        config.train_dir,
        config.train_ann
    )
    
    val_dataset = GroundingDINODataset(
        config.val_dir,
        config.val_ann
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Keep batch size 1 for validation
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader
    

class GroundingDINOTrainer:
    def __init__(
        self,
        model,
        device="cuda",
        ema_decay=0.999,
        ema_update_after_step=150,
        ema_update_every=20,
        warmup_epochs=5,
        class_loss_coef=1.0,
        bbox_loss_coef=5.0,  
        giou_loss_coef=1.0,  
        learning_rate=2e-4,   
        use_ema=False,      
        num_epochs=500,
        num_steps_per_epoch=None,
        lr_scheduler="onecycle",
        eos_coef=0.1,
        max_txt_len=256,
        use_lora=False
    ):
        self.model = model.to(device)
        self.device = device
        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        if use_lora:
            lora_params=get_lora_optimizer_params(model)
            self.optimizer = torch.optim.AdamW(
                lora_params,
                lr=learning_rate,
                #weight_decay=1e-4  # Removed for overfitting
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4  # Removed for overfitting
            )
        
        # Initialize scheduler with warmup
        if lr_scheduler=="onecycle":
            total_steps = num_steps_per_epoch * num_epochs
            #warmup_steps = num_steps_per_epoch * warmup_epochs  
            #self.scheduler = get_cosine_schedule_with_warmup(
            #    self.optimizer,
            #    num_warmup_steps=warmup_steps,
            #    num_training_steps=total_steps
            #)
            # One Cycle LR with warmup
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.1,  # 10% of training for warmup
                div_factor=25,
                final_div_factor=1e4,
                anneal_strategy='cos'
            )
        else:
            # Simple step scheduler
            total_steps = num_steps_per_epoch * num_epochs
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=total_steps//20, 
                gamma=0.5
            )
        
        # Initialize EMA
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMA(
                model,
                beta=ema_decay,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every
            )

        self.matcher=build_matcher(set_cost_class=class_loss_coef*2,
            set_cost_bbox=bbox_loss_coef,
            set_cost_giou=giou_loss_coef)
        
        losses = ['labels', 'boxes']
        self.weights_dict= {'loss_ce': class_loss_coef, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        # Give more weightage to bobx loss in loss calculation compared to matcher 
        self.weights_dict_loss = {'loss_ce': class_loss_coef, 'loss_bbox': bbox_loss_coef*2, 'loss_giou': giou_loss_coef}
        self.criterion = SetCriterion(max_txt_len, self.matcher, eos_coef, losses)
        self.criterion.to(device)

    def prepare_batch(self, batch):
        images, targets = batch
        # Convert list of images to NestedTensor and move to device
        if isinstance(images, (list, tuple)):
            images = nested_tensor_from_tensor_list(images)  # Convert list to NestedTensor
        images = images.to(self.device)

        captions=[]
        for target in targets:
            target['boxes']=target['boxes'].to(self.device)
            target['size']=target['size'].to(self.device)
            target['labels']=target['labels'].to(self.device)
            captions.append(target['caption'])
            
        return images, targets, captions

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        #self.get_ema_model().train()
        self.optimizer.zero_grad() 
        # Prepare batch
        images, targets, captions = self.prepare_batch(batch)
        outputs = self.model(images, captions=captions)
        loss_dict=self.criterion(outputs, targets, captions=captions, tokenizer=self.model.tokenizer)
        total_loss = sum(loss_dict[k] * self.weights_dict_loss[k] for k in loss_dict.keys() if k in self.weights_dict_loss)
        ## backward pass
        total_loss.backward()
        loss_dict['total_loss']=total_loss
        #total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)
        #print(f"Gradient norm: {total_norm:.4f}")
        self.optimizer.step()
        
        # Step scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update EMA model
        if self.use_ema:
            self.ema_model.update()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}


    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_losses = defaultdict(float)
        num_batches = 0
        
        for batch in val_loader:
            images, targets, captions = self.prepare_batch(batch)
            outputs = self.model(images, captions=captions)
            
            # Calculate losses
            loss_dict = self.criterion(outputs, targets, captions=captions, tokenizer=self.model.tokenizer)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                val_losses[k] += v.item()
                
            val_losses['total_loss'] += sum(loss_dict[k] * self.weights_dict[k] 
                                        for k in loss_dict.keys() if k in self.weights_dict_loss).item()
            num_batches += 1

        # Average losses
        return {k: v/num_batches for k, v in val_losses.items()}


    def get_ema_model(self):
        """Return EMA model for evaluation"""
        return self.ema_model.ema_model

    def save_checkpoint(self, path, epoch, losses, use_lora=False):
        """Save checkpoint with EMA and scheduler state""" 
        if use_lora:
            lora_state_dict = get_peft_model_state_dict(self.model)
            print(lora_state_dict)
            checkpoint = {
            'epoch': epoch,
            'model': lora_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'losses': losses,}
        else:
            checkpoint = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'ema_state_dict': self.ema_model.state_dict() if self.use_ema else None,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'losses': losses,
            }
        torch.save(checkpoint, path)

def train(config_path: str, save_dir: Optional[str] = None) -> None:
    """
    Main training function with configuration management
    
    Args:
        config_path: Path to the YAML configuration file
        save_dir: Optional override for save directory
    """

    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

    model = setup_model(model_config, training_config.use_lora)
    
    if save_dir:
        training_config.save_dir = save_dir
    
    # Setup save directory with timestamp
    save_dir = os.path.join(
        training_config.save_dir,
        datetime.now().strftime("%Y%m%d_%H%M")
    )
    os.makedirs(save_dir, exist_ok=True)
    
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'data': vars(data_config),
            'model': vars(model_config),
            'training': vars(training_config)
        }, f, default_flow_style=False)
    
    train_loader, val_loader = setup_data_loaders(data_config)

    steps_per_epoch = len(train_loader.dataset) // data_config.batch_size
    
    visualizer = GroundingDINOVisualizer(save_dir=save_dir)
    
    if not training_config.use_lora:
        print("Freezing most of model except few layers!")
        freeze_model_layers(model)
    
    else:
         print( f"Is only Lora trainable?  {verify_only_lora_trainable(model)} ")

    print_frozen_status(model)

    trainer = GroundingDINOTrainer(
        model,
        num_steps_per_epoch=steps_per_epoch,
        num_epochs=training_config.num_epochs,
        warmup_epochs=training_config.warmup_epochs,
        learning_rate=training_config.learning_rate,
        use_lora=training_config.use_lora
    )   
    # Training loop
    print(f"Starting training for {training_config.num_epochs} epochs")
    
    # Track best validation metrics
    best_metrics = {
        'loss': float('inf'),
        'mean_ap': 0.0
    }
    
    # Save training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'evaluations': {}
    }
    
    for epoch in range(training_config.num_epochs):
        train_losses = defaultdict(float)
        
        # Train epoch
        print(f"Epoch {epoch+1}/{training_config.num_epochs}")
        for i, batch in enumerate(train_loader):
            loss_dict = trainer.train_step(batch)
            
            # Log losses
            for k, v in loss_dict.items():
                train_losses[k] += v
                
            if i % 10 == 0:
                print(f"Step {i}/{len(train_loader)}: Loss = {loss_dict['total_loss']:.4f}")
        
        # Average training losses
        avg_train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
        
        # Validate
        val_losses = trainer.validate(val_loader)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} summary:")
        print(f"  Train loss: {avg_train_losses['total_loss']:.4f}")
        print(f"  Val loss: {val_losses['total_loss']:.4f}")
        
        # Update history
        history['train_loss'].append(avg_train_losses['total_loss'])
        history['val_loss'].append(val_losses['total_loss'])
        
        # Save checkpoint
        if (epoch + 1) % training_config.save_frequency == 0:
            checkpoint_path = os.path.join(
                training_config.save_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1, val_losses, training_config.use_lora)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save model if it's the best so far
        if val_losses['total_loss'] < best_metrics['loss']:
            best_metrics['loss'] = val_losses['total_loss']
            best_checkpoint_path = os.path.join(
                training_config.save_dir, "best_model.pth"
            )
            trainer.save_checkpoint(best_checkpoint_path, epoch + 1, val_losses, training_config.use_lora)
            print(f"Saved best model to {best_checkpoint_path}")
        
        # Visualize predictions periodically
        if (epoch + 1) % training_config.visualization_frequency == 0:
            print("Generating visualizations...")
            visualizer.visualize_epoch(
                model=trainer.model, 
                val_loader=val_loader, 
                epoch=epoch+1, 
                prepare_data=trainer.prepare_batch
            )
        
        # Evaluate model using our evaluation metrics
        if (training_config.evaluate_during_training and 
            training_config.evaluation and 
            (epoch + 1) % training_config.evaluation_frequency == 0):
            
            print("Evaluating model...")
            # Prepare batch function
            prepare_batch_fn = lambda batch: trainer.prepare_batch(batch)
            
            # Run evaluation
            evaluation_metrics = evaluate_model(
                model=trainer.model,
                data_loader=val_loader,
                prepare_batch_fn=prepare_batch_fn
            )
            
            # Print evaluation report
            print_evaluation_report(evaluation_metrics)
            
            # Store evaluation metrics in history
            history['evaluations'][epoch + 1] = evaluation_metrics
            
            # Save evaluation metrics to a file
            eval_metrics_file = os.path.join(training_config.evaluation.metrics_output_dir, f"evaluation_epoch_{epoch+1}.json")
            with open(eval_metrics_file, 'w') as f:
                json.dump(evaluation_metrics, f, indent=2)
                
            # Update best metrics if this is the best model by mAP
            if evaluation_metrics.get('mean_ap', 0) > best_metrics['mean_ap']:
                best_metrics['mean_ap'] = evaluation_metrics.get('mean_ap', 0)
                best_map_path = os.path.join(
                    training_config.save_dir, "best_map_model.pth"
                )
                trainer.save_checkpoint(best_map_path, epoch + 1, val_losses, training_config.use_lora)
                print(f"Saved model with best mAP ({best_metrics['mean_ap']:.4f}) to {best_map_path}")
                
            # Generate visualizations if enabled
            if training_config.evaluation.generate_visualizations:
                print("Generating evaluation visualizations...")
                results = visualize_predictions(
                    model=trainer.model,
                    data_loader=val_loader,
                    prepare_batch_fn=prepare_batch_fn,
                    num_samples=training_config.evaluation.num_visualization_samples,
                    score_threshold=training_config.evaluation.score_threshold
                )
                
                # Save visualizations
                if results:
                    vis_eval_dir = os.path.join(training_config.evaluation.metrics_output_dir, f"visualizations_epoch_{epoch+1}")
                    os.makedirs(vis_eval_dir, exist_ok=True)
                    
                    for i, result in enumerate(results):
                        img = result['image'].copy()
                        pred_boxes = result['pred_boxes']
                        pred_scores = result['pred_scores']
                        gt_boxes = result['gt_boxes']
                        
                        # Draw ground truth in green
                        for box in gt_boxes:
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw predictions in red
                        for j, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Save image
                        cv2.imwrite(
                            os.path.join(vis_eval_dir, f"sample_{i}.jpg"), 
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        )
    
    # Save training history
    history_path = os.path.join(training_config.save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training complete!")
    print(f"Best validation loss: {best_metrics['loss']:.4f}")
    print(f"Best validation mAP: {best_metrics['mean_ap']:.4f}")

if __name__ == "__main__":
    train('configs/train_config.yaml')
