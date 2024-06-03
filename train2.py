import os
import cv2
import csv
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from groundingdino.util.train import load_model, load_image,train_image, annotate
from torchvision.ops import box_convert  
from torchvision.ops import generalized_box_iou  


def box_xyxy_to_cxcywh(boxes):
    """Convert boxes from xyxy to cxcywh format"""
    return box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from cxcywh to xyxy format"""
    return box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Sigmoid focal loss as used in Grounding DINO.
    
    Args:
        inputs: Prediction tensor (unnormalized) of shape [N, *]
        targets: Target tensor (binary) of shape [N, *]
        alpha: Weighting factor for positive examples (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'none' | 'mean' | 'sum'
        
    Returns:
        Computed focal loss
    """
    # Flatten the tensors if needed
    if inputs.ndim > 2:
        inputs = inputs.view(inputs.size(0), -1)
    if targets.ndim > 2:
        targets = targets.view(targets.size(0), -1)
        
    # Convert targets to float for calculations
    targets = targets.to(dtype=torch.float32)
    
    # Apply sigmoid to inputs
    prob = torch.sigmoid(inputs)
    
    # Calculate cross entropy
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    
    # Calculate focal term
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    
    # Apply alpha weighting
    loss = focal_term * ce_loss
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    
    # Apply reduction
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    
    return loss


def create_positive_map(caption_objects, tokenized, tokenizer):
    """
    Create positive mapping between boxes and text tokens
    Returns tensor of shape [B, num_boxes, num_tokens]
    """
    num_tokens = tokenized.input_ids.size(1)
    positive_map = torch.zeros((1, len(caption_objects), num_tokens), 
                             dtype=torch.bool,
                             device=tokenized.input_ids.device)
    
    for box_idx, obj in enumerate(caption_objects):
        # Get token span for this object
        obj_tokens = tokenizer(obj + ".", add_special_tokens=False)['input_ids']
        
        # Find token span in full text
        for i in range(num_tokens):
            if tokenized.input_ids[0][i:i+len(obj_tokens)-1].tolist() == obj_tokens[:-1]:
                positive_map[0, box_idx, i:i+len(obj_tokens)-1] = True
                break
                
    return positive_map



def compute_contrastive_loss(self, 
                           pred_logits: torch.Tensor,  # [B, num_queries, hidden_dim]
                           text_embeddings: torch.Tensor,  # [B, num_tokens, hidden_dim] 
                           text_token_mask: torch.Tensor,  # [B, num_tokens]
                           positive_map: torch.Tensor,  # [B, num_boxes, num_tokens]
                           num_boxes: int,
                           temperature: float = 0.07):
    """
    Compute contrastive loss between predicted box features and text token features
    Args:
        pred_logits: Box feature predictions [B, num_queries, hidden_dim]
        text_embeddings: Text token embeddings [B, num_tokens, hidden_dim]
        text_token_mask: Mask for valid text tokens [B, num_tokens]
        positive_map: Binary matrix mapping boxes to text tokens [B, num_boxes, num_tokens]
        num_boxes: Number of ground truth boxes
        temperature: Temperature parameter for contrastive loss
    """
    # Normalize features
    pred_logits = F.normalize(pred_logits, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.bmm(pred_logits, text_embeddings.transpose(-2, -1)) / temperature
    
    # Mask out padding tokens
    similarity = similarity.masked_fill(~text_token_mask[:, None, :], float('-inf'))
    
    # For each box, get its matching text span
    sim_max, sim_max_idx = similarity.max(dim=-1)  # [B, num_queries]
    
    # Compute contrastive loss 
    pos_mask = positive_map.any(dim=-1)  # [B, num_boxes]
    neg_mask = ~pos_mask
    
    pos_sim = sim_max[pos_mask]
    neg_sim = sim_max[neg_mask]
    
    labels = torch.zeros_like(sim_max, dtype=torch.long)
    labels[pos_mask] = 1
    
    loss = sigmoid_focal_loss(
        sim_max.unsqueeze(-1),
        labels.unsqueeze(-1),
        alpha=0.25,
        gamma=2.0,
        reduction='mean'
    )
    return loss



class GroundingDINODataset(Dataset):
    def __init__(self, img_dir, ann_file):
        """
        Args:
            img_dir (str): Path to image directory
            ann_file (str): Path to annotation CSV
            transform: Optional transform to be applied on image
        """
        self.img_dir = img_dir
        self.annotations = self.read_dataset(img_dir, ann_file)
        self.image_paths = list(self.annotations.keys())
    
    def read_dataset(self, img_dir, ann_file):
        """Read dataset annotations"""
        ann_dict = defaultdict(lambda: defaultdict(list))
        with open(ann_file) as file_obj:
            ann_reader = csv.DictReader(file_obj)
            for row in ann_reader:
                img_path = os.path.join(img_dir, row['image_name'])
                x1 = int(row['bbox_x'])
                y1 = int(row['bbox_y'])
                x2 = x1 + int(row['bbox_width'])
                y2 = y1 + int(row['bbox_height'])
                ann_dict[img_path]['boxes'].append([x1, y1, x2, y2])
                ann_dict[img_path]['phrases'].append(row['label_name'])
        return ann_dict

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and transform image
        image_source, image = load_image(img_path)
        orig_size = image.shape[-2:]
        
        # Get annotations
        boxes = torch.tensor(self.annotations[img_path]['boxes'], dtype=torch.float32)
        phrases = self.annotations[img_path]['phrases']
        
        # Create target dict
        target = {
            'boxes': boxes,
            'phrases': phrases,
            'image_size': orig_size,
            'image_path': img_path
        }
        
        return image, target

class GroundingDINOTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device="cuda",
        class_loss_coef=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        temperature=0.07
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.temperature = temperature

    def prepare_batch(self, batch):
        """Prepare batch for training"""
        images, targets = batch
        
        # Move images to device
        if isinstance(images, (list, tuple)):
            images = [image.to(self.device) for image in images]
        else:
            images = images.to(self.device)
            
        # Prepare text inputs and targets for each image
        processed_targets = []
        captions = []
        
        for target in targets:
            # Create caption from phrases
            phrases = target['phrases']
            caption = ".".join(phrases) + "." ## Prefered in their github readme
            captions.append(caption)
                
            # Create positive map
            boxes = target['boxes'].to(self.device)
            processed_target = {
                'boxes': boxes,
                'phrases': phrases,
                'image_size': target['image_size']
            }
            processed_targets.append(processed_target)
            
        return images, processed_targets, captions

    def compute_loss(self, outputs, targets, captions):
        batch_losses = defaultdict(float)
        batch_size = len(targets)
        
        for idx, (target, caption) in enumerate(zip(targets, captions)):
            # Get predictions for this image
            pred_logits = outputs["pred_logits"][idx]
            pred_boxes = outputs["pred_boxes"][idx]
            text_embeddings = outputs.get("proj_tokens")
            
            # Create positive map for text-box alignment
            positive_map = create_positive_map(
                target['phrases'],
                caption,
                self.model.tokenizer,
                self.device
            )
            
            # Compute contrastive loss
            if text_embeddings is not None:
                contrastive_loss = compute_contrastive_loss(
                    pred_logits.unsqueeze(0),
                    text_embeddings[idx:idx+1],
                    positive_map,
                    len(target_boxes),
                    self.temperature
                )
                batch_losses['contrastive_loss'] += contrastive_loss
            
            h, w = target['image_size']
            scale_fct = torch.tensor([w, h, w, h], device=self.device)
            target_boxes = box_xyxy_to_cxcywh(target['boxes']) / scale_fct
            
            # Compute box losses
            bbox_loss = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum() / len(target_boxes)
            giou_loss = 1 - generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            ).diag().mean()

            batch_losses['bbox_loss'] += bbox_loss
            batch_losses['giou_loss'] += giou_loss
        
        # Average losses over batch
        for k in batch_losses:
            batch_losses[k] /= batch_size
        
        # Combine losses
        total_loss = (
            self.class_loss_coef * batch_losses['contrastive_loss'] +
            self.bbox_loss_coef * batch_losses['bbox_loss'] +
            self.giou_loss_coef * batch_losses['giou_loss']
        )
        
        batch_losses['total_loss'] = total_loss
        return batch_losses

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare batch
        images, targets, captions = self.prepare_batch(batch)
        
        # Forward pass
        outputs = self.model(images, captions=captions)
        
        # Compute losses
        losses = self.compute_loss(outputs, targets, captions)
        
        # Backward pass
        losses['total_loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}

def train(
    model,
    data_dict,
    num_epochs=10,
    batch_size=1,
    learning_rate=1e-4,
    save_dir='weights',
    save_frequency=1
):
    """Main training loop"""
    # Create dataset and dataloader
    train_dataset = GroundingDINODataset(
        data_dict['train_dir'],
        data_dict['train_ann']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x)) 
    )
    
    # Initialize optimizer and trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = GroundingDINOTrainer(model, optimizer)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch)
            
            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            # Print progress
            if batch_idx % 10 == 0:
                loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, {loss_str}")
        
        # Compute epoch averages
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1} complete. Average losses:", 
              ", ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()))
        
        # Save checkpoint
        if (epoch + 1) % save_frequency == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': avg_losses,
            }
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")



if __name__ == "__main__":
    
    data_dict = {
        'train_dir': "multimodal-data/fashion_dataset_subset/images/train",
        'train_ann': "multimodal-data/fashion_dataset_subset/train_annotations2.csv",
        'val_dir': "multimodal-data/fashion_dataset_subset/images/val",
        'val_ann': "multimodal-data/fashion_dataset_subset/val_annotations.csv"
    }
    
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    train(model, data_dict)
