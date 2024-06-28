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
from groundingdino.util.misc import nested_tensor_from_tensor_list
from torchvision.ops import box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import supervision as sv
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss


def box_xyxy_to_cxcywh(boxes):
    """Convert boxes from xyxy to cxcywh format"""
    return box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from cxcywh to xyxy format"""
    return box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher from DETR
    Computes an assignment between the targets and the predictions of the network
    """
    def __init__(self, class_cost: float = 2, bbox_cost: float = 5, giou_cost: float = 2):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost 
        self.giou_cost = giou_cost
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target boxes
        target_boxes = torch.cat([t["boxes"] for t in targets])
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
        
        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )
        
        # Compute text similarity cost
        pred_logits = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, hidden_dim]
        text_embeddings = outputs.get("proj_tokens", None)
        if text_embeddings is not None:
            text_embeddings = text_embeddings.flatten(0, 1)  # [batch_size * num_tokens, hidden_dim]
            # Normalize features
            pred_logits = F.normalize(pred_logits, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            # Compute similarity
            cost_class = -torch.mm(pred_logits, text_embeddings.transpose(0, 1))
        else:
            cost_class = torch.zeros_like(cost_bbox)

        # Final cost matrix
        C = (
            self.bbox_cost * cost_bbox +
            self.class_cost * cost_class + 
            self.giou_cost * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i].cpu()) 
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(torch.as_tensor(i, dtype=torch.int64), 
                torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def visualize_predictions(model, image, caption, epoch, idx, save_dir):
    """Visualize and save predictions with better annotations"""
    with torch.no_grad():
        outputs = model(image, captions=[caption])
    
    # Convert image tensor to numpy
    img = image.tensors[0].cpu().permute(1,2,0).numpy()
    h, w, _ = img.shape
    
    # Process boxes and logits
    boxes = outputs["pred_boxes"][0]
    logits = outputs["pred_logits"][0].sigmoid().max(dim=1)[0]
    boxes = boxes * torch.tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Create detections
    detections = sv.Detections(xyxy=xyxy)
    
    # Create labels with confidence scores
    labels = [f"{caption.split('.')[0]} {logit:.2f}" for logit in logits]
    
    # Initialize annotator with custom settings
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_scale=0.8,
        text_padding=3
    )
    
    # Convert RGB to BGR and annotate
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(
        scene=img_bgr, 
        detections=detections, 
        labels=labels
    )
    
    # Save
    os.makedirs(f"{save_dir}/epoch_{epoch}", exist_ok=True)
    cv2.imwrite(f"{save_dir}/epoch_{epoch}/pred_{idx}.jpg", annotated_frame)


def create_positive_map_from_phrases(phrases, tokenized, tokenizer):
    """Create positive map between boxes and text tokens"""
    # Get token ids for full text
    token_ids = tokenized.input_ids[0]  # [num_tokens]
    positive_map = torch.zeros(len(phrases), len(token_ids), dtype=torch.bool, 
                            device=token_ids.device)
    
    for i, phrase in enumerate(phrases):
        # Tokenize individual phrase - no special tokens
        phrase_tokens = tokenizer(
            phrase, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids[0]
        
        # Find matching span in full text
        phrase_len = len(phrase_tokens)
        for j in range(len(token_ids) - phrase_len + 1):
            if token_ids[j:j+phrase_len].tolist() == phrase_tokens.tolist():
                positive_map[i, j:j+phrase_len] = True
                break
                
    return positive_map


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
            'image_source': image_source
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
        """Prepare batch according to Grounding DINO specifications"""
        images, targets = batch
        
        # Convert list of images to NestedTensor and move to device
        if isinstance(images, (list, tuple)):
            images = nested_tensor_from_tensor_list(images)  # Convert list to NestedTensor
        images = images.to(self.device)
                
        processed_targets = []
        captions = []
        
        for target in targets:
            phrases = target['phrases']
            # Format caption with periods without spaces
            caption = ".".join(phrases) + "."
            captions.append(caption)
            
            boxes = target['boxes'].to(self.device)
            processed_target = {
                'boxes': boxes,
                'image_size': target['image_size'],
                'image_source': target['image_source'],
                'phrases': phrases
            }
            processed_targets.append(processed_target)
            
        return images, processed_targets, captions

    def compute_loss(self, outputs, targets, captions):
        """Compute losses using model's functions"""
        batch_losses = defaultdict(float)
        batch_size = len(targets)
        
        #Hungarian matching indices
        matcher = HungarianMatcher(
            class_cost=self.class_loss_coef,
            bbox_cost=self.bbox_loss_coef,
            giou_cost=self.giou_loss_coef
        )
        indices = matcher(outputs, targets)
        
        for idx, ((pred_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
            # Get predictions
            pred_boxes = outputs["pred_boxes"][idx][pred_idx]  # [num_matched, 4]
            pred_logits = outputs["pred_logits"][idx]  # [num_queries, max_text_len]
            # Below is same as valid tokens since we have preds logits of shape 900x 256 but if only something like 8 first tokens are valid in out input
            ## rest will be pred as -inf
            valid_mask = ~torch.isinf(pred_logits)
            
            # Box losses
            # Normalize target boxes
            h, w = target['image_size']
            scale_fct = torch.tensor([w, h, w, h], device=self.device)
            target_boxes = box_xyxy_to_cxcywh(target['boxes'][tgt_idx]) / scale_fct
            
            # L1 loss - from their util
            bbox_loss = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum() / len(target_boxes)
            
            # GIoU loss
            giou_loss = 1 - generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            ).diag().mean()
            
            # Classification loss using their sigmoid_focal_loss
            num_queries = pred_logits.shape[0]
            
            # Initialize target tensor - zeros for all queries
            target_labels = torch.zeros_like(pred_logits)  # [num_queries, max_text_len]
            # Create positive map for the text tokens
            positive_map = create_positive_map_from_phrases(
                target['phrases'],
                self.model.tokenizer(captions[idx], return_tensors="pt").to(self.device),
                self.model.tokenizer
            )  # [num_gt, num_tokens]

            # For matched pairs, assign corresponding positive map rows
            for i, j in zip(pred_idx, tgt_idx):
                target_labels[i, :positive_map.shape[1]] = positive_map[j]

            # Use their sigmoid_focal_loss
            class_loss = sigmoid_focal_loss(
                pred_logits[valid_mask].unsqueeze(0),
                target_labels[valid_mask].unsqueeze(0),
                num_boxes=len(target_boxes),
                alpha=0.25,
                gamma=2.0
            )
            
            batch_losses['class_loss'] += class_loss
            batch_losses['bbox_loss'] += bbox_loss
            batch_losses['giou_loss'] += giou_loss
        
        # Average over batch
        for k in batch_losses:
            batch_losses[k] /= batch_size
        
        # Combine with coefficients
        total_loss = (
            self.class_loss_coef * batch_losses['class_loss'] +
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
        outputs = self.model(images, captions=captions)
        losses = self.compute_loss(outputs, targets, captions)
        losses['total_loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}

def train(
    model,
    data_dict,
    num_epochs=10,
    batch_size=2,
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
            if batch_idx % 5 == 0:
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
