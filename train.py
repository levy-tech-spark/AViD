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
from transformers import get_cosine_schedule_with_warmup
from ema_pytorch import EMA

# Ignore tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def box_xyxy_to_cxcywh(boxes):
    """Convert boxes from xyxy to cxcywh format"""
    return box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from cxcywh to xyxy format"""
    return box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')


class HungarianMatcher(nn.Module):
    
    def __init__(self, class_cost: float = 2, bbox_cost: float = 3, giou_cost: float = 2):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost 
        self.giou_cost = giou_cost
        
    @torch.no_grad()
    def forward(self, outputs, targets,captions,tokenizer):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Get predicted boxes
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Process target boxes
        target_boxes_list = []
        for t in targets:
            # Convert target boxes to cxcywh and normalize
            h, w = t['image_size']
            scale_fct = torch.tensor([w, h, w, h], device=pred_boxes.device)
            t_boxes = box_xyxy_to_cxcywh(t["boxes"]) / scale_fct
            target_boxes_list.append(t_boxes)
            
        target_boxes = torch.cat(target_boxes_list)
        
        # Compute the L1 cost between boxes (now in same coordinate system)
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
        
        # Compute the giou cost between boxes
        # Convert both to xyxy for GIoU computation (still normalized)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )
        
        # Compute text similarity cost
        pred_logits = outputs["pred_logits"].flatten(0, 1)
        valid_mask = ~torch.isinf(pred_logits)
        # Get only valid token predictions
        pred_probs = pred_logits[valid_mask].sigmoid()  # [num_valid_tokens]
        pred_probs= pred_probs.reshape(bs * num_queries, -1)
        
        text_embeddings = outputs.get("proj_tokens", None)
        
        if text_embeddings is not None:
            text_embeddings = text_embeddings.flatten(0, 1)  
            # Normalize features
            pred_logits = F.normalize(pred_logits, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            cost_class = -torch.mm(pred_logits, text_embeddings.transpose(0, 1))
        else:
            #cost_class = torch.zeros_like(cost_bbox)
            all_positive_maps = []
            max_length = max(len(tokenizer(caption, return_tensors="pt").input_ids[0]) 
                            for caption in captions)
            
            for target, caption in zip(targets, captions):
                tokens = tokenizer(caption, return_tensors="pt").to(pred_probs.device)
                positive_map = create_positive_map_from_phrases(
                    target['phrases'],
                    tokens,
                    tokenizer,
                    max_length=max_length
                )
                all_positive_maps.append(positive_map)
                
            positive_maps = torch.cat(all_positive_maps, dim=0)
            
            # Ensure pred_probs matches token dimension
            if pred_probs.shape[1] != max_length:
                new_pred_probs = torch.zeros(
                    pred_probs.shape[0], 
                    max_length, 
                    device=pred_probs.device,
                    dtype=pred_probs.dtype
                )
                min_length = min(pred_probs.shape[1], max_length)
                new_pred_probs[:, :min_length] = pred_probs[:, :min_length]
                pred_probs = new_pred_probs
            
            positive_maps = positive_maps.to(pred_probs.dtype).to(pred_probs)
            
            cost_class = -(pred_probs @ positive_maps.t())
            pos_token_sum = positive_maps.sum(dim=1)
            cost_class = cost_class / torch.clamp(pos_token_sum, min=1)


        # Final cost matrix
        temperature = 0.1
        C = (
            self.bbox_cost * cost_bbox +
            self.class_cost * cost_class / temperature + 
            self.giou_cost * cost_giou
        )
        #k = 5  
        #values, indices = torch.topk(C[:,1], k=k, largest=False)  # False for smallest values
        #print(f"Top {k} smallest values:", values)
        #print(f"Their indices:", indices)

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i].cpu()) 
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(torch.as_tensor(i, dtype=torch.int64), 
                torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        

class GroundingDINOVisualizer:
    def __init__(self, save_dir, visualize_frequency=20):
        self.save_dir = save_dir
        self.visualize_frequency = visualize_frequency
        self.pred_annotator = sv.BoxAnnotator(
            color=sv.Color.red(),
            thickness=8,
            text_scale=0.8,
            text_padding=3
        )
        self.gt_annotator = sv.BoxAnnotator(
            color=sv.Color.green(),
            thickness=2,
            text_scale=0.8,
            text_padding=3
        )

    def extract_phrases(self, logits, tokenized, tokenizer, text_threshold=0.2):
        """Extract phrases from logits using tokenizer
        Args:
            logits (torch.Tensor): Prediction logits [num_queries, seq_len]
            tokenized: Tokenized text output
            tokenizer: Model tokenizer
            text_threshold: Confidence threshold for token selection
        """
        phrases = []
        token_ids = tokenized.input_ids[0]
        
        for logit in logits:
            # Create mask for tokens above threshold
            text_mask = logit > text_threshold
            
            # Find valid token positions
            valid_tokens = []
            for idx, (token_id, mask) in enumerate(zip(token_ids, text_mask)):
                # Skip special tokens
                if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    continue
                if mask:
                    valid_tokens.append(token_id.item())
            
            if valid_tokens:
                phrase = tokenizer.decode(valid_tokens)
                conf = logit.max().item()
                phrases.append(f"{phrase} ({conf:.2f})")
            
        return phrases

    def visualize_epoch(self, model, val_loader, epoch, prepare_data):
        model.eval()
        save_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                images, targets, captions = prepare_data(batch)
                outputs = model(images, captions=captions)

                img = targets[0]["image_source"]
                h, w, _ = img.shape
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Get predictions & filter by confidence
                pred_logits = outputs["pred_logits"][0].cpu().sigmoid()
                pred_boxes = outputs["pred_boxes"][0].cpu()
                
                # Filter confident predictions
                scores = pred_logits.max(dim=1)[0]
                mask = scores > 0.3  # Box threshold
                
                filtered_boxes = pred_boxes[mask]
                filtered_logits = pred_logits[mask]

                # Get phrase predictions
                tokenized = model.tokenizer(captions[0], return_tensors="pt")
                phrases = self.extract_phrases(filtered_logits, tokenized, model.tokenizer)

                # Draw predictions
                if len(filtered_boxes):
                    boxes = filtered_boxes * torch.tensor([w, h, w, h])
                    xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                    
                    detections = sv.Detections(xyxy=xyxy)
                    img_bgr = self.pred_annotator.annotate(
                        scene=img_bgr,
                        detections=detections,
                        labels=phrases
                    )

                # Draw ground truth
                if "boxes" in targets[0]:
                    gt_xyxy = targets[0]["boxes"].cpu().numpy()
                    gt_detections = sv.Detections(xyxy=gt_xyxy)
                    img_bgr = self.gt_annotator.annotate(
                        scene=img_bgr,
                        detections=gt_detections,
                        labels=targets[0].get("phrases", None)
                    )

                cv2.imwrite(f"{save_dir}/val_pred_{idx}.jpg", img_bgr)

                if idx >= self.visualize_frequency:
                    break


def create_positive_map_from_phrases(phrases, tokenized, tokenizer, max_length=None):
    """Create positive map between boxes and text tokens"""
    # Get token IDs and handle padding if needed
    token_ids = tokenized.input_ids[0]  # [seq_len]
    curr_length = token_ids.shape[0]
    
    if max_length is not None and curr_length < max_length:
        # Pad token_ids to max_length
        padding = torch.full(
            (max_length - curr_length,),
            tokenizer.pad_token_id,
            dtype=token_ids.dtype,
            device=token_ids.device
        )
        token_ids = torch.cat([token_ids, padding])
    
    # Initialize positive map
    positive_map = torch.zeros(
        len(phrases), 
        len(token_ids), 
        dtype=torch.bool,
        device=token_ids.device
    )
    
    for i, phrase in enumerate(phrases):
        # Tokenize individual phrase
        phrase_tokens = tokenizer(
            phrase,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids[0].to(token_ids.device)
        
        phrase_len = len(phrase_tokens)
        
        # Find matching span in token_ids
        for j in range(len(token_ids) - phrase_len + 1):
            if torch.equal(token_ids[j:j+phrase_len], phrase_tokens):
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
        device="cuda",
        class_loss_coef=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        learning_rate=1e-5,
        ema_decay=0.99999,
        ema_update_after_step=500,
        ema_update_every=20,
        warmup_epochs=5,
        num_epochs=100,
        num_steps_per_epoch=None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1},
                {'params': model.transformer.parameters()},
            ],
            lr=learning_rate,
            #weight_decay=0.0001
        )
        
        # Initialize scheduler with warmup
        if num_steps_per_epoch:
            total_steps = num_steps_per_epoch * num_epochs
            warmup_steps = num_steps_per_epoch * warmup_epochs
            
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
        
        # Initialize EMA
        self.ema_model = EMA(
            model,
            beta=ema_decay,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every
        )
        
        self.matcher = HungarianMatcher(
            class_cost=self.class_loss_coef,
            bbox_cost=self.bbox_loss_coef,
            giou_cost=self.giou_loss_coef
        )
        

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
        indices = self.matcher(outputs, targets,captions,self.model.tokenizer)
        
        for idx, ((pred_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
            if len(pred_idx) == 0:
                continue
            # Get predictions
            pred_boxes = outputs["pred_boxes"][idx][pred_idx]  # [num_matched, 4]
            pred_logits = outputs["pred_logits"][idx][pred_idx]  # [num_matched, max_text_len]
            # Below is same as valid tokens since we have preds logits of shape 900x 256 but if only something like 8 first tokens are valid in out input
            ## rest will be pred as -inf
            valid_mask = ~torch.isinf(pred_logits)
            
            # Box losses
            # Normalize target boxes
            h, w = target['image_size']
            scale_fct = torch.tensor([w, h, w, h], device=self.device)
            target_boxes = box_xyxy_to_cxcywh(target['boxes'][tgt_idx]) / scale_fct
            
            # L1 loss - from their util
            bbox_loss = torch.clamp(F.l1_loss(pred_boxes, target_boxes, reduction='none'),max=10.0).sum() / len(target_boxes)

            # GIoU loss
            giou_loss = 1 - torch.clamp(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_boxes),
                    box_cxcywh_to_xyxy(target_boxes)
                ).diag(),
                min=-1.0,
                max=1.0
            ).mean()
            
            # Classification loss using their sigmoid_focal_loss
            num_queries = pred_logits.shape[0]
            
            # Initialize target tensor - zeros for all queries
            target_labels = torch.zeros_like(pred_logits)  # [num_matched, max_text_len]
            # Create positive map for the text tokens
            positive_map = create_positive_map_from_phrases(
                target['phrases'],
                self.model.tokenizer(captions[idx], return_tensors="pt").to(self.device),
                self.model.tokenizer
            )  # [num_gt, num_tokens]

            # For matched pairs, assign corresponding positive map rows
            for i, j in enumerate(tgt_idx):
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
        #self.get_ema_model().train()
        
        self.optimizer.zero_grad()
        
        # Prepare batch
        images, targets, captions = self.prepare_batch(batch)
        outputs = self.model(images, captions=captions)
        losses = self.compute_loss(outputs, targets, captions)
        ## backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        self.optimizer.step()
        
        # Step scheduler if it exists
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update EMA model
        self.ema_model.update()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    
    def get_ema_model(self):
        """Return EMA model for evaluation"""
        return self.ema_model.ema_model

    def save_checkpoint(self, path, epoch, losses):
        """Save checkpoint with EMA and scheduler state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'losses': losses,
        }
        torch.save(checkpoint, path)

def train(
    model,
    data_dict,
    num_epochs=100,
    batch_size=2,
    learning_rate=1e-5,
    save_dir='weights',
    save_frequency=1,
    warmup_epochs=5,
):
    
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
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x)) 
    )
    
    steps_per_epoch = len(train_dataset) // batch_size
    
    trainer = GroundingDINOTrainer(model,
                                    num_steps_per_epoch=steps_per_epoch,
                                    num_epochs=num_epochs,
                                    warmup_epochs=warmup_epochs,
                                    learning_rate=learning_rate)
    
    visualizer = GroundingDINOVisualizer(save_dir="visualizations")
    
    for epoch in range(num_epochs):
        
        ## Do visualization on val dataset passed as input loop through it
        visualizer.visualize_epoch(trainer.get_ema_model(), val_loader, epoch, trainer.prepare_batch)
        
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            losses = trainer.train_step(batch)
            
            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            if batch_idx % 5 == 0:
                loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, {loss_str}")
            
        
        # Compute epoch averages
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1} complete. Average losses:", 
              ", ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()))
        
        if (epoch + 1) % save_frequency == 0:
            
            trainer.save_checkpoint(
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch,
                avg_losses
            )



if __name__ == "__main__":
    
    data_dict = {
        'train_dir': "multimodal-data/fashion_dataset_subset/images/train",
        'train_ann': "multimodal-data/fashion_dataset_subset/train_annotations2.csv",
        'val_dir': "multimodal-data/fashion_dataset_subset/images/val",
        'val_ann': "multimodal-data/fashion_dataset_subset/val_annotations.csv"
    }
    
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    train(model, data_dict)
