import os
import cv2
import csv
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from groundingdino.util.train import load_model, load_image
from torchvision.ops import generalized_box_iou  
from groundingdino.util.misc import nested_tensor_from_tensor_list
from torchvision.ops import generalized_box_iou
import torch.nn as nn
import supervision as sv
from groundingdino.util.class_loss import BCEWithLogitsLoss,MultilabelFocalLoss,FocalLoss
from ema_pytorch import EMA
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from groundingdino.util.vl_utils import build_captions_and_token_span
from typing import Dict, NamedTuple
from groundingdino.util.model_utils import freeze_model_layers,print_frozen_status
from torch.optim.lr_scheduler import OneCycleLR
from matchers import build_matcher
from groundingdino.util.inference import GroundingDINOVisualizer
from groundingdino.util.model_utils import freeze_model_layers, print_frozen_status
from groundingdino.util.lora import get_lora_weights
from datetime import datetime

# Ignore tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SetCriterion(nn.Module):
    
    def __init__(self, num_classes, matcher, eos_coef, losses,loss_type= 'focal'):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef  # Will use this to weight no-object loss
        self.losses = losses
        self.matcher = matcher
        if loss_type == 'bce':
            self.cls_loss = BCEWithLogitsLoss(eos_coef=eos_coef)
        elif loss_type == 'multilabelfocal':
            self.cls_loss = MultilabelFocalLoss(eos_coef=eos_coef)
        elif loss_type == 'focal':
            self.cls_loss = FocalLoss(eos_coef=eos_coef)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    class ProcessedTargets(NamedTuple):
        """Container for preprocessed targets and masks"""
        pred_logits: torch.Tensor     # All predictions
        target_labels: torch.Tensor   # All targets
        text_mask: torch.Tensor       # Only valid text mask
        valid_mask: torch.Tensor      # Combined mask src and targets indices along with text masks 
        device: torch.device

    def preprocess_targets(self, outputs: Dict[str, torch.Tensor],
                        cls_labels: torch.Tensor,
                        indices: list) -> ProcessedTargets:
        """
        Preprocess targets and create masks for loss computation
        """
        bs, num_queries, num_classes = outputs['pred_logits'].shape
        text_mask = outputs['text_mask']  # [bs, num_classes]
        device = outputs['pred_logits'].device
        
        # Create target tensors
        target_mask = torch.zeros((bs, num_queries, num_classes), dtype=torch.bool, device=device)
        target_labels = torch.zeros((bs, num_queries, num_classes), dtype=cls_labels.dtype, device=device)
        
        # Fill target tensors
        offset = 0
        for batch_idx, (pred_indices, tgt_indices) in enumerate(indices):
            target_mask[batch_idx, pred_indices] = text_mask[batch_idx]
            num_targets = len(tgt_indices)
            batch_tgt_labels = cls_labels[offset:offset + num_targets]
            target_labels[batch_idx, pred_indices] = batch_tgt_labels[tgt_indices]
            offset += num_targets
        
        return self.ProcessedTargets(
            pred_logits=outputs['pred_logits'],
            target_labels=target_labels,
            text_mask=text_mask,
            valid_mask=target_mask,
            device=device
        )

    def loss_labels(self, outputs, targets, indices, log=False, **kwargs):
        """Compute the classification loss"""
        assert 'pred_logits' in outputs
        
        # Get processed targets
        tgt_labels = kwargs['cls_labels']
        processed = self.preprocess_targets(outputs, tgt_labels, indices)
        
        # Compute loss using the new interface
        loss = self.cls_loss(
            preds=processed.pred_logits,
            targets=processed.target_labels,
            valid_mask=processed.valid_mask,
            text_mask=processed.text_mask
        )
        
        losses = {'loss_ce': loss}
        
        # Compute accuracies if logging is enabled
        if log:
            acc_dict=self._compute_accuracy(processed=processed)    
            losses.update(acc_dict)
        
        return losses

    def _compute_accuracy(self, processed: ProcessedTargets):
        with torch.no_grad():
            bs, num_queries, num_classes = processed.pred_logits.shape
            
            # For matched queries text token accuracy
            valid_preds = processed.pred_logits[processed.valid_mask.bool()]
            valid_targets = processed.target_labels[processed.valid_mask.bool()]
            
            if valid_preds.numel() > 0:
                token_acc = (valid_preds.sigmoid() > 0.5) == valid_targets
                token_acc = token_acc.float().mean() * 100
                
                # Reshape masks to match logits
                unmatched_mask = ~processed.valid_mask.bool()  # bs x queries x 256
                text_mask = processed.text_mask.unsqueeze(1).expand(-1, num_queries, -1)  # Expand to match shape
                
                # Apply both masks
                total_mask = unmatched_mask & text_mask
                unmatched_preds = processed.pred_logits[total_mask]
                background_acc = (unmatched_preds.sigmoid() < 0.5).float().mean() * 100

            else:
                token_acc = background_acc = torch.tensor(0.0, device=processed.device)
                
            return {
                'Matched_Token_Accuracy': token_acc,
                'UnMatched_Token_Accuracy': background_acc
            }

    def loss_boxes(self, outputs, targets, indices, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        # Get target boxes and normalize them
        tgt_boxes_list = []
        for t, (_, i) in zip(targets, indices):
            # Convert and normalize the matched target boxes
            h, w = t['size']
            scale_fct = torch.tensor([w, h, w, h], device=src_boxes.device)
            t_boxes = t["boxes"][i] / scale_fct
            tgt_boxes_list.append(t_boxes)
            
        target_boxes = torch.cat(tgt_boxes_list, dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        num_boxes=target_boxes.shape[0]
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    
    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """ 
        # Retrieve the matching between the outputs of the last layer and the targets also returning cls labels to avoid caclulating it during loss
        indices, cls_labels = self.matcher(outputs, targets)
        #print(f"indeices of hungarian matcher are {indices}")
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, cls_labels=cls_labels))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class GroundingDINODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        """
        Args:
            img_dir (str): Path to image directory
            ann_file (str): Path to annotation CSV
            transforms: Optional transform to be applied
    
        """
        self.img_dir = img_dir
        self.transforms = transforms
        self.annotations = self.read_dataset(img_dir, ann_file)
        self.image_paths = list(self.annotations.keys())

    def read_dataset(self, img_dir, ann_file):
        """
        Read dataset annotations and convert to [x,y,w,h] format
        """
        ann_dict = defaultdict(lambda: defaultdict(list))
        with open(ann_file) as file_obj:
            ann_reader = csv.DictReader(file_obj)
            for row in ann_reader:
                img_path = os.path.join(img_dir, row['image_name'])
                # Store in [x,y,w,h] format directly
                x = int(row['bbox_x'])
                y = int(row['bbox_y'])
                w = int(row['bbox_width'])
                h = int(row['bbox_height'])
                
                # Convert to center format [cx,cy,w,h]
                cx = x + w/2
                cy = y + h/2
                ann_dict[img_path]['boxes'].append([cx, cy, w, h])
                ann_dict[img_path]['phrases'].append(row['label_name'])
        return ann_dict
    

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load and transform image
        image_source, image = load_image(img_path)
        h, w = image_source.shape[0:2]  


        boxes = torch.tensor(self.annotations[img_path]['boxes'], dtype=torch.float32)
        str_cls_lst = self.annotations[img_path]['phrases']
        
        # Create caption mapping and format
        caption_dict = {item: idx for idx, item in enumerate(str_cls_lst)}
        captions,cat2tokenspan = build_captions_and_token_span(str_cls_lst,force_lowercase=True)
        classes = torch.tensor([caption_dict[p] for p in str_cls_lst], dtype=torch.int64)

        target = {
            'boxes': boxes,  # Already in [cx,cy,w,h] format
            'size': torch.as_tensor([int(h), int(w)]),
            'orig_img': image_source,  
            'str_cls_lst': str_cls_lst,  
            'caption': captions,
            'labels': classes, 
            'cat2tokenspan': cat2tokenspan
        }

        return image, target
    

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
        max_txt_len=256
    ):
        self.model = model.to(device)
        self.device = device
        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.0  # Removed for overfitting
        )
        
        # Initialize scheduler with warmup
        if lr_scheduler=="onecycle":
            total_steps = num_steps_per_epoch * num_epochs
            warmup_steps = num_steps_per_epoch * warmup_epochs  
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
        """Prepare batch according to Grounding DINO specifications"""
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
        # Log gradients
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
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': get_lora_weights(model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'losses': losses,}
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'ema_state_dict': self.ema_model.state_dict() if self.use_ema else None,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'losses': losses,
            }
        torch.save(checkpoint, path)

def train(
    model,
    data_dict,
    num_epochs=200,
    batch_size=2,
    learning_rate=1e-3,
    save_dir='weights',
    save_frequency=5,
    warmup_epochs=5,
    use_lora=False
):
    

    save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M"))
    
    # Create directory
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = GroundingDINODataset(
        data_dict['train_dir'],
        data_dict['train_ann']
    )

    val_dataset = GroundingDINODataset(
        data_dict['val_dir'],
        data_dict['val_ann']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=lambda x: tuple(zip(*x)) 
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    
    visualizer = GroundingDINOVisualizer(save_dir=save_dir)
    
    # if we are using lora then it is takien care of while setting up lora
    if not use_lora:
       print(f"Freezing most of model except few layers!! ")
       freeze_model_layers(model)
    
    print_frozen_status(model)

    for epoch in range(num_epochs):  
        ## Do visualization on val dataset passed as input loop through it
        if epoch % 5 == 0:
            visualizer.visualize_epoch(model, val_loader, epoch, trainer.prepare_batch)
        
        epoch_losses = defaultdict(list)

        for batch_idx, batch in enumerate(train_loader):
            
            losses = trainer.train_step(batch)
            
            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            if batch_idx % 5 == 0:
                loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, {loss_str}")
                print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
            break
        
        
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1} complete. Average losses:", ", ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()))

        if (epoch + 1) % save_frequency == 0:
            trainer.save_checkpoint(
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                epoch,
                avg_losses,
                use_lora=use_lora
            )

            
if __name__ == "__main__":
    
    data_dict = {
        'train_dir': "multimodal-data/fashion_dataset_subset/images/train",
        'train_ann': "multimodal-data/fashion_dataset_subset/train_annotations.csv",
        'val_dir': "multimodal-data/fashion_dataset_subset/images/val",
        'val_ann': "multimodal-data/fashion_dataset_subset/val_annotations.csv"
    }
    use_lora = False
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth",use_lora=use_lora)
    train(model, data_dict, use_lora=False) 


##
##Frozen Parameters: 172,972,294 (99.77%)
##Trainable Parameters: 391,232 (0.23%)
