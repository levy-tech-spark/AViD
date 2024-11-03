from torchvision.ops import generalized_box_iou  
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import torch
from typing import Dict, NamedTuple
import torch.nn as nn
import torch.nn.functional as F
from groundingdino.util.class_loss import BCEWithLogitsLoss,MultilabelFocalLoss,FocalLoss



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
