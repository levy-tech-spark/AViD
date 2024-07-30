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
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import supervision as sv
from groundingdino.util.class_loss import BCEWithLogitsLoss,MultilabelFocalLoss
from transformers import get_cosine_schedule_with_warmup
from ema_pytorch import EMA
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from groundingdino.util.vl_utils import build_captions_and_token_span,create_positive_map_from_span
from typing import Dict, NamedTuple
from model_utils import freeze_model_layers,print_frozen_status
from torch.optim.lr_scheduler import OneCycleLR


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.stability_eps = 1e-6
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid().clamp(min=self.stability_eps, max=1-self.stability_eps) # [batch_size * num_queries, num_classes]
        pred_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        
        tgt_boxes_list = []
        token_spans=[]
        for t in targets:
            # Convert target boxes to cxcywh and normalize
            h, w = t['size']
            scale_fct = torch.tensor([w, h, w, h], device=pred_bbox.device)
            #t_boxes = box_xyxy_to_cxcywh(t["boxes"]) / scale_fct
            t_boxes = t["boxes"] / scale_fct
            tgt_boxes_list.append(t_boxes)
            token_span=[t['cat2tokenspan'][cls] for cls in t['str_cls_lst']]
            token_spans.append(token_span)
            
        tgt_bbox = torch.cat(tgt_boxes_list)
        tgt_labels_list = []
        #max_length = max(len(tokenizer(caption, return_tensors="pt").input_ids[0]) for caption in captions)
        # To easily match for matrix multiplication below we can also limit output preds to max len and then multiply this is faster though
        max_length = 256
        for span in token_spans:
            positive_map=create_positive_map_from_span(outputs['tokenized'], span, max_text_len=max_length).to(out_prob.device)
            tgt_labels_list.append(positive_map)
                
        tgt_labels = torch.cat(tgt_labels_list, dim=0)
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        normalized_tgt_labels = tgt_labels / (tgt_labels.sum(dim=1, keepdim=True) + self.stability_eps)
    
        # Compute costs using normalized labels
        cost_class = -(out_prob.unsqueeze(1) * normalized_tgt_labels.unsqueeze(0).float()).sum(dim=-1)
        #cost_class = -(out_prob.unsqueeze(1) * tgt_labels.unsqueeze(0).float()).sum(dim=-1) ## shape is [num_queries x num of boxes]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1) ## shape is [num_queries x num of boxes]

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_bbox), box_cxcywh_to_xyxy(tgt_bbox)).clamp(min=-1.0) ## shape is [num_queries x num of boxes]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # Return targetlabels here also to avoid calcualting them during loss
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], tgt_labels



class FirstNMatcher(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]

        token_spans=[]
        for t in targets:
            token_span=[t['cat2tokenspan'][cls] for cls in t['str_cls_lst']]
            token_spans.append(token_span)
            

        tgt_labels_list = []
        #max_length = max(len(tokenizer(caption, return_tensors="pt").input_ids[0]) for caption in captions)
        # To easily match for matrix multiplication below we can also limit output preds to max len and then multiply this is faster though
        max_length = 256
        for span in token_spans:
            positive_map=create_positive_map_from_span(outputs['tokenized'], span, max_text_len=max_length).to(outputs["pred_logits"].device)
            tgt_labels_list.append(positive_map)
                
        tgt_labels = torch.cat(tgt_labels_list, dim=0)
        
        for i in range(bs):
            n = len(targets[i]['boxes'])
            # Match first n predictions to first n targets
            src_idx = torch.arange(n,dtype=torch.int64)
            tgt_idx = torch.arange(n,dtype=torch.int64) 
            indices.append((src_idx, tgt_idx))
            
        return indices,tgt_labels


def build_matcher(set_cost_class,set_cost_bbox,set_cost_giou,name="Hungarian"):
    if name=="Hungarian":
        return HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)
    elif name=="Simple":
        print(f"Building first matcher!!")
        return FirstNMatcher()
    else:
        pass
        
