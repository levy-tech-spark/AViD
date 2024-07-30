import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class ClassificationLoss(ABC):
    """Base class for classification losses"""
    @abstractmethod
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute loss given predictions and targets"""
        pass

class BCEWithLogitsLoss(ClassificationLoss):
    def __init__(self, eos_coef: float = 0.1):
        self.eos_coef = eos_coef
        
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor , text_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            preds: [bs, num_queries, num_classes]
            targets: [bs, num_queries, num_classes]
            valid_mask: [bs, num_queries, num_classes]
        """
        # Find matched queries (any target > 0 along class dimension)


        matched_queries = (targets.sum(dim=-1) > 0)  # [bs, num_queries]
        
        total_loss = torch.tensor(0.0, device=preds.device)
        
        # Matched queries loss - only on valid text tokens
        if matched_queries.any():
            matched_preds = preds[matched_queries]      # [num_matched, num_classes]
            matched_targets = targets[matched_queries]  # [num_matched, num_classes]
            matched_valid = valid_mask[matched_queries] # [num_matched, num_classes]

            matched_preds=matched_preds[matched_valid] # N size of num_matched x valid text
            matched_targets=matched_targets[matched_valid]

            #print(matched_preds)
            #print(matched_targets)
            
            if matched_valid.any():
                n=matched_queries.sum()
                matched_loss = F.binary_cross_entropy_with_logits(
                    matched_preds,
                    matched_targets,
                    reduction='sum'
                )
                matched_loss/=n
                total_loss += matched_loss
        
        # Unmatched queries loss - push all valid tokens to zero
        if (~matched_queries).any():
            unmatched_mask = ~valid_mask.bool()  # bs x queries x 256
            text_mask = text_mask.unsqueeze(1).expand(-1, preds.shape[1], -1)  # Expand to match shape    
            # Apply both masks
            total_mask = unmatched_mask & text_mask

            unmatched_queries = (~matched_queries > 0).sum() 

            #print(unmatched_queries)

            unmatched_preds = preds[total_mask]

            unmatched_loss = F.binary_cross_entropy_with_logits(
                unmatched_preds,
                torch.zeros_like(unmatched_preds),
                reduction='sum'
            ) 
            #unmatched_loss/=unmatched_queries
            unmatched_loss*= self.eos_coef

            total_loss += unmatched_loss
    
        return total_loss

class MultilabelFocalLoss(ClassificationLoss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eos_coef: float = 0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.eos_coef = eos_coef
        
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor , text_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            preds: [bs, num_queries, num_classes]
            targets: [bs, num_queries, num_classes]
            valid_mask: [bs, num_queries, num_classes]
        """
        # Find matched queries
        matched_queries = (targets.sum(dim=-1) > 0)  # [bs, num_queries]
        
        total_loss = torch.tensor(0.0, device=preds.device)
        
        # Matched queries loss
        if matched_queries.any():
            matched_preds = preds[matched_queries]      # [num_matched, num_classes]
            matched_targets = targets[matched_queries]  # [num_matched, num_classes]
            matched_valid = valid_mask[matched_queries] # [num_matched, num_classes]
            
            if matched_valid.any():
                # Get valid predictions and targets
                valid_preds = matched_preds[matched_valid]
                valid_targets = matched_targets[matched_valid]
                num_valid = valid_targets.numel()
                
                # Simple focal loss computation
                prob = torch.sigmoid(valid_preds)
                ce_loss = F.binary_cross_entropy_with_logits(valid_preds, valid_targets, reduction="none")
                p_t = prob * valid_targets + (1 - prob) * (1 - valid_targets)
                loss = ce_loss * ((1 - p_t) ** self.gamma)
                
                # Apply alpha weighting
                alpha_t = self.alpha * valid_targets + (1 - self.alpha) * (1 - valid_targets)
                loss = alpha_t * loss
                
                total_loss += loss.sum() / max(num_valid, 1)
        
        '''
        # Unmatched queries loss
        if (~matched_queries).any():
            unmatched_preds = preds[~matched_queries]      # [num_unmatched, num_classes]
            unmatched_valid = valid_mask[~matched_queries] # [num_unmatched, num_classes]
            
            if unmatched_valid.any():
                valid_preds = unmatched_preds[unmatched_valid]
                num_valid = valid_preds.numel()
                
                # Zero targets for unmatched
                valid_targets = torch.zeros_like(valid_preds)
                
                # Simple focal loss computation
                prob = torch.sigmoid(valid_preds)
                ce_loss = F.binary_cross_entropy_with_logits(valid_preds, valid_targets, reduction="none")
                p_t = prob * valid_targets + (1 - prob) * (1 - valid_targets)
                loss = ce_loss * ((1 - p_t) ** self.gamma)
                
                # Apply alpha weighting
                alpha_t = self.alpha * valid_targets + (1 - self.alpha) * (1 - valid_targets)
                loss = alpha_t * loss
                
                total_loss += self.eos_coef * loss.sum() / max(num_valid, 1)
        '''
        return total_loss



class FocalLoss(ClassificationLoss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eos_coef: float = 0.1):
        self.gamma = gamma
        self.alpha = alpha
        
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor, text_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            preds: [bs, num_queries, num_classes]
            targets: [bs, num_queries, num_classes]
            valid_mask: [bs, num_queries, num_classes]
        """
        # Find matched queries
        text_mask = text_mask.repeat(1, preds.size(1)).view(text_mask.shape[0],-1,text_mask.shape[1])
        pred_logits = torch.masked_select(preds, text_mask)
        new_targets = torch.masked_select(targets, text_mask)

        new_targets=new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * new_targets + (1 - self.alpha) * (1 - new_targets)
            loss = alpha_t * loss

        loss=loss.sum()/preds.shape[0]

        return loss
        
