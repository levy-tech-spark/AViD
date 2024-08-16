from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
from torchvision.ops import box_iou
import torch.nn.functional as F
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.class_loss import FocalLoss
import os
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda",strict: bool =False):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    if "model" in checkpoint.keys():
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=strict)
    else:
        # The state dict is the checkpoint
        model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed



@torch.no_grad()
class GroundingDINOVisualizer:
    def __init__(self, save_dir, visualize_frequency=50):
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
                images, targets, captions= prepare_data(batch)
                outputs = model(images, captions=captions)

                img = targets[0]["orig_img"]
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
                tokenized = outputs['tokenized']
                phrases = self.extract_phrases(filtered_logits, tokenized, model.tokenizer)

                # Draw predictions
                if len(filtered_boxes):
                    boxes = filtered_boxes * torch.tensor([w, h, w, h])
                    xyxy = box_cxcywh_to_xyxy(boxes).numpy()
                    
                    detections = sv.Detections(xyxy=xyxy)
                    img_bgr = self.pred_annotator.annotate(
                        scene=img_bgr,
                        detections=detections,
                        labels=phrases
                    )

                # Draw ground truth
                if "boxes" in targets[0]:
                    gt_xyxy = box_cxcywh_to_xyxy(targets[0]["boxes"]).cpu().numpy()
                    gt_detections = sv.Detections(xyxy=gt_xyxy)
                    img_bgr = self.gt_annotator.annotate(
                        scene=img_bgr,
                        detections=gt_detections,
                        labels=targets[0].get("str_cls_lst", None)
                    )

                cv2.imwrite(f"{save_dir}/val_pred_{idx}.jpg", img_bgr)
                if idx >= self.visualize_frequency:
                    break


    def visualize_image(self, model, image, caption,image_source,fname,device="cuda"):
        model.eval()
        save_dir = os.path.join(self.save_dir, f'inference')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():          
            caption = preprocess_caption(caption=caption)
            model = model.to(device)
            image = image.to(device)
            outputs = model(image[None], captions=[caption])

            ## Original source image
            img = image_source
            h, w, _ = image_source.shape
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
            tokenized = outputs['tokenized']
            phrases = self.extract_phrases(filtered_logits, tokenized, model.tokenizer)

            # Draw predictions
            if len(filtered_boxes):
                boxes = filtered_boxes * torch.tensor([w, h, w, h])
                xyxy = box_cxcywh_to_xyxy(boxes).numpy()
                
                detections = sv.Detections(xyxy=xyxy)
                img_bgr = self.pred_annotator.annotate(
                    scene=img_bgr,
                    detections=detections,
                    labels=phrases
                )
                cv2.imwrite(f"{save_dir}/{fname}.jpg", img_bgr)
            else:
                print(f"No boxes found for the image above given thresholds!")



def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
