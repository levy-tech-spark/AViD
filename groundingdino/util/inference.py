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

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
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


def train_image(model,
        image_source,
        image: torch.Tensor,
        caption_objects: list,
        box_target: list,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False):

    # Currently supporting 
    tokenizer = model.tokenizer

    caption= list(set(caption_objects))
    # Added . for easy matching later on
    objects_token={item:tokenizer(item+".")['input_ids'] for item in caption}

    if len(caption)>1:
            caption=caption[0]+"."+(".".join(caption[1:]))
    else:
            caption=caption[0]

    caption = preprocess_caption(caption=caption)
    tokenized = tokenizer(caption)

    # Get the positions of objects_token in the tokenized so we can maximize logits of these positions
    object_positions_dict = {}
    print(tokenized['input_ids'])
    print(objects_token)

    for obj_name, obj_token in objects_token.items():
        start_pos = 0  # Initial position can start from 0
        while start_pos <= len(tokenized['input_ids']) - len(obj_token) + 1:  # Adjust the end condition to prevent overrunning
            # Check if the current position's token matches the inner section of the object token
            #print(f"Comparing {tokenized['input_ids'][start_pos:start_pos+len(obj_token)-2]} and {obj_token[1:-1]}")
            if tokenized['input_ids'][start_pos:start_pos+len(obj_token)-2] == obj_token[1:-1]:
                # Store the range in the dictionary (not inclusive of the last element)
                object_positions_dict[obj_name] = [start_pos, start_pos+len(obj_token)-2]
                break  # Break out of the loop once the match is found
            start_pos += 1

    
    print(object_positions_dict)
    

    # Get the positions of objects_token in the tokanized so we can maximize logits of these positions

    #print(f"Preprocessed caption is {caption}")
    #print(f"Tokanizer decoded Caption is {tokenizer.decode(tokenized['input_ids'])}")
    #return
    #print(f"Tokanized Caption is {tokenized}")
    #print(f"Objects tokerns are {objects_token}")
    
    model = model.to(device)
    image = image.to(device)
    
    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"][0]  # prediction_logits.shape = (nq, 256)
    boxes = outputs["pred_boxes"][0]  # prediction_boxes.shape = (nq, 4)
    # Calculate losses for bounding boxes
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h]).cuda()
    box_predicted = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    box_target = torch.tensor(box_target).cuda()
    ious=box_iou(box_target,box_predicted)
    maxvals,maxidx=torch.max(ious,dim=1)
    selected_preds = box_predicted.gather(0, maxidx.unsqueeze(-1).repeat(1, box_predicted.size(1)))
    regression_loss = F.smooth_l1_loss(box_target, selected_preds)
    # IoU-based Loss
    iou_loss = 1.0 - maxvals.mean()
    # Combine the two losses
    lambda_factor = 1.0  
    reg_loss = iou_loss + lambda_factor * regression_loss
    print(f"Reg loss is {reg_loss}")
    # Calculate losses for logistic regression
    selected_logits = logits.gather(0, maxidx.unsqueeze(-1).repeat(1, logits.size(1)))
    print(f"Selected logits are {selected_logits.shape}")
    targets_logits=[]
    for idx,logit in enumerate(selected_logits):
        tgt_lgt=torch.zeros_like(logit)
        rng=object_positions_dict[caption_objects[idx]]
        tgt_lgt[rng[0]:rng[1]]=1.0
        tgt_lgt=tgt_lgt.unsqueeze(dim=0)
        targets_logits.append(tgt_lgt)
        #print(f"Target logit for label {caption_objects[idx]} {tgt_lgt}")

    targets_logits=result = torch.cat(targets_logits, dim=0)
    print(f"Target Logits shape is {targets_logits.shape}")





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
