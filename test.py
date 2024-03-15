from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch

import torchvision.ops as ops
from torchvision.ops import box_convert

def apply_nms_per_phrase(image_source,boxes, logits, phrases, threshold=0.3):
    # Initialize lists to store NMS results.
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []
    print(f"The unique phrases are {set(phrases)}")
    # Loop over each unique phrase.
    for unique_phrase in set(phrases):
        #if unique_phrase=="peduncle":
        #    continue
        # Get indices of boxes associated with this phrase.
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        
        # Filter boxes and logits using these indices.
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]
        
        # Apply NMS
        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        
        # Collect NMS results
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))
        
    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list


model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
model.load_state_dict(torch.load("weights/model_weights2000.pth"))

#IMAGE_PATH = "multimodal-data/images/CARABAGGIO_220329_203615_598979.jpg"
IMAGE_PATH=  "test_pepper.jpg"
TEXT_PROMPT = "peduncle.fruit."
BOX_TRESHOLD = 0.8
TEXT_TRESHOLD = 0.40

image_source, image = load_image(IMAGE_PATH)
#cv2.imwrite("inp.jpg",image_source)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)
print(f"Original boxes size {boxes.shape}")
#print(f"Original phrases {phrases}")
boxes, logits, phrases = apply_nms_per_phrase(image_source,boxes, logits,phrases)
print(f"NMS boxes size {boxes.shape}")
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("result.jpg", annotated_frame)
