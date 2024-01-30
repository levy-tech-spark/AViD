from groundingdino.util.inference import load_model, load_image, predict,train, annotate
import cv2
import os

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

images_files=sorted(os.listdir("multi-model-data/images"))
ann_files=sorted(os.listdir("multi-model-data/"))

for 
IMAGE_PATH = "test_pepper.jpg"
TEXT_PROMPT = "fruit.stem"
BOX_TRESHOLD = 0.1
TEXT_TRESHOLD = 0.2

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = train(
    model=model,
    image_source=image_source,
    image=image,
    caption=TEXT_PROMPT,
    box_target=box_target,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)
