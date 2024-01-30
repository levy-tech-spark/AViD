from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

IMAGE_PATH = "test_pepper.jpg"
TEXT_PROMPT = "fruit.stem"
BOX_TRESHOLD = 0.1
TEXT_TRESHOLD = 0.2

image_source, image = load_image(IMAGE_PATH)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)
