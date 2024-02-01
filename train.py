from groundingdino.util.inference import load_model, load_image, predict,train, annotate
import cv2
import os
import json
import csv
from collections import defaultdict

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

images_files=sorted(os.listdir("multimodal-data/images"))
ann_file="multimodal-data/annotation/annotation.csv"

ann_Dict= defaultdict(lambda: defaultdict(list))

with open(ann_file) as file_obj:
    ann_reader= csv.DictReader(file_obj)  
    # Iterate over each row in the csv file
    # using reader object
    for row in ann_reader:
        #print(row)
        img_n=os.path.join("multimodal-data/images",row['image_name'])
        x1=int(row['bbox_x'])
        y1=int(row['bbox_y'])
        x2=x1+int(row['bbox_width'])
        y2=y1+int(row['bbox_height'])
        label=row['label_name']
        ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
        ann_Dict[img_n]['caption'].append(label)

print(ann_Dict)
exit()


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
