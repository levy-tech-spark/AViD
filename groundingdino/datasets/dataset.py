import os
import csv
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from groundingdino.util.train import load_image
from groundingdino.util.vl_utils import build_captions_and_token_span

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
    
