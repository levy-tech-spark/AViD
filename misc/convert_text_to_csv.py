import pandas as pd

# Load the file
file_path = '/home/levy/dev/Adaptive-Vision-Language-Detector/multimodal-data/fashion_dataset_subset/train_annotations.csv'  # Replace with the actual file path
output_csv = '/home/levy/dev/Adaptive-Vision-Language-Detector/multimodal-data/fashion_dataset_subset/train_annotations2.csv'

# Define column names for the structured data
columns = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'width', 'height']

# Read the data file and convert to DataFrame
data = []
with open(file_path, 'r') as file:
    for line in file:
        # Split each line based on spaces
        parts = line.strip().split()
        assert len(parts)==8, "Each line should have 8 values"
        print(f"Cheking the second argument is width height not x2,y2")
        x1=int(parts[1])
        y1=int(parts[2])
        
        x2=int(parts[3])
        y2=int(parts[4])
        
        w=x2-x1
        h=y2-y1
        
        parts[3]=w
        parts[4]=h
        
        im_w=int(parts[-2])
        im_h=int(parts[-1])
        
        
        assert x1+w<im_w, "The second argument is x2 not w!"
        assert y1+h<im_h, "The second arhument is y2 now h"
        #print(f"So far good!!")
        
        if len(parts) == 8:
            data.append(parts)

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

# Save as CSV
df.to_csv(output_csv, index=False)
print(f"Annotations saved to {output_csv}")
