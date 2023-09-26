import PIL
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import math

def load_seg_pt(path):
    with open(path,'r') as f:
        data = json.load(f)
    return np.array(data['shapes'][0]['points'])
    
    
def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros([img_shape[1],img_shape[0]], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    
    assert len(xy) > 2, "Polygon must have points more than 2"
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=np.uint8)
    mask = Image.fromarray(mask)

    return mask

def read_to_mask(mask_path):
    mask = Image.open(mask_path)
    return mask