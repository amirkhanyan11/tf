
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tempfile
from urllib.request import urlopen
from io import BytesIO

# Load a pre-trained object detection model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")  # SSD MobileNet V2 1

def load_img(url):
    _, fname = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    img = Image.open(BytesIO(response.read())).convert("RGB")
    img_arr = np.array(img) / 255.0
    return img, img_arr

def draw_boxes(image, boxes, class_names, scores, min_score=0.3):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size
    for box, name, score in zip(boxes, class_names, scores):
        if score < min_score: continue
        ymin, xmin, ymax, xmax = box
        draw.rectangle([xmin*w, ymin*h, xmax*w, ymax*h], outline="red", width=2)
        draw.text((xmin*w, ymin*h), f"{name.decode('ascii')}: {score:.2f}", fill="white", font=font)
    return image

# Example usage:
url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Hebehnar.jpg"
pil_img, img_arr = load_img(url)

# Perform detection
	img_tensor = tf.expand_dims(img_arr, axis=0)
result = model(img_tensor)

boxes = result["detection_boxes"][0].numpy()
class_ids = result["detection_class_entities"][0].numpy()
scores = result["detection_scores"][0].numpy()

output = draw_boxes(pil_img.copy(), boxes, class_ids, scores)
plt.imshow(output)
plt.axis("off")