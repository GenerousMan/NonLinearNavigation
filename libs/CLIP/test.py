import torch
import clip
from PIL import Image
import time
import numpy as np

start = time.time()
device = 'cpu'
model, preprocess = clip.load("./ViT-B-32.pt", device=device)
image_ori = Image.open("CLIP.png")
print(type(image_ori))
image = preprocess(image_ori).to(device).numpy()
image = torch.from_numpy(np.array([image,image]))
print(type(image),image.shape)

print(image.shape)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
print(image.shape)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
end = time.time()

print(end-start)