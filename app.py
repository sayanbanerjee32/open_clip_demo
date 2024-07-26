import os
import numpy as np
import torch

import skimage
from PIL import Image

import open_clip

import gradio as gr

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

target_labels = ["page","chelsea","astronaut","rocket",
                 "motorcycle_right","camera","horse","coffee",
                 'logo']

original_images = []
images = []
file_names = []

for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in target_labels:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")

    original_images.append(image)
    images.append(preprocess(image))
    file_names.append(filename)

image_input = torch.tensor(np.stack(images))
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)


def identify_image(input_description):
    if input_description is None: return None
    text_tokens = tokenizer([input_description])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T)
    top_probs, _ = text_probs.cpu().topk(1, dim=-1)
    return original_images[top_probs.argmax().item()]

with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Image Search </h1>")
    gr.HTML("<h4 align = 'center'> Idenitify the most suitable image for description provided.</h4>")
    
    gr.Gallery(value = original_images,
        label="Images to search from", show_label=True, elem_id="gallery"
        , columns=[3], rows=[3], object_fit="contain", height="auto")
    
    content = gr.Textbox(label = "Enter search text here")
    inputs = [
            content,
            ]
    gr.Examples(["Page of text about segmentation",
                "Facial photo of a tabby cat",
                "Portrait of an astronaut with the American flag",
                "Rocket standing on a launchpad",
                "Red motorcycle standing in a garage",
                "Person looking at a camera on a tripod",
                "Black-and-white silhouette of a horse",
                "Cup of coffee on a saucer",
                 "A snake in the background"], 
                inputs = inputs)
    
    generate_btn = gr.Button(value = 'Identify')
    outputs  = [gr.Image(label = "Is this the image you are referring to?",
                         height = 512, width = 512)]
    generate_btn.click(fn = identify_image, inputs= inputs, outputs = outputs)

# for collab
# demo.launch(debug=True) 

if __name__ == '__main__':
    demo.launch() 
