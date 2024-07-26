# Open CLIP demo
## Objective
- Building a CLIP application on gradio/spaces using open-source models.

## Steps

### Experiment

1. Used this [github page](https://github.com/mlfoundations/open_clip) and created the [notebook](https://github.com/sayanbanerjee32/open_clip_demo/blob/main/openClip_Gradio.ipynb) for experiment.

### Gradio App in HuggingFace Spaces
1. Created app.py that can read the model artifacts from HuggingFace Model Hub and launch the app
2. Pushed the app.py and requirements.txt to HuggingFace spaces using huggingface API from this [notebook](https://github.com/sayanbanerjee32/open_clip_demo/blob/main/openClip_Gradio.ipynb)

## The HuggingFace Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/sayanbanerjee32/open_clip_demo)

![image](https://github.com/user-attachments/assets/9e6ba66b-bb1b-41d5-aa40-da8af8015470)

- The app provides a set of images to search from. These images are sourced from scikit-learn image
- It takes free text as input for image search
- Displays the most suitable image from the given image as output
