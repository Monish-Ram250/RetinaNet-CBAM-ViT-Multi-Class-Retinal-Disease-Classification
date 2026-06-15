import os
import json
import cv2
import numpy as np
import torch
from torchvision import transforms
import gradio as gr
from model_def import DualBackboneCBAMViT, CLAHETransform

with open("config.json") as f:
    config = json.load(f)

with open("class_names.json") as f:
    class_names = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DualBackboneCBAMViT(
    num_classes=config["num_classes"],
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"]
).to(device)

model.load_state_dict(
    torch.load("best_dual_backbone_cbam_vit.pth", map_location=device)
)
model.eval()

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config["img_size"], config["img_size"])),
    CLAHETransform(),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config["imagenet_mean"], std=config["imagenet_std"])
])


def predict(image):
    if image is None:
        return {}

    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Retinal Image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="FD3611 Retinal Disease Classifier",
    description="Dual-backbone (EfficientNet-B0 + ResNet50) + CBAM + Transformer model for retinal disease classification (5 classes).",
)

if __name__ == "__main__":
    demo.launch()