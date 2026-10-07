"""Live demo: upload a fabric image -> prediction, confidence, and Grad-CAM defect heatmap.

Run:  streamlit run app.py
Expects outputs/model.pt produced by train.py
"""
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from matplotlib import cm

import os
MODEL_PATH = next((p for p in ("outputs/model.pt", "model.pt") if os.path.exists(p)), "outputs/model.pt")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class GradCAM:
    def __init__(self, model, layer):
        self.acts, self.grads = None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, model, x):
        model.zero_grad()
        out = model(x)
        cls = int(out.argmax(1))
        out[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), out.softmax(1)[0].detach().cpu().numpy()


@st.cache_resource
def load():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], ckpt["img_size"], GradCAM(model, model.layer4)


st.set_page_config(page_title="Fabric Defect Detection", layout="wide")
st.title("Fabric Defect Detection")
st.caption("ResNet18 transfer learning with Grad-CAM explanations")

model, classes, size, cam_tool = load()
files = st.file_uploader("Upload fabric image(s)", type=["jpg", "jpeg", "png", "bmp"],
                         accept_multiple_files=True)

tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])

for f in files or []:
    img = Image.open(f).convert("RGB")
    x = tf(img).unsqueeze(0)
    cam, probs = cam_tool(model, x)
    pred = classes[int(probs.argmax())]
    base = img.resize((size, size))
    heat = Image.fromarray((cm.jet(cam)[..., :3] * 255).astype(np.uint8))
    overlay = Image.blend(base, heat, 0.4)

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.image(base, caption="Input", use_container_width=True)
    c2.image(overlay, caption="Grad-CAM (where the model looked)", use_container_width=True)
    with c3:
        st.subheader(f"Prediction: {pred}")
        st.metric("Confidence", f"{probs.max() * 100:.1f}%")
        st.bar_chart({c: float(p) for c, p in zip(classes, probs)})
    st.divider()
