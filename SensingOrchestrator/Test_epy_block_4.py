#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import torch
import torch.nn as nn
import timm
import functools
import json
from PIL import Image
from torchvision import transforms
from gnuradio import gr
import torch.serialization
import numpy._core.multiarray

# ---- Allow NumPy arrays inside trusted checkpoint (safe only for trusted source) ----
torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])

# ---- Constants ----
IMG_SIZE = 224
MODEL_PATH = "/home/happy/Downloads/efficientnet_lite1_multitask.pth"

# ---- Transform pipeline ----
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---- Multitask Model Architecture ----
class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=6, n_regress=2):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_lite1',
            pretrained=False,
            num_classes=num_classes,
            norm_layer=functools.partial(nn.GroupNorm, num_groups=8)
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
        self.regressor = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, n_regress))
        self.freq_predictor = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 1))
        self.bw_predictor   = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        f = self.backbone(x)
        cls = self.classifier(f)
        reg = self.regressor(f)
        fp  = self.freq_predictor(f)
        bp  = self.bw_predictor(f)
        return cls, reg, fp, bp

# ---- Load Model BEFORE block class so GRC parser doesn't fail ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MultiTaskModel(num_classes=6, n_regress=2)

# Force full load because file is trusted
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

# Strip module. if present
clean_state = {}
for k, v in ckpt.items():
    if isinstance(k, str) and k.startswith("module."):
        k = k.replace("module.", "")
    clean_state[k] = v

model.load_state_dict(clean_state, strict=False)
model.to(device)
model.eval()

LABELS = ['BLE', 'ZIGBEE', 'CW', 'FHSS', 'MICROWAVE', 'NONE']

def format_freq_hz_to_ghz(freq_hz: float) -> str:
    return f"{freq_hz/1e9:.3f}GHz"

def format_bandwidth_hz_to_mhz(bw_hz: float) -> str:
    return f"{bw_hz/1e6:.3f}MHz"

def overall_confidence(class_probs: dict) -> float:
    return float(max(class_probs.values())) if class_probs else 0.0

# ---- Main GNU Radio Block ----
class Interference_Predictor(gr.sync_block):
    def __init__(self):
        super().__init__(
            name="Interference_Predictor",
            in_sig=[(numpy.uint8, IMG_SIZE * IMG_SIZE)],
            out_sig=[
                (numpy.int16, 6),      # ← index vector of all interferers
                numpy.complex64,       # freq + j*bw
                numpy.float32,         # confidence
            ]
        )

    @torch.no_grad()
    def work(self, input_items, output_items):
        vec = input_items[0][0]
        img = vec.reshape((IMG_SIZE, IMG_SIZE)).astype(numpy.uint8)

        pil_img = Image.fromarray(img, mode='L').convert("RGB")
        x = INFER_TRANSFORM(pil_img).unsqueeze(0).to(device)

        cls_logits, _, freq_p, bw_p = model(x)

        # PROBABILITIES
        probs = torch.sigmoid(cls_logits).cpu().numpy().reshape(-1)
        preds = (probs >= 0.5).astype(int)      # multi-hot vector

        # ---- Build interference vector output ----
        # preds shape = [6], example: [1,0,0,1,0,0]
        interference_vec = preds.astype(numpy.int16)

        # ---- Interference confidence ----
        confidence = float(max(probs))*1.4

        # ---- Regression outputs ----
        freq_raw = float(freq_p.cpu().numpy()[0][0])
        bw_raw = float(bw_p.cpu().numpy()[0][0])
    

        freq_norm = 1.0 / (1.0 + numpy.exp(-freq_raw))
        bw_norm   = 1.0 / (1.0 + numpy.exp(-bw_raw))

        cf_hz = freq_norm * 100e6 + 2.4e9   # maps to 2.400 - 2.500 GHz
        bw_hz = bw_norm * 20e6              # maps to 0 - 20 MHz

        # ---- Write outputs ----
        output_items[0][0][:] = interference_vec               # vector
        output_items[1][0]     = numpy.complex64(cf_hz + 1j*bw_hz)
        output_items[2][0]     = numpy.float32(confidence)


        detected = [LABELS[i] for i, v in enumerate(interference_vec) if v == 1]

        sample_output = {
            "interferences": detected,
            "center_freq": float(cf_hz),
            "bandwidth": float(bw_hz),
            "confidence": round(confidence, 3)
        }

        #print(sample_output, flush=True)

        return 1
