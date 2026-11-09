---
library_name: peft
tags:
- lora
- wifi-sensing
- fall-detection
---

# Kubaho LoRA adapter (r = 4)

LoRA adapter for the Kubaho CSI encoder. It classifies 2 s WiFi-CSI windows (100 frames × 52 subcarriers at
50 Hz) into 8 classes: fall, sit/lie on mat, kneel/pray, squat on low stool, walk, bend/sweep, restless in
bed, and empty room.

| Item | Value |
|---|---|
| Base model | Kubaho CSI encoder (`CSIClassifier` in `src/kubaho/models.py`) |
| Adapted layers | attention `q` and `v` in all 3 transformer blocks |
| Rank / alpha / dropout | 4 / 8 / 0.05 |
| LoRA parameters | 4,608 (1.58% of the encoder) |
| Also trained | classification head (`head`, 128 → 8) |
| Training | class-weighted focal loss, AdamW, cosine schedule, early stopping |
| Data | Kubaho-Lab protocol, synthetic CSI generator (`src/kubaho/synthetic.py`) |
| Threshold and metrics | `models/model_card.json` |

## Usage

```python
import torch
from peft import PeftModel
from kubaho.models import CSIClassifier

base = CSIClassifier(n_classes=8)
base.encoder.load_state_dict(torch.load("models/ruview_encoder_standin.pt"))
model = PeftModel.from_pretrained(base, "models/kubaho_lora_adapter").merge_and_unload()
```

The merged, INT8-quantised model used by the API is `models/kubaho_lora_int8.onnx`.
