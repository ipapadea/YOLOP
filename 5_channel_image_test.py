import torch
from lib.config import cfg
from lib.models import get_net

# Βήμα 1: Φτιάξε ένα ψεύτικο multispectral input
# (1 εικόνα, 5 κανάλια, 512x512)
fake_input = torch.randn(1, 5, 600, 600)

# Βήμα 2: Φτιάξε dummy model_cfg (κανάλια encoder output κ.λπ.)
model_cfg = {
    'nc': 2,              # for instance segmentation
    'nm': 32,             # number of mask coefficients
    'npr': 256,           # number of prototype masks
    'chanels': [32, 64, 128, 256, 512],  # adjust based on your encoder out
}

# Βήμα 3: Φτιάξε το μοντέλο σου
model = get_net(cfg)

# Βήμα 4: Κάνε eval και βάλε με no_grad για να ελέγξεις
model.eval()
with torch.no_grad():
    output = model(fake_input)


det_pred, proto = output[0]
print("mask_coeffs:", det_pred.shape)
print("proto shape:", proto.shape)

# Βήμα 5: Δες τα outputs
print(f"🔢 Total outputs: {len(output)}")

# detection (pred, proto)
det_out = output[0]
print(f"📦 Detection output: {type(det_out)}")
if isinstance(det_out, tuple):
    print(f"  - Num predictions levels: {len(det_out[0])}")
    print(f"  - Proto shape: {det_out[1].shape}")

# segmentation (optional)
for i, seg in enumerate(output[1:], start=1):
    print(f"🧩 Segmentation output {i}: {seg.shape}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Χρήση
total, trainable = count_parameters(model)
print(f"📦 Total parameters: {total:,}")
print(f"🛠️ Trainable parameters: {trainable:,}")
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (5, 600, 600), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f"🧠 MACs: {macs}")
    print(f"📦 Params: {params}")