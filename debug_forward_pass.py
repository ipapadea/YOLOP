import torch
import torch.nn as nn
from functools import wraps
import sys
import os

# === Ρύθμισε το path αν χρειάζεται ===
sys.path.append(os.getcwd())

from lib.models import get_net
from lib.config import cfg  # Αν χρησιμοποιείς cfg αρχείο, διαφορετικά αφαίρεσέ το


# === DEBUG WRAPPER για Conv2d και Upsample ===
def debug_shapes(fn):
    @wraps(fn)
    def wrapper(self, x):
        class_name = self.__class__.__name__
        in_shape = x.shape if isinstance(x, torch.Tensor) else [t.shape for t in x]
        print(f"[{class_name}] input shape: {in_shape}")
        out = fn(self, x)
        out_shape = out.shape if isinstance(out, torch.Tensor) else [t.shape for t in out]
        print(f"[{class_name}] output shape: {out_shape}")
        return out
    return wrapper

# Εφαρμογή σε PyTorch built-ins
nn.Conv2d.forward = debug_shapes(nn.Conv2d.forward)
nn.Upsample.forward = debug_shapes(nn.Upsample.forward)

# Αν έχεις custom layers όπως ELANBlock_Head, RepConv κ.λπ.:
# from lib.models.common_yolopv3 import ELANBlock_Head
# ELANBlock_Head.forward = debug_shapes(ELANBlock_Head.forward)


# === MAIN DEBUGGING FORWARD ===
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Χτίσε το μοντέλο ===
    try:
        model = get_net(cfg).to(device)  # Αν δεν έχεις cfg, πέρασε απλά get_net()
    except Exception as e:
        print(f"[ERROR] Failed to build model: {e}")
        sys.exit(1)

    # === Dummy input (π.χ. RGB εικόνα 512x512) ===
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # === Forward pass με debugging ===
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("\n✅ Model forward pass completed.")
    except Exception as e:
        print("\n❌ Error during forward pass:")
        print(e)
