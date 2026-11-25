import sys
import torch
import clip
import scipy
from PIL import Image

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('SciPy:', scipy.__version__)
assert torch.cuda.is_available(), 'CUDA GPU unavailable'
print('GPU:', torch.cuda.get_device_name(0))
model, preprocess = clip.load('ViT-B/16', device='cuda')
model.eval()
with torch.inference_mode():
    features = model.encode_image(preprocess(Image.new('RGB', (224, 224))).unsqueeze(0).cuda())
assert features.shape == (1, 512)
assert torch.isfinite(features).all().item()
print('Feature shape:', features.shape)
print('Device:', features.device)
print('PASS: GPU and CLIP environment test')
