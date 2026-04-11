import cv2
import torch
import numpy as np

REGIONS = {
    "eyes":  (0.15, 0.35, 0.25, 0.75),
    "nose":  (0.35, 0.60, 0.35, 0.65),
    "mouth": (0.55, 0.75, 0.30, 0.70),
    "jaw":   (0.65, 0.95, 0.20, 0.80),
}

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor):
        self.model.eval()
        output = self.model(img_tensor)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).squeeze()
        heatmap = torch.relu(heatmap).detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
        return heatmap

    def region_scores(self, heatmap):
        h, w = heatmap.shape
        return {
            name: float(heatmap[int(y1*h):int(y2*h), int(x1*w):int(x2*w)].mean())
            for name, (y1, y2, x1, x2) in REGIONS.items()
        }

    def overlay(self, img_tensor, heatmap):
        """Returns a BGR image with heatmap overlaid."""
        img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img * 0.229 + 0.485)  # rough denormalize
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        hm  = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        hm  = np.uint8(255 * hm)
        hm  = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        return cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, hm, 0.4, 0)