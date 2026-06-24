import lpips
import torch

_lpips_model = None
_device = None


def _get_lpips_model(net='alex', device=None):
    global _lpips_model, _device
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net)
    if device is not None and _device != device:
        _lpips_model = _lpips_model.to(device)
    _device = device
    return _lpips_model


def calculate_lpips(img1,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False,
                    net='alex',
                    **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Lower LPIPS means higher perceptual similarity.

    Args:
        img1 (tensor): Images with range [0, 1] or [0, 255], shape CHW or NCHW.
        img2 (tensor): Images with range [0, 1] or [0, 255], shape CHW or NCHW.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Ignored; tensor inputs are expected in CHW/NCHW.
        test_y_channel (bool): Deprecated. Ignored.
        net (str): Backbone network for LPIPS ('alex', 'vgg', 'squeeze').
            Default: 'alex'.

    Returns:
        float: lpips result.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    assert img1.device == img2.device, 'Input images must be on the same device.'

    assert torch.is_tensor(img1), 'Now only support tensor inputs.'

    device = img1.device
    img1 = img1.clone().to(device=device, dtype=torch.float32)
    img2 = img2.clone().to(device=device, dtype=torch.float32)

    # Add batch dimension if needed
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # Crop border
    if crop_border != 0:
        img1 = img1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    # Normalize to [-1, 1] as required by lpips
    max_val = 1.0 if torch.max(img2) <= 1.0 else 255.0
    img1 = img1 / max_val * 2.0 - 1.0
    img2 = img2 / max_val * 2.0 - 1.0

    loss_fn = _get_lpips_model(net=net, device=device)
    loss_fn.eval()

    with torch.no_grad():
        score = loss_fn(img1, img2)

    return score.mean().item()
