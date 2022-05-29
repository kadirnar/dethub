import torch
from torch import nn

# POSTPROCESSING METHODS
# inspired by https://github.com/facebookresearch/detr/blob/master/models/detr.py#L258
def post_process(outputs, target_sizes):
    """
    Converts the output of [`DetrForObjectDetection`] into the format expected by the COCO api. Only supports
    PyTorch.
    Args:
        outputs ([`DetrObjectDetectionOutput`]):
            Raw outputs of the model.
        target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
            Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
            image size (before any data augmentation). For visualization, this should be the image size after data
            augment, but before padding.
    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
        in the batch as predicted by the model.
    """
    out_logits, out_bbox = outputs.logits, outputs.pred_boxes

    if len(out_logits) != len(target_sizes):
        raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
    if target_sizes.shape[1] != 2:
        raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

    prob = nn.functional.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

    return results


def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
