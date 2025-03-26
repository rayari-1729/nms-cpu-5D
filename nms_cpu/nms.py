import torch

def compute_iou(boxes1, boxes2):
    """Compute IoU between rotated 3D bounding boxes (ignores height for simplicity)."""
    x1, y1, x2, y2, ry1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3], boxes1[:, 4]
    x1_b, y1_b, x2_b, y2_b, ry2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3], boxes2[:, 4]

    x_overlap = torch.max(torch.zeros_like(x1), torch.min(x2, x2_b.unsqueeze(0)) - torch.max(x1, x1_b.unsqueeze(0)))
    y_overlap = torch.max(torch.zeros_like(y1), torch.min(y2, y2_b.unsqueeze(0)) - torch.max(y1, y1_b.unsqueeze(0)))
    intersection = x_overlap * y_overlap

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_b - x1_b) * (y2_b - y1_b)

    union = area1.unsqueeze(1) + area2 - intersection
    iou = intersection / union

    return iou

def nms_cpu(boxes, scores, iou_threshold, pre_maxsize=None, post_max_size=None):
    """CPU-based 3D NMS that supports [x1, y1, x2, y2, ry]."""
    order = scores.argsort(descending=True)
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        ious = compute_3d_iou(boxes[0:1], boxes[1:])
        mask = ious[0] < iou_threshold
        order = order[1:][mask]
        boxes = boxes[1:][mask]

    keep = torch.tensor(keep, dtype=torch.long)
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep
