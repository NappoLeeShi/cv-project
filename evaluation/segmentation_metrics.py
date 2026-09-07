import torch


def pixel_accuracy(pred, target, ignore_index=255):

    valid = target != ignore_index

    pred = pred[valid]
    target = target[valid]

    correct = (pred == target).sum()

    total = target.numel()

    if total == 0:
        return 0.0

    return (correct.float() / total).item()


def mean_iou(
    pred,
    target,
    num_classes=19,
    ignore_index=255
):

    ious = []

    for cls in range(num_classes):

        pred_cls = pred == cls
        target_cls = target == cls

        if ignore_index is not None:
            valid = target != ignore_index

            pred_cls = pred_cls & valid
            target_cls = target_cls & valid

        intersection = (
            pred_cls & target_cls
        ).sum().float()

        union = (
            pred_cls | target_cls
        ).sum().float()

        if union == 0:
            continue

        iou = intersection / union

        ious.append(iou)

    if len(ious) == 0:
        return 0.0

    return torch.stack(ious).mean().item()