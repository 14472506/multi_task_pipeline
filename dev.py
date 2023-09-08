from torch import tensor
import torch
from torchmetrics.detection import MeanAveragePrecision
mask_pred = [
  [0, 0, 0, 0, 0],
  [0, 0, 1, 1, 0],
  [0, 0, 1, 1, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
]
mask_tgt = [
  [0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 1, 1, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0],
]
preds = [
  dict(
    masks=tensor([mask_pred], dtype=torch.bool),
    scores=tensor([0.536]),
    labels=tensor([0]),
  )
]
target = [
  dict(
    masks=tensor([mask_tgt], dtype=torch.bool),
    labels=tensor([0]),
  )
]

print(preds)
metric = MeanAveragePrecision(iou_type="segm")
metric.update(preds, target)