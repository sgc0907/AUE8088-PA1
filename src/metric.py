from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("confmat", default=torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == target.dim() + 1:
            preds = torch.argmax(preds, dim=1)
        if preds.shape != target.shape:
            raise ValueError(f"Shape error! preds={preds.shape}, target={target.shape}")
        
        preds = preds.view(-1)
        target = target.view(-1)
        idx = target * self.num_classes + preds
        cm = torch.bincount(idx, minlength=self.num_classes * self.num_classes).view(self.num_classes, self.num_classes)
        self.confmat += cm

    def compute(self) -> torch.Tensor:
        conf = self.confmat.float()

        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        return f1.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        if preds.dim() == target.dim() + 1:
            preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Shape error! preds={preds.shape}, target={target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
