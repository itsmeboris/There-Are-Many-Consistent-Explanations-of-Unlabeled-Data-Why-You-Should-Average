import os
import time
from typing import List, Any

import torch
from torch import Tensor, nn
from torchmetrics import Metric, PrecisionRecallCurve, AUC, StatScores, F1
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class SkAucRoc(Metric):
    """
    Calculate area under curve
    """
    def __init__(self, classes):
        super(SkAucRoc, self).__init__()
        self.labels = list()
        self.preds = list()
        self.classes = [i for i, _ in enumerate(classes)]

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.labels.append(target)

    def compute(self) -> Any:
        y_score = torch.nn.Softmax(dim=1)(torch.cat(self.preds)).numpy()
        y_true = torch.cat(self.labels).int().numpy()
        try:
            return roc_auc_score(y_true, y_score, average='weighted', multi_class='ovo', labels=self.classes)
        except:
            return 0


class ModifiedF1(Metric):
    """
    Wrapper of F1 Metric to recive probabilites instead of labels
    """
    def __init__(self, classes) -> None:
        super(ModifiedF1, self).__init__()
        self.f1 = F1(num_classes=len(classes))

    def update(self, preds: Tensor, target: Tensor, *_: Any) -> None:
        class_preds = torch.argmax(preds, dim=1)
        self.f1.update(class_preds, target)

    def compute(self) -> Any:
        return self.f1.compute()


class TprFpr(Metric):
    """
    Calculates True Positive Rate (TPR) and False Positive Rate (FPR)
    """
    def __init__(self, classes: List[str]):
        super(TprFpr, self).__init__()
        self.classes = classes
        self.stats = StatScores(num_classes=len(classes))

    def update(self, preds: Tensor, target: Tensor, *_: Any) -> None:
        class_preds = torch.argmax(preds, dim=1)
        self.stats.update(class_preds, target)

    def compute(self) -> Any:
        tp, fp, tn, fn, _ = self.stats.compute().numpy()
        tpr = tp / (tp+fn)
        fpr = fp / (fp+tn)
        return {"TPR": tpr, "FPR": fpr}


class AreaUnderPrecisionCurve(Metric):
    """
    Calculates the area under the precision-recall curve for each class
    """
    def __init__(self, classes: List[str], output_dir: str, fold_num: int, model_name: str, dataset_name: str):
        super(AreaUnderPrecisionCurve, self).__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.fold_num = fold_num
        self.output_dir = output_dir
        self.classes = classes
        self.pr_curve = PrecisionRecallCurve(num_classes=len(classes))

    def update(self, preds: Tensor, target: Tensor, *_: Any) -> None:
        self.pr_curve.update(preds, target)

    def compute(self) -> Any:
        ans = dict()
        for idx, cls in enumerate(self.classes):
            auc = AUC()
            precision = self.pr_curve.compute()[0][idx]
            recall = self.pr_curve.compute()[1][idx]
            auc.update(recall, precision)
            ans[cls] = auc.compute().numpy().tolist()
            plt.plot(recall, precision)
        ans['avg'] = sum(ans.values())/len(ans)
        plt.legend(self.classes)
        plt.title(f"Precision Recall for {self.model_name}\n"
                  f"With dataset: {self.dataset_name} fold: {self.fold_num}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(self.output_dir,
                                 f"precision_recall_dataset_{self.dataset_name}_fold_{self.fold_num}.jpg"))
        return ans


class Time1000Samples(Metric):
    """
    Calculates time to infer 1000 samples
    """
    def __init__(self, model: nn.Module):
        super(Time1000Samples, self).__init__()
        self.x = list()
        self.length = 0
        self.y_true = list()
        self.model = model

    def update(self, x: Tensor, *_: Any) -> None:
        if self.length == 1000:
            return
        added_len = len(x)
        if added_len + self.length < 1000:
            self.x.append(x)
            self.length += added_len
        else:
            to_keep = 1000 - self.length
            self.x.append(x[:to_keep])
            self.length = 1000

    def compute(self) -> Any:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        start_time = time.time()
        trained = 0
        for x in self.x:
            if trained >= 1000:
                break
            to_train = 1000 - trained
            inp = x[:to_train].to(device)
            self.model(inp)
            trained += to_train
        end_time = time.time()
        return end_time - start_time
