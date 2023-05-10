import io
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import ConfusionMatrix, F1Score, Precision, Recall, Accuracy
from pytorch_lightning import LightningDataModule, LightningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.savefig("cm.png", format="png")

    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calc_metric(
    model: LightningModule,
    datamodule: LightningDataModule,
    log_cm: bool = False,
    valset=False,
):
    model = model.to(device)
    model.eval()

    preds = []
    targets = []
    dataloader = datamodule.val_dataloader() if valset else datamodule.test_dataloader()

    acc = Accuracy(task="multiclass", num_classes=6)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            p = model(x)
            logits = p if i == 0 else torch.concat([logits, p])

            _, p = torch.max(p, 1)

            preds.extend(p.tolist())
            targets.extend(y.tolist())

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    confmat = ConfusionMatrix(task="multiclass", num_classes=6)
    print(":: Confusion Matrix ->")
    cm = confmat(preds, targets)
    print(cm)

    if log_cm:
        cm = cm.cpu().detach().numpy()
        fig = plot_confusion_matrix(cm, datamodule.classes)
        fig = torch.tensor(np.transpose(fig, (2, 0, 1)))
        model.logger.experiment.add_image("comfution_matrix", fig, 0)

    acc_score = acc(preds, targets)
    loss = criterion(logits, targets)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    metric_dict = {
        "accuracy": {
            "value": acc_score.item(),
            "per_class_accuracy": {
                c: a for c, a in zip(datamodule.classes, per_class_accuracy.tolist())
            },
            "standard_deviation": per_class_accuracy.std().item(),
        },
        "loss": loss.item(),
    }

    f1 = F1Score(task="multiclass", num_classes=6)
    metric_dict["f1_score"] = -f1(preds, targets).item()

    precision = Precision(task="multiclass", average="micro", num_classes=6)
    metric_dict["precission_micro"] = precision(preds, targets).item()

    precision = Precision(task="multiclass", average="macro", num_classes=6)
    metric_dict["precission_macro"] = precision(preds, targets).item()

    precision = Precision(task="multiclass", average="weighted", num_classes=6)
    metric_dict["precission_weighted"] = precision(preds, targets).item()

    recall = Recall(task="multiclass", average="micro", num_classes=6)
    metric_dict["recall"] = recall(preds, targets).item()

    print(":: Eval Metrices-> ", metric_dict)
    return metric_dict
