"""Metrics of performance.
"""

def confusion_matrix(predictions, labels):
    """Compute confusion matrix.
    """
    tp, fp, tn, fn = 0.00001, 0.00001, 0.00001, 0.00001
    for prediction, label in zip(predictions, labels):
        tp = tp + 1 if prediction == 1 and prediction == label else tp
        fp = fp + 1 if prediction == 1 and prediction != label else fp
        fn = fn + 1 if prediction == 0 and prediction != label else fn
        tn = tn + 1 if prediction == 0 and prediction == label else tn
    # Total
    pos = tp + fn
    neg = tn + fp
    # Recall and precision
    recall, precision = tp/(tp + fn), tp/(tp + fp)
    # Accuracy
    acc = (tp + tn)/(tp + fn + tn + fp)
    # F1 score
    f1_score = 2*precision*recall/(precision+recall)
    return {
        "true positive": tp,
        "false positive": fp,
        "false negative": fn,
        "true negative": tn,
        "recall": round(recall, 2),
        "precision": round(precision, 2),
        "accuracy": round(acc, 2),
        "f1 score": round(f1_score, 2)
    }
