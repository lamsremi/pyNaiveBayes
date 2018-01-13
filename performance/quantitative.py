"""
Qualitative performance.
"""
import pandas as pd


def main(predicted_data_df, label_column, prediction_column):
    """
    Main function for quantitative performance.
    """
    # Confusion matrix
    result = build_confusion_matrix(predicted_data_df,
                                              label_column,
                                              prediction_column)
    return result


def build_confusion_matrix(predicted_data_df, label_column, prediction_column):
    """Compute confusion matrix."""
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for index, serie in predicted_data_df.iterrows():
        prediction = serie[prediction_column]
        label = serie[label_column]
        if prediction == 1:
            if label == 1:
                true_positive += 1
            elif label == 0:
                false_positive += 1
        elif prediction == 0:
            if label == 1:
                false_negative += 1
            elif label == 0:
                true_negative += 1
    matrix = pd.DataFrame(
        [
            [true_positive, false_positive, int(true_positive+false_positive)],
            [false_negative, true_negative, int(false_negative+true_negative)],
            [int(true_positive+false_negative), int(false_positive+true_negative), 0]
        ],
        columns=["real positive", "real negative", "total predicted"],
        index=["predict positive", "predict negative", "total real"]
    )
    precision = true_positive/(true_positive+false_positive+1)
    recall = true_positive/(true_positive+false_negative+1)
    f_score = precision*recall/(precision+recall+1)
    return {
        "confusion_matrix": matrix,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f_score": round(f_score, 2)
    }
