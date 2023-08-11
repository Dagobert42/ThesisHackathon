import numpy as np
import evaluate

metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def get_predictions(trainer, samples, label_list):
    all_logits, labels, _ = trainer.predict(samples)
    predictions = np.argmax(all_logits, axis=2)

    # Remove ignored index (special tokens)
    logits = [
        [token_lgts for (token_lgts, l) in zip(instance_lgts, label) if l != -100]
        for instance_lgts, label in zip(all_logits, labels)
    ]
    id_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    id_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Replace ids with actual classes
    real_predictions = [
        [label_list[p] for p in prediction]
        for prediction in id_predictions
    ]
    real_labels = [
        [label_list[l] for l in label]
        for label in id_labels
    ]
    return logits, id_predictions, id_labels, real_predictions, real_labels


def get_logits(trainer, samples, label_list):
    predictions, labels, _ = trainer.predict(samples)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l]  for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_predictions, true_labels
