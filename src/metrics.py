"""Evaluation metrics: Character Error Rate, Word Error Rate, accuracy."""

import jiwer


def cer(reference, hypothesis):
    """Character Error Rate for a single or list of strings."""
    return jiwer.cer(reference, hypothesis)


def wer(reference, hypothesis):
    """Word Error Rate for a single or list of strings."""
    return jiwer.wer(reference, hypothesis)


def exact_match_accuracy(references, hypotheses):
    """Fraction of predictions that match the reference exactly (case-insensitive)."""
    if not references:
        return 0.0
    correct = sum(
        1
        for ref, hyp in zip(references, hypotheses)
        if ref.strip().lower() == hyp.strip().lower()
    )
    return correct / len(references)


def char_accuracy(gt, pred):
    """Positional character accuracy for a single pair (0-100)."""
    gt, pred = gt.lower(), pred.lower()
    total = max(len(gt), len(pred))
    if total == 0:
        return 100.0
    correct = sum(1 for a, b in zip(gt, pred) if a == b)
    return correct / total * 100
