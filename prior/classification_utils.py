from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score


def _normalize_degrade_type(value):
    name = str(value).strip()
    if not name:
        raise ValueError('degrade_type entries must be non-empty strings.')
    return name


def split_degrade_type_tokens(degrade_type):
    parts = [part.strip() for part in _normalize_degrade_type(degrade_type).split('-') if part.strip()]
    if not parts:
        raise ValueError(f'Invalid degrade_type: {degrade_type!r}')

    tokens = []
    for part in parts:
        if part not in tokens:
            tokens.append(part)
    return tuple(tokens)


@dataclass(frozen=True)
class ClassificationSpec:
    degrade_types: tuple[str, ...]
    label_names: tuple[str, ...]
    degrade_to_indices: dict[str, tuple[int, ...]]
    multi_label: bool = False
    threshold: float = 0.5
    dataset_name: str | None = None

    @property
    def cls_num(self):
        return len(self.label_names)

    @property
    def metric_mode(self):
        return 'multi_label' if self.multi_label else 'single_label'

    def encode_one(self, degrade_type, device=None):
        degrade_key = _normalize_degrade_type(degrade_type)
        if degrade_key not in self.degrade_to_indices:
            raise KeyError(f'Unknown degrade_type: {degrade_key}')

        if self.multi_label:
            target = torch.zeros(self.cls_num, dtype=torch.float32, device=device)
            target[list(self.degrade_to_indices[degrade_key])] = 1.0
            return target

        return torch.tensor(self.degrade_to_indices[degrade_key][0], dtype=torch.long, device=device)

    def encode_batch(self, degrade_types, device=None):
        encoded = [self.encode_one(degrade_type, device=device) for degrade_type in degrade_types]
        if self.multi_label:
            return torch.stack(encoded, dim=0)
        return torch.stack(encoded, dim=0).long()

    def probabilities_from_logits(self, logits):
        if self.multi_label:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    def predict_from_logits(self, logits):
        probs = self.probabilities_from_logits(logits)
        if self.multi_label:
            return (probs >= self.threshold).to(dtype=torch.int64)
        return probs.argmax(dim=1)

    def batch_accuracy(self, preds, labels):
        if self.multi_label:
            return (preds == labels.to(dtype=preds.dtype)).all(dim=1).float().mean()
        return (preds == labels).float().mean()


class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)

def build_classification_spec(
        degrade_types,
        *,
        multi_label=False,
        dataset_name=None,
        threshold=0.5):
    normalized = tuple(_normalize_degrade_type(item) for item in degrade_types)
    if not normalized:
        raise ValueError('degrade_types must not be empty.')

    if not multi_label:
        return ClassificationSpec(
            degrade_types=normalized,
            label_names=normalized,
            degrade_to_indices={name: (idx,) for idx, name in enumerate(normalized)},
            multi_label=False,
            threshold=float(threshold),
            dataset_name=dataset_name)

    label_names = []
    label_to_index = {}
    degrade_to_indices = {}
    for degrade_type in normalized:
        token_indices = []
        for token in split_degrade_type_tokens(degrade_type):
            if token not in label_to_index:
                label_to_index[token] = len(label_names)
                label_names.append(token)
            token_indices.append(label_to_index[token])
        degrade_to_indices[degrade_type] = tuple(token_indices)

    return ClassificationSpec(
        degrade_types=normalized,
        label_names=tuple(label_names),
        degrade_to_indices=degrade_to_indices,
        multi_label=True,
        threshold=float(threshold),
        dataset_name=dataset_name)


def build_classification_loss(multi_label=False, label_smoothing=0.0):
    if multi_label:
        return BCEWithLogitsLossWithLabelSmoothing(label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def summarize_prediction_outputs(outputs, classification_spec):
    if not outputs:
        return None

    preds = torch.cat([item['preds'] for item in outputs], dim=0)
    labels = torch.cat([item['labels'] for item in outputs], dim=0)
    degrade_types = []
    for item in outputs:
        degrade_types.extend([_normalize_degrade_type(value) for value in item['degrade_type']])

    return summarize_predictions(preds, labels, degrade_types, classification_spec)


def summarize_predictions(preds, labels, sample_degrade_types, classification_spec):
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.asarray(preds)

    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)

    sample_degrade_types = [_normalize_degrade_type(item) for item in sample_degrade_types]
    if len(sample_degrade_types) != len(labels):
        raise ValueError(
            f'sample_degrade_types length ({len(sample_degrade_types)}) does not match labels ({len(labels)}).')

    if classification_spec.multi_label:
        return _summarize_multilabel_predictions(preds, labels, sample_degrade_types, classification_spec)
    return _summarize_single_label_predictions(preds, labels, sample_degrade_types, classification_spec)


def _summarize_single_label_predictions(preds, labels, sample_degrade_types, classification_spec):
    labels = labels.astype(np.int64, copy=False).reshape(-1)
    preds = preds.astype(np.int64, copy=False).reshape(-1)

    per_type = []
    for degrade_type in classification_spec.degrade_types:
        class_id = classification_spec.degrade_to_indices[degrade_type][0]
        mask = labels == class_id
        total = int(mask.sum())
        correct = int((preds[mask] == labels[mask]).sum()) if total > 0 else 0
        per_type.append({
            'degrade_type': degrade_type,
            'accuracy': correct / total if total > 0 else float('nan'),
            'correct': correct,
            'total': total,
        })

    overall_accuracy = float((preds == labels).mean()) if len(labels) > 0 else float('nan')
    macro_accuracy = float(np.nanmean([item['accuracy'] for item in per_type])) if per_type else float('nan')
    macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0)) if len(labels) > 0 else float('nan')

    return {
        'metric_mode': classification_spec.metric_mode,
        'labels': labels,
        'preds': preds,
        'degrade_types': list(classification_spec.degrade_types),
        'sample_degrade_types': list(sample_degrade_types),
        'overall_accuracy': overall_accuracy,
        'macro_accuracy': macro_accuracy,
        'macro_f1': macro_f1,
        'per_type_accuracy': per_type,
        'num_samples': int(len(labels)),
    }


def _summarize_multilabel_predictions(preds, labels, sample_degrade_types, classification_spec):
    labels = labels.astype(np.int64, copy=False)
    preds = preds.astype(np.int64, copy=False)
    exact_matches = (preds == labels).all(axis=1)

    per_type = []
    sample_degrade_types_np = np.asarray(sample_degrade_types)
    for degrade_type in classification_spec.degrade_types:
        mask = sample_degrade_types_np == degrade_type
        total = int(mask.sum())
        correct = int(exact_matches[mask].sum()) if total > 0 else 0
        per_type.append({
            'degrade_type': degrade_type,
            'accuracy': correct / total if total > 0 else float('nan'),
            'correct': correct,
            'total': total,
        })

    per_label = []
    for idx, label_name in enumerate(classification_spec.label_names):
        y_true = labels[:, idx]
        y_pred = preds[:, idx]
        per_label.append({
            'label': label_name,
            'accuracy': float((y_true == y_pred).mean()) if len(y_true) > 0 else float('nan'),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else float('nan'),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else float('nan'),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else float('nan'),
            'support': int(y_true.sum()),
        })

    exact_match_accuracy = float(accuracy_score(labels, preds)) if len(labels) > 0 else float('nan')
    macro_accuracy = float(np.nanmean([item['accuracy'] for item in per_type])) if per_type else float('nan')
    hamming_accuracy = float(1.0 - hamming_loss(labels, preds)) if len(labels) > 0 else float('nan')

    return {
        'metric_mode': classification_spec.metric_mode,
        'labels': labels,
        'preds': preds,
        'degrade_types': list(classification_spec.degrade_types),
        'label_names': list(classification_spec.label_names),
        'sample_degrade_types': list(sample_degrade_types),
        'overall_accuracy': exact_match_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'macro_accuracy': macro_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'micro_precision': float(precision_score(labels, preds, average='micro', zero_division=0)) if len(labels) > 0 else float('nan'),
        'micro_recall': float(recall_score(labels, preds, average='micro', zero_division=0)) if len(labels) > 0 else float('nan'),
        'micro_f1': float(f1_score(labels, preds, average='micro', zero_division=0)) if len(labels) > 0 else float('nan'),
        'macro_precision': float(precision_score(labels, preds, average='macro', zero_division=0)) if len(labels) > 0 else float('nan'),
        'macro_recall': float(recall_score(labels, preds, average='macro', zero_division=0)) if len(labels) > 0 else float('nan'),
        'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)) if len(labels) > 0 else float('nan'),
        'per_type_accuracy': per_type,
        'per_label_metrics': per_label,
        'num_samples': int(len(labels)),
    }


__all__ = [
    'BCEWithLogitsLossWithLabelSmoothing',
    'ClassificationSpec',
    'build_classification_loss',
    'build_classification_spec',
    'split_degrade_type_tokens',
    'summarize_prediction_outputs',
    'summarize_predictions',
]