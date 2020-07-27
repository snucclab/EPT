from typing import Tuple, Dict

import torch
from torch import nn

from page.const import PAD_ID


class SmoothedCrossEntropyLoss(nn.Module):
    """
    Computes cross entropy loss with uniformly smoothed targets.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = PAD_ID, reduction: str = 'batchmean'):
        """
        Cross entropy loss with uniformly smoothed targets.

        :param float smoothing: Label smoothing factor, between 0 and 1 (exclusive; default is 0.1)
        :param int ignore_index: Index to be ignored. (PAD_ID by default)
        :param str reduction: Style of reduction to be done. One of 'batchmean'(default), 'none', or 'sum'.
        """
        assert 0 < smoothing < 1, "Smoothing factor should be in (0.0, 1.0)"
        assert reduction in {'batchmean', 'none', 'sum'}
        super().__init__()

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Computes cross entropy loss with uniformly smoothed targets.
        Since the entropy of smoothed target distribution is always same, we can compute this with KL-divergence.

        :param torch.Tensor input: Log probability for each class. This is a Tensor with shape [B, C]
        :param torch.LongTensor target: List of target classes. This is a LongTensor with shape [B]
        :rtype: torch.Tensor
        :return: Computed loss
        """
        target = target.view(-1, 1)

        # Prepare smoothed target
        # Set all probability of the targets which should be ignored as zero.
        # Since D_KL(p, q) = p (log(p) - log(q)), by setting p(x) ≡ 0, these target cannot affect loss anymore.
        smoothed_target = torch.zeros(input.shape, requires_grad=False, device=target.device)

        # Set target values zero if predicted values are masked with -inf.
        for r, row in enumerate(input):
            tgt = target[r].item()
            if tgt == self.ignore_index:
                continue

            finites = torch.isfinite(row)
            n_cls = finites.sum().item()
            assert n_cls > 0

            smoothing_prob = self.smoothing / n_cls
            smoothed_target[r].masked_fill_(finites, smoothing_prob)
            smoothed_target[r, tgt] = 1.0 - self.smoothing

        # Compute loss: - p log q
        loss = - smoothed_target * input.masked_fill(~torch.isfinite(input), 0.0)

        if self.reduction == 'batchmean':
            return loss.sum() / input.shape[0]
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def accuracy(greedy_choice_correct: torch.Tensor, target_focus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute accuracy by comparing two Bool Tensors

    :param torch.Tensor greedy_choice_correct:
        Bool Tensor indicating whether prediction is correct or not.
        `True` if greedy choice based on prediction is correct on the entry.
    :param torch.Tensor target_focus:
        Bool Tensor indicating whether we interested in the position or not.
        `True` if we don't ignore the position.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    :return:
        Tuple of two Float Tensors.
        - [0] indicates token level accuracy.
        - [1] indicates sequence level accuracy.
    """
    with torch.no_grad():
        token_lv_acc = (greedy_choice_correct & target_focus).sum().float()
        token_lv_acc /= target_focus.sum().float()

        # Set NaN as 1.0 since there are no such token to measure the accuracy.
        token_lv_acc.masked_fill_(torch.isnan(token_lv_acc), 1.0)

        if target_focus.dim() > 1:
            # Case of [B, *]. (multiple values per a sequence)
            seq_lv_acc = ((~greedy_choice_correct & target_focus).sum(dim=-1) == 0).sum().float()  # Add by batch
            seq_lv_acc /= greedy_choice_correct.shape[0]
        else:
            # Case of predicting a single value per a sequence.
            seq_lv_acc = token_lv_acc

    return token_lv_acc, seq_lv_acc


def loss_and_accuracy(predicted: torch.Tensor, target: torch.Tensor, prefix: str,
                      loss_factor: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Compute loss and accuracy. Loss will be selected by following rules.
    - If target.dim + 1 == predicted.dim and target: LongTensor and predicted: FloatTensor -> use Cross-Entropy
    - If target and predicted dimensions are the same and both are FloatTensor -> use KL-divergence
    - If target and predicted dimensions are the same and target: BoolTensor and predicted: FloatTensor -> use BinaryCE.

    :param torch.Tensor predicted: Tensor of predicted result.
    :param torch.Tensor target: Tensor of targeted result.
    :param str prefix: String prefix for dictionary keys.
    :param float loss_factor: Factor for up- or down-weighting loss. (1.0 by default)
    :rtype: Dict[str, torch.Tensor]
    :return: Dictionary that contains the following items
        - [prefix]/loss: Loss value
        - [prefix]/acc_seq: Sequence level accuracy
        - [prefix]/acc_token: Token level accuracy.
    """

    tdim = target.dim()
    pdim = predicted.dim()
    tdtype = target.dtype

    result = {}

    if tdtype == torch.long and tdim + 1 == pdim:
        # This is the case for Cross-Entropy.
        # Compute accuracy.
        target_focus = target != PAD_ID
        greedy_choice_correct = predicted.argmax(dim=-1) == target
        token_lv_acc, seq_lv_acc = accuracy(greedy_choice_correct, target_focus)

        # Flatten predicted to [*, C] and target to [*]
        predicted = predicted.flatten(0, -2)
        target = target.flatten()

        # Prepare loss function
        # loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        loss_fct = SmoothedCrossEntropyLoss(ignore_index=PAD_ID)

        result.update({
            'acc_token': token_lv_acc,
            'acc_seq': seq_lv_acc
        })
    else:
        raise NotImplementedError('There are no such rules for computing loss between %s-dim predicted %s tensor '
                                  'and %s-dim target %s tensor' % (pdim, predicted.dtype, tdim, tdtype))

    # Compute loss
    loss = loss_fct(predicted, target)

    if loss_factor != 1.0:
        loss = loss * loss_factor

    # For debugging.
    if not torch.isfinite(loss).all().item():
        print('NAN')

    result['loss'] = loss

    return {prefix + '/' + key: value for key, value in result.items()}


__all__ = ['loss_and_accuracy']
