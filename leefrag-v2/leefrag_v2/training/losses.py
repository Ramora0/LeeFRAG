"""Loss functions: CE on answer tokens and KL to the offline teacher."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ce_on_answer(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token CE over answer positions (labels == -100 elsewhere)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def kl_to_teacher(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_vals: torch.Tensor,
    teacher_idx: torch.Tensor,
) -> torch.Tensor:
    """Top-k KL(teacher || student) over answer positions.

    teacher_vals/teacher_idx: [N_ans, k] precomputed offline by precompute_teacher.py
    at the SAME (shifted) answer positions that ce_on_answer supervises.
    """
    shift_logits = student_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    student_ans = shift_logits[mask]  # [N_ans, V]

    n = min(student_ans.shape[0], teacher_vals.shape[0])
    if n == 0:
        return student_logits.new_zeros(())
    student_ans = student_ans[:n]
    teacher_vals = teacher_vals[:n].to(student_ans.device)
    teacher_idx = teacher_idx[:n].to(student_ans.device)

    gathered = student_ans.gather(-1, teacher_idx)  # [n, k]
    t_logp = F.log_softmax(teacher_vals.float(), dim=-1)
    s_logp = F.log_softmax(gathered.float(), dim=-1)
    return F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")
