"""Repeated-response aggregation happens after F_s projection and before the prior."""

from collections import defaultdict
import contextlib
import torch


@torch.no_grad()
def aggregate_conditions(model, loader, device, config):
    sums = {}
    counts = defaultdict(int)
    events = defaultdict(list)
    for batch in loader:
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and config["precision"] == "bf16"
            else contextlib.nullcontext()
        )
        with amp:
            conditions = model.condition(batch["eeg"].to(device))
        conditions = conditions.float().cpu()
        if not torch.isfinite(conditions).all():
            raise FloatingPointError("Non-finite F_s")
        for i, visual in enumerate(batch["visual_index"].tolist()):
            sums[visual] = (
                sums.get(visual, torch.zeros_like(conditions[i])) + conditions[i]
            )
            counts[visual] += 1
            events[visual].append(int(batch["trial_row"][i]))
    if not sums:
        raise ValueError("No EEG responses")
    return {v: sums[v] / counts[v] for v in sorted(sums)}, dict(counts), dict(events)
