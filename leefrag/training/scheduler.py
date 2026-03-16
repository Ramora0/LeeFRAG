from leefrag.config import TrainingConfig


class CompressionScheduler:
    """Schedule compression ratio across training steps.

    Each compression level gets an equal share of total steps.
    Default: [2, 4, 8, 16] over 4 epochs → 1 epoch per ratio.
    """

    def __init__(self, config: TrainingConfig, total_steps: int):
        self.ratios = config.compression_schedule
        self.total_steps = total_steps
        self.steps_per_phase = total_steps // len(self.ratios)

    def get_compression_ratio(self, step: int) -> int:
        """Return the compression ratio for the given training step."""
        phase_idx = min(step // self.steps_per_phase, len(self.ratios) - 1)
        return self.ratios[phase_idx]

    def get_phase(self, step: int) -> int:
        """Return the current phase index (0-based)."""
        return min(step // self.steps_per_phase, len(self.ratios) - 1)

    def get_compression_ratio_by_progress(self, progress: float) -> int:
        """Return the compression ratio for a time-based progress fraction [0, 1]."""
        phase_idx = min(int(progress * len(self.ratios)), len(self.ratios) - 1)
        return self.ratios[phase_idx]

    def get_phase_by_progress(self, progress: float) -> int:
        """Return the current phase index for a time-based progress fraction [0, 1]."""
        return min(int(progress * len(self.ratios)), len(self.ratios) - 1)
