import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Multi-scale STFT loss combining:
      - Log-magnitude L1 (spectral convergence variant)
      - Linear-magnitude L1
      - Phase-aware loss (Instantaneous Frequency or direct phase L1)
    """

    def __init__(
        self,
        fft_sizes:      list[int]   = [2048, 1024, 512, 256],
        hop_ratios:     list[float] = [0.25, 0.25, 0.25, 0.25],
        win_ratios:     list[float] = [1.0,  1.0,  1.0,  1.0],
        window:         str         = "hann",
        mag_weight:     float       = 1.0,
        log_mag_weight: float       = 1.0,
        phase_weight:   float       = 0.1,
        phase_mode:     str         = "direct",   # "if" | "direct" | "none"
    ):
        """
        Args:
            fft_sizes:      list of FFT sizes for multi-scale analysis
            hop_ratios:     hop_size = fft_size * hop_ratio  per scale
            win_ratios:     win_size = fft_size * win_ratio  per scale
            window:         window type: "hann" | "hamming" | "blackman"
            mag_weight:     weight for linear magnitude L1 loss
            log_mag_weight: weight for log-magnitude L1 loss
            phase_weight:   weight for phase loss term
            phase_mode:     "if"     → Instantaneous Frequency loss (recommended)
                            "direct" → wrapped phase L1 (simpler)
                            "none"   → disable phase loss
        """
        super().__init__()
        assert phase_mode in ("if", "direct", "none")

        self.fft_sizes      = fft_sizes
        self.hop_sizes      = [max(1, int(n * r)) for n, r in zip(fft_sizes, hop_ratios)]
        self.win_sizes      = [max(1, int(n * r)) for n, r in zip(fft_sizes, win_ratios)]
        self.mag_weight     = mag_weight
        self.log_mag_weight = log_mag_weight
        self.phase_weight   = phase_weight
        self.phase_mode     = phase_mode
        self.phase_warmup_steps = 5000

        # Pre-register windows as buffers (one per scale)
        for i, win_len in enumerate(self.win_sizes):
            win = self._make_window(window, win_len)
            self.register_buffer(f"window_{i}", win)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_window(self, name: str, length: int) -> torch.Tensor:
        if name == "hann":
            return torch.hann_window(length)
        elif name == "hamming":
            return torch.hamming_window(length)
        elif name == "blackman":
            return torch.blackman_window(length)
        else:
            raise ValueError(f"Unknown window: {name}")

    def _stft(
        self,
        x:        torch.Tensor,   # (B, T)
        fft_size: int,
        hop_size: int,
        win_size: int,
        window:   torch.Tensor,   # (win_size,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mag:   (B, F, T')  linear magnitude
            phase: (B, F, T')  instantaneous phase in [-π, π]
        """
        x_padded = F.pad(x, (win_size // 2, 0))  # causal left-pad only

        # torch.stft requires (B, T) or (T,)
        S = torch.stft(
            x_padded,
            n_fft       = fft_size,
            hop_length  = hop_size,
            win_length  = win_size,
            window      = window,
            return_complex = True,
            pad_mode    = "reflect",
            center      = False,
        )                                   # (B, F, T') complex
        mag   = S.abs()                     # (B, F, T')
        phase = S.angle()                   # (B, F, T') ∈ [-π, π]
        return mag, phase

    # ------------------------------------------------------------------
    # Phase losses
    # ------------------------------------------------------------------

    @staticmethod
    def _instantaneous_frequency_loss(
        phase_pred: torch.Tensor,   # (B, F, T')
        phase_tgt:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Instantaneous Frequency loss: compare finite differences along
        time of the unwrapped phase. Diff of wrapped phase ≈ IF when
        within [-π, π], no explicit unwrap needed.

        IF[n] = phase[n] - phase[n-1]  (mod 2π, mapped to [-π, π])
        """
        def _if(p: torch.Tensor) -> torch.Tensor:
            dp = p[..., 1:] - p[..., :-1]          # (B, F, T'-1)
            return torch.remainder(dp + torch.pi, 2 * torch.pi) - torch.pi

        if_pred = _if(phase_pred)
        if_tgt  = _if(phase_tgt)

        # Circular L1
        diff = torch.remainder(if_pred - if_tgt + torch.pi, 2 * torch.pi) - torch.pi
        return diff.abs().mean()

    @staticmethod
    def _direct_phase_loss(
        phase_pred: torch.Tensor,
        phase_tgt:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Circular L1 on raw phase.  Handles the 2π wrap.
        """
        diff = torch.remainder(phase_pred - phase_tgt + torch.pi, 2 * torch.pi) - torch.pi
        return diff.abs().mean()

    # ------------------------------------------------------------------
    # Magnitude losses
    # ------------------------------------------------------------------

    @staticmethod
    def _log_mag_loss(mag_pred: torch.Tensor, mag_tgt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        log_pred = torch.log(mag_pred.clamp(min=eps))
        log_tgt  = torch.log(mag_tgt.clamp(min=eps))
        return F.l1_loss(log_pred, log_tgt)

    @staticmethod
    def _lin_mag_loss(mag_pred: torch.Tensor, mag_tgt: torch.Tensor) -> torch.Tensor:
        # Normalize by target energy for scale-invariance
        denom = mag_tgt.norm() + 1e-8
        return (mag_pred - mag_tgt).norm() / denom

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred:   torch.Tensor,   # (B, T) or (B, T, 1)
        target: torch.Tensor,   # (B, T) or (B, T, 1)
        step=0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor (differentiable)
            breakdown:  dict with per-term losses for logging
        """
        # Flatten channel dim if present
        if pred.dim() == 3:
            pred   = pred.squeeze(-1)
            target = target.squeeze(-1)

        # Disable phase loss for first N steps — let magnitude converge first
        phase_w = self.phase_weight if step > self.phase_warmup_steps else 0.0

        total      = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        loss_mag   = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        loss_log   = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        loss_phase = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        n_scales   = len(self.fft_sizes)

        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            window = getattr(self, f"window_{i}")

            mag_p, phase_p = self._stft(pred,   fft_size, hop_size, win_size, window)
            mag_t, phase_t = self._stft(target, fft_size, hop_size, win_size, window)

            loss_mag += self._lin_mag_loss(mag_p, mag_t)
            loss_log += self._log_mag_loss(mag_p, mag_t)

            if self.phase_mode == "if":
                loss_phase += self._instantaneous_frequency_loss(phase_p, phase_t)
            elif self.phase_mode == "direct":
                loss_phase += self._direct_phase_loss(phase_p, phase_t)

        # Average over scales
        loss_mag   /= n_scales
        loss_log   /= n_scales
        loss_phase /= n_scales

        total = (
            self.mag_weight     * loss_mag
          + self.log_mag_weight * loss_log
          + phase_w   * loss_phase
        )

        breakdown = {
            "loss/mag":   loss_mag.item(),
            "loss/log":   loss_log.item(),
            "loss/phase": loss_phase.item(),
            "loss/total": total.item(),
        }

        return total#, breakdown