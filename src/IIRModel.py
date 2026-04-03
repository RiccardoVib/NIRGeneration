import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IIR(nn.Module):
    def __init__(self, filter_length, cond_dim=1):
        super().__init__()
        self.filter_length = filter_length // 2
        self.cond_dim = cond_dim

        self.n_coeffs = self.filter_length + (self.filter_length - 1)
        self.filters = nn.Linear(cond_dim, 2 * self.n_coeffs)
        nn.init.zeros_(self.filters.weight)
        nn.init.zeros_(self.filters.bias)

    def train_step(self, X, C, targets, optimizer, criterion):
        optimizer.zero_grad()
        # outputs, poles_mags = self.forward(X, C)
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        # Penalize poles outside unit circle
        # stability_loss = torch.relu(poles_mags - 0.99).sum()
        # loss = loss + stability_loss
        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self, X, C, targets, criterion):
        # outputs, poles_mags = self.forward(X, C)
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        # Penalize poles outside unit circle
        # stability_loss = torch.relu(poles_mags - 0.99).sum()
        # loss = loss + stability_loss

        return loss.item()

    def _poles(self, a: torch.Tensor):
        """
        Compute poles as eigenvalues of the companion matrix of A(z).
        Differentiable through torch.linalg.eigvals.

        a: (batch, 1, filter_order)  — coefficients of z^{-1}...z^{-order}
        Returns pole magnitudes: (batch, filter_order)
        """
        batch = a.shape[0]
        order = self.filter_length
        a_flat = a.squeeze(1)  # (batch, order)

        # Build companion matrix for each item in batch
        # Monic polynomial: z^order + a[0]*z^{order-1} + ... + a[order-1]
        C = torch.zeros(batch, order, order, device=a.device, dtype=a.dtype)
        C[:, 1:, :-1] = torch.eye(order - 1, device=a.device).unsqueeze(0)
        C[:, :, -1] = -a_flat  # last column = -a coeffs

        poles = torch.linalg.eigvals(C)  # (batch, order) complex
        return torch.abs(poles)  # (batch, order) real

    def forward(self, inputs, c):
        """
        Differentiable IIR filter using FFT overlap-add.
        y[n] = sum(b[i] * x[n-i]) - sum(a[i] * y[n-i])
        Args:
            inputs: (1, seq_len, 1) - single channel audio (broadcasts over batch)
            c: (batch_size, 1, cond_dim) - conditioning
        Returns:
            Filtered output, same shape as inputs
        """
        batch_size, _, cond_dim = c.shape
        batch_size, seq_len, _ = inputs.shape

        # Expand inputs to batch: (batch_size, seq_len, 1)
        inputs = inputs.expand(batch_size, -1, -1)

        # Generate per-batch filters from conditioning
        filters = self.filters(c)  # (batch_size, filter_length-1)
        filters_real, filters_imag = torch.chunk(filters, 2, dim=-1)
        filters_complex = torch.complex(filters_real, filters_imag)

        # Split into b_coeffs and a_coeffs
        split_idx = self.filter_length - 1
        b_coeffs, a_coeffs_rest = torch.split(filters_complex, [self.filter_length, split_idx], dim=-1)
        # Prepend the implicit a0=1 before zero-padding
        a0 = torch.ones(*a_coeffs_rest.shape[:-1], 1, device=a_coeffs_rest.device, dtype=a_coeffs_rest.dtype)
        a_coeffs = torch.cat([a0, a_coeffs_rest], dim=-1)  # (batch, 1, filter_order+1)

        # poles_mags = torch.zeros_like(a_coeffs.real)#self._poles(a_coeffs)

        # Pad input: (batch_size, seq_len + filter_length, 1)
        # x_padded = F.pad(inputs, (0, 0, 0, self.filter_length))
        x_padded = F.pad(inputs, (0, 0, self.filter_length - 1, 0))  # pad at the START

        # Initialize output buffer per batch
        y = torch.zeros(batch_size, seq_len + self.filter_length, dtype=a_coeffs.dtype, device=inputs.device)

        # FFT filtering per batch item (vectorized over batch)
        B = torch.fft.fft(b_coeffs, n=self.filter_length)
        A = torch.fft.fft(a_coeffs, n=self.filter_length)
        H = B / (A + 1e-8)  # Avoid div-by-zero

        block_size = self.filter_length // 2
        for i in range(0, seq_len, block_size):
            block = x_padded[:, i: i + block_size]  # (batch, block_size)
            X = torch.fft.fft(block, n=self.filter_length, dim=1)  # (batch, n_fft//2+1)
            Y_block = torch.fft.ifft(X.permute(0, 2, 1) * H, n=self.filter_length, dim=-1).squeeze(1)
            end = i + self.filter_length

            # Overlap-add
            y[:, i:end] += Y_block  # overlap-add (no windowing)

        y = y[:, :seq_len, None].real  # Trim and add channel dim

        return y  # , poles_mags


def forward_time_domain(self, inputs, c):
    """
    Time-domain IIR filtering using Direct Form II Transposed.
    y[n] = sum_k(b[k] * x[n-k]) - sum_k(a[k] * y[n-k])  for k >= 1

    Args:
        inputs: (batch_size, seq_len, 1)
        c:      (batch_size, 1, cond_dim)
    Returns:
        Filtered output: (batch_size, seq_len, 1)
    """
    batch_size, seq_len, _ = inputs.shape

    # --- Generate coefficients (same as training forward) ---
    filters = self.filters(c)  # (batch, 1, 2*n_coeffs)
    filters_real, filters_imag = torch.chunk(filters, 2, dim=-1)
    filters_complex = torch.complex(filters_real, filters_imag)

    split_idx = self.filter_length - 1
    b_coeffs, a_coeffs_rest = torch.split(
        filters_complex, [self.filter_length, split_idx], dim=-1
    )  # (batch, 1, filter_length), (batch, 1, filter_length-1)

    # Prepend a[0] = 1
    a0 = torch.ones(*a_coeffs_rest.shape[:-1], 1,
                    device=a_coeffs_rest.device,
                    dtype=a_coeffs_rest.dtype)
    a_coeffs = torch.cat([a0, a_coeffs_rest], dim=-1)  # (batch, 1, filter_length)

    # Use real parts only for audio processing
    b = b_coeffs.squeeze(1).real  # (batch, filter_length)
    a = a_coeffs.squeeze(1).real  # (batch, filter_length)
    # a[:, 0] == 1 always — skip dividing by it

    x = inputs.squeeze(-1)  # (batch, seq_len)
    y = torch.zeros_like(x)

    order = self.filter_length  # = len(b) = len(a)

    # Direct Form II Transposed delay line
    # w: (batch, order) state buffer
    w = torch.zeros(batch_size, order, device=x.device, dtype=x.dtype)

    for n in range(seq_len):
        # w[0] holds the current input after feedback
        xn = x[:, n]  # (batch,)

        # Output: b[0]*w[0] already handled below via state
        yn = b[:, 0] * xn + w[:, 0]  # (batch,)
        y[:, n] = yn

        # Shift delay line (Direct Form II Transposed update)
        new_w = torch.zeros_like(w)
        for k in range(1, order):
            new_w[:, k - 1] = b[:, k] * xn - a[:, k] * yn + (w[:, k] if k < order - 1 else 0.0)
        w = new_w

    return y.unsqueeze(-1)


class IIR_real(nn.Module):
    def __init__(self, filter_length, cond_dim=1):
        super().__init__()
        self.filter_length = filter_length // 2
        self.cond_dim = cond_dim

        self.n_coeffs = self.filter_length + (self.filter_length - 1)
        self.filters = nn.Linear(cond_dim, self.n_coeffs)
        nn.init.zeros_(self.filters.weight)
        nn.init.zeros_(self.filters.bias)

    def train_step(self, X, C, targets, optimizer, criterion):
        optimizer.zero_grad()
        # outputs, poles_mags = self.forward(X, C)
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        # Penalize poles outside unit circle
        # stability_loss = torch.relu(poles_mags - 0.99).sum()
        # loss = loss + stability_loss
        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self, X, C, targets, criterion):
        # outputs, poles_mags = self.forward(X, C)
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        # Penalize poles outside unit circle
        # stability_loss = torch.relu(poles_mags - 0.99).sum()
        # loss = loss + stability_loss

        return loss.item()

    def _poles(self, a: torch.Tensor):
        """
        Compute poles as eigenvalues of the companion matrix of A(z).
        Differentiable through torch.linalg.eigvals.

        a: (batch, 1, filter_order)  — coefficients of z^{-1}...z^{-order}
        Returns pole magnitudes: (batch, filter_order)
        """
        batch = a.shape[0]
        order = self.filter_length
        a_flat = a.squeeze(1)  # (batch, order)

        # Build companion matrix for each item in batch
        # Monic polynomial: z^order + a[0]*z^{order-1} + ... + a[order-1]
        C = torch.zeros(batch, order, order, device=a.device, dtype=a.dtype)
        C[:, 1:, :-1] = torch.eye(order - 1, device=a.device).unsqueeze(0)
        C[:, :, -1] = -a_flat  # last column = -a coeffs

        poles = torch.linalg.eigvals(C)  # (batch, order) complex
        return torch.abs(poles)  # (batch, order) real

    def forward(self, inputs, c):
        """
        Differentiable IIR filter using FFT overlap-add.
        y[n] = sum(b[i] * x[n-i]) - sum(a[i] * y[n-i])
        Args:
            inputs: (1, seq_len, 1) - single channel audio (broadcasts over batch)
            c: (batch_size, 1, cond_dim) - conditioning
        Returns:
            Filtered output, same shape as inputs
        """
        batch_size, _, cond_dim = c.shape
        batch_size, seq_len, _ = inputs.shape

        # Expand inputs to batch: (batch_size, seq_len, 1)
        inputs = inputs.expand(batch_size, -1, -1)

        # Generate per-batch filters from conditioning
        filters = self.filters(c)  # (batch_size, filter_length-1)

        # Split into b_coeffs and a_coeffs
        split_idx = self.filter_length - 1
        b_coeffs, a_coeffs_rest = torch.split(filters, [self.filter_length, split_idx], dim=-1)
        # Prepend the implicit a0=1 before zero-padding
        a0 = torch.ones(*a_coeffs_rest.shape[:-1], 1, device=a_coeffs_rest.device, dtype=a_coeffs_rest.dtype)
        a_coeffs = torch.cat([a0, a_coeffs_rest], dim=-1)  # (batch, 1, filter_order+1)

        # poles_mags = torch.zeros_like(a_coeffs.real)#self._poles(a_coeffs)

        # Pad input: (batch_size, seq_len + filter_length, 1)
        # x_padded = F.pad(inputs, (0, 0, 0, self.filter_length))
        x_padded = F.pad(inputs, (0, 0, self.filter_length - 1, 0))  # pad at the START

        # Initialize output buffer per batch
        y = torch.zeros(batch_size, seq_len + self.filter_length, dtype=a_coeffs.dtype, device=inputs.device)

        # FFT filtering per batch item (vectorized over batch)
        B = torch.fft.rfft(b_coeffs, n=self.filter_length)
        A = torch.fft.rfft(a_coeffs, n=self.filter_length)
        H = B / (A + 1e-8)  # Avoid div-by-zero

        block_size = self.filter_length // 2
        for i in range(0, seq_len, block_size):
            block = x_padded[:, i: i + block_size]  # (batch, block_size)
            X = torch.fft.rfft(block, n=self.filter_length, dim=1)  # (batch, n_fft//2+1)
            Y_block = torch.fft.irfft(X.permute(0, 2, 1) * H, n=self.filter_length, dim=-1).squeeze(1)
            end = i + self.filter_length

            # Overlap-add
            y[:, i:end] += Y_block  # overlap-add (no windowing)

        y = y[:, :seq_len, None].real  # Trim and add channel dim

        return y  # , poles_mags