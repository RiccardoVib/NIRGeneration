import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTConv(nn.Module):
    def __init__(self, filter_length, cond_dim=1):
        super().__init__()
        self.filter_length = filter_length
        self.cond_dim = cond_dim

        self.filters = nn.Linear(cond_dim, self.filter_length)
        self.phase_net = nn.Linear(cond_dim, self.filter_length)  # Phase control
        nn.init.normal_(self.filters.weight, 0, 0.02)
        nn.init.zeros_(self.filters.bias)
        nn.init.zeros_(self.phase_net.weight)  # Start with zero phase
        nn.init.zeros_(self.phase_net.bias)

    def train_step(self, X, C, targets, optimizer, criterion):
        optimizer.zero_grad()
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self, X, C, targets, criterion):
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        return loss.item()

    def forward(self, inputs, c):
        """
        FFT-domain convolution reverb emulation.
        Args:
            inputs: (batch_size, seq_len, 1) - dry audio
            c: (batch_size, 1, cond_dim) - conditioning for impulse response
        Returns:
            Reverbed output, same shape
        """
        batch_size, _, cond_dim = c.shape
        batch_size, seq_len, _ = inputs.shape

        # Generate per-batch impulse responses: (batch_size, filter_length)
        mag = torch.sigmoid(self.filters(c).squeeze(1))  # (batch_size, filter_length)
        phase = self.phase_net(c).squeeze(1)
        H = torch.polar(mag, phase)  # (batch, freq_bins) complex

        # FFT convolution per batch (vectorized)
        # H = torch.fft.rfft(impulse)  # (batch_size, filter_length//2 + 1)

        # Pad input: (batch_size, seq_len + filter_length, 1)
        #x_padded = F.pad(inputs, (0, 0, 0, self.filter_length))
        x_padded = F.pad(inputs, (0, 0, self.filter_length - 1, 0))  # pad at the START

        # Initialize output per batch
        y = torch.zeros(batch_size, seq_len + self.filter_length - 1,
                        dtype=H.dtype, device=inputs.device)

        block_size = self.filter_length // 2
        for i in range(0, seq_len, block_size):
            block = x_padded[:, i:i + block_size, 0]  # (batch_size, filter_length)

            X = torch.fft.fft(block, n=self.filter_length)
            Y_block = torch.fft.ifft(X * H, n=self.filter_length)
            # Overlap-add
            y[:, i:i + self.filter_length] += Y_block

        y = y[:, :seq_len, None].real  # Trim + channel dim

        return y


class FFTConv_real(nn.Module):
    def __init__(self, filter_length, cond_dim=1):
        super().__init__()
        self.filter_length = filter_length
        self.cond_dim = cond_dim

        self.filters = nn.Linear(cond_dim, filter_length)
        nn.init.normal_(self.filters.weight, 0, 0.02)
        nn.init.zeros_(self.filters.bias)

    def train_step(self, X, C, targets, optimizer, criterion):
        optimizer.zero_grad()
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        return loss.item()

    def val_step(self, X, C, targets, criterion):
        outputs = self.forward(X, C)
        loss = criterion(outputs, targets)

        return loss.item()

    def forward(self, inputs, c):
        """
        FFT-domain convolution reverb emulation.
        Args:
            inputs: (batch_size, seq_len, 1) - dry audio
            c: (batch_size, 1, cond_dim) - conditioning for impulse response
        Returns:
            Reverbed output, same shape
        """
        batch_size, _, cond_dim = c.shape
        batch_size, seq_len, _ = inputs.shape

        # Generate per-batch impulse responses: (batch_size, filter_length)
        mag = torch.sigmoid(self.filters(c).squeeze(1))  # (batch_size, filter_length)

        # FFT convolution per batch (vectorized)
        H = torch.fft.rfft(mag, n=self.filter_length)  # (batch_size, filter_length//2 + 1)

        # Pad input: (batch_size, seq_len + filter_length, 1)
        #x_padded = F.pad(inputs, (0, 0, 0, self.filter_length))
        x_padded = F.pad(inputs, (0, 0, self.filter_length - 1, 0))  # pad at the START

        # Initialize output per batch
        y = torch.zeros(batch_size, seq_len + self.filter_length - 1,
                        dtype=H.dtype, device=inputs.device)

        block_size = self.filter_length // 2
        for i in range(0, seq_len, block_size):
            block = x_padded[:, i:i + block_size, 0]  # (batch_size, filter_length)

            X = torch.fft.rfft(block, n=self.filter_length)
            Y_block = torch.fft.irfft(X * H, n=self.filter_length)
            # Overlap-add
            y[:, i:i + self.filter_length] += Y_block

        y = y[:, :seq_len, None].real  # Trim + channel dim

        return y
