import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, filters, window_size=8, attention_heads=4, activation='relu', verbose=False):
        super(SelfAttention, self).__init__()
        if filters % attention_heads != 0:
            if verbose:
                print(f"Warning: {filters} no es divisible por {attention_heads}. Ajustando filters.")
            filters = filters - (filters % attention_heads)
            filters = max(filters, attention_heads)

        self.filters = filters
        self.attention_heads = attention_heads
        self.window_size = window_size
        self.verbose = verbose
        self.d_head = self.filters // self.attention_heads  # Canales por cabeza
        
        # Convertir el string de activación a objeto de activación
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            else:
                self.activation = nn.ReLU()  # Default
        else:
            self.activation = activation
        
        # Inicializar convoluciones como None para configurarlas en forward()
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.projection_conv = None

    def forward(self, x):
        B, C, H, W = x.shape
        ws = min(self.window_size, H, W)  # Asegurar que `window_size` no sea mayor que H o W
        
        # Padding si H o W no son múltiplos de window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]
        
        num_windows_h = H // ws
        num_windows_w = W // ws
        x_windows = x.view(B, C, num_windows_h, ws, num_windows_w, ws)
        x_windows = x_windows.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x_windows.view(-1, C, ws, ws)

        # Ajustar convoluciones dinámicamente
        if self.query_conv is None or self.query_conv.in_channels != C:
            self.query_conv = nn.Conv2d(in_channels=C, out_channels=self.filters, kernel_size=1).to(x.device)
            self.key_conv = nn.Conv2d(in_channels=C, out_channels=self.filters, kernel_size=1).to(x.device)
            self.value_conv = nn.Conv2d(in_channels=C, out_channels=self.filters, kernel_size=1).to(x.device)

        Q = self.query_conv(windows)
        K = self.key_conv(windows)
        V = self.value_conv(windows)

        B_w, C_w, H_w, W_w = Q.shape
        N = H_w * W_w
        Q = Q.view(B_w, self.attention_heads, self.d_head, N).permute(0, 1, 3, 2)
        K = K.view(B_w, self.attention_heads, self.d_head, N).permute(0, 1, 3, 2)
        V = V.view(B_w, self.attention_heads, self.d_head, N).permute(0, 1, 3, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out_window = torch.matmul(attn, V)

        out_window = out_window.permute(0, 1, 3, 2).contiguous()

        # Validación antes de `view()`
        expected_elements_out = B * C * num_windows_h * ws * num_windows_w * ws
        actual_elements_out = out_window.numel()

        if expected_elements_out != actual_elements_out:
            if self.verbose:
                print(f"⚠️ ERROR: Tamaño incompatible en `view()`")
                print(f"Esperado: {expected_elements_out}, Real: {actual_elements_out}")
                print(f"Forma de `out_window` antes de `view()`: {out_window.shape}")
            
            # Ajuste seguro
            out = out_window.reshape(B, C, -1, num_windows_w * ws)
        else:
            out = out_window.view(B, C, num_windows_h * ws, num_windows_w * ws)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H - pad_h, :W - pad_w]
        return out


class DontCareLayer(nn.Module):
    def __init__(self):
        super(DontCareLayer, self).__init__()

    def forward(self, x):
        return x