import torch
import torch.nn as nn

class PIDeepONet1D(nn.Module):
    def __init__(self, branch_layers=[256, 128, 128, 64],
                 trunk_layers=[2, 128, 128, 64],
                 activation=nn.Tanh, nu=0.001):
        super().__init__()
        self.nu = nu
        self.branch_dim = branch_layers[0]   # 256
        self.trunk_dim = trunk_layers[0]     # 2

        # Branch net
        layers = []
        for i in range(len(branch_layers) - 1):
            layers.append(nn.Linear(branch_layers[i], branch_layers[i+1]))
            if i < len(branch_layers) - 2:
                layers.append(activation())
        self.branch = nn.Sequential(*layers)

        # Trunk net
        layers = []
        for i in range(len(trunk_layers) - 1):
            layers.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            if i < len(trunk_layers) - 2:
                layers.append(activation())
        self.trunk = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: 展平后的拼接输入 [..., branch_dim + trunk_dim]
        例如 [B*N, 258]
        """
        branch_input = x[..., :self.branch_dim]   # [..., 256]
        trunk_input = x[..., self.branch_dim:]    # [..., 2]
        b = self.branch(branch_input)             # [..., hidden]
        t = self.trunk(trunk_input)               # [..., hidden]
        return torch.sum(b * t, dim=-1)           # [..., 1] or [...]

    def compute_pde_residual(self, branch_input, trunk_input):
        """
        计算 Burgers 残差，需要 (x,t) 坐标的梯度。
        branch_input: [B, 256]
        trunk_input:  [B, N, 2]
        """
        x = trunk_input[..., 0:1]   # [B, N, 1]
        t = trunk_input[..., 1:2]

        x.requires_grad_(True)
        t.requires_grad_(True)
        trunk = torch.cat([x, t], dim=-1)   # [B, N, 2]

        B, N, _ = trunk.shape
        # 构建展平输入以复用 forward
        branch_exp = branch_input.unsqueeze(1).expand(-1, N, -1)    # [B, N, 256]
        inp = torch.cat([branch_exp, trunk], dim=-1)               # [B, N, 258]
        inp_flat = inp.reshape(-1, self.branch_dim + self.trunk_dim)

        u_flat = self.forward(inp_flat)       # [B*N]
        u = u_flat.view(B, N)                 # [B, N]

        # 导数
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]            # [B, N, 1]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]

        # Burgers 残差
        residual = u_t + u.unsqueeze(-1) * u_x - self.nu * u_xx
        return residual.squeeze(-1)   # [B, N]


PI_DeepONet = PIDeepONet1D