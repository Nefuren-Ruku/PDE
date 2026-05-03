import torch
import torch.nn as nn

class PIDeepONet1D(nn.Module):
    def __init__(self, branch_layers=[256, 128, 128, 64],
                 trunk_layers=[2, 128, 128, 64],
                 activation=nn.Tanh, nu=0.001):
        super().__init__()
        self.nu = nu

        # 分支网络
        layers = []
        for i in range(len(branch_layers) - 1):
            layers.append(nn.Linear(branch_layers[i], branch_layers[i+1]))
            if i < len(branch_layers) - 2:
                layers.append(activation())
        self.branch = nn.Sequential(*layers)

        # 主干网络
        layers = []
        for i in range(len(trunk_layers) - 1):
            layers.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            if i < len(trunk_layers) - 2:
                layers.append(activation())
        self.trunk = nn.Sequential(*layers)

    def forward(self, branch_input, trunk_input):
        """返回预测 u(x,t)"""
        b = self.branch(branch_input)        # [batch, hidden]
        t = self.trunk(trunk_input)          # [batch, n_points, hidden]
        return torch.sum(b.unsqueeze(1) * t, dim=-1)  # [batch, n_points]

    def compute_pde_residual(self, branch_input, trunk_input):
        """
        计算 Burgers 方程残差：u_t + u * u_x - nu * u_xx
        trunk_input: [batch, n_points, 2]，最后一维是 (x, t)
        """
        x = trunk_input[..., 0:1]   # [batch, n_points, 1]
        t = trunk_input[..., 1:2]   # [batch, n_points, 1]

        # 需要计算对 x 和 t 的导数，因此输入需可导
        x.requires_grad_(True)
        t.requires_grad_(True)
        trunk = torch.cat([x, t], dim=-1)

        u = self.forward(branch_input, trunk)  # [batch, n_points]

        # 一阶导：u_x
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        # 二阶导：u_xx
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        # 对时间导数：u_t
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]

        # Burgers 方程残差
        residual = u_t + u.unsqueeze(-1) * u_x - self.nu * u_xx
        return residual.squeeze(-1)  # [batch, n_points]


PI_DeepONet = PIDeepONet1D   # 统一别名