# pde_static.py
import torch

def get_gradient(y, x):
    """y를 x로 미분한 값을 리턴 (B, N, D)"""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]

def compute_static_pde_loss(pos, u_pred, mu_pred, nu=0.45):
    """
    정적 Navier-Cauchy 평형 방정식 손실 계산
    식: μ∇²u + (μ / (1 - 2ν)) ∇(∇·u) = 0
    """
    ux = u_pred[:, :, 0:1]
    uy = u_pred[:, :, 1:2]
    uz = u_pred[:, :, 2:3]

    # 1. Displacement Gradient (Jacobian) ∇u
    grad_ux = get_gradient(ux, pos)
    grad_uy = get_gradient(uy, pos)
    grad_uz = get_gradient(uz, pos)

    # 2. Divergence (∇·u) 
    div_u = grad_ux[:, :, 0:1] + grad_uy[:, :, 1:2] + grad_uz[:, :, 2:3]

    # 3. Laplacian ∇²u 계산 (ux, uy, uz 각각의 2차 미분)
    lap_ux = (
        get_gradient(grad_ux[:, :, 0:1], pos)[:, :, 0:1]
        + get_gradient(grad_ux[:, :, 1:2], pos)[:, :, 1:2]
        + get_gradient(grad_ux[:, :, 2:3], pos)[:, :, 2:3]
    )
    lap_uy = (
        get_gradient(grad_uy[:, :, 0:1], pos)[:, :, 0:1]
        + get_gradient(grad_uy[:, :, 1:2], pos)[:, :, 1:2]
        + get_gradient(grad_uy[:, :, 2:3], pos)[:, :, 2:3]
    )
    lap_uz = (
        get_gradient(grad_uz[:, :, 0:1], pos)[:, :, 0:1]
        + get_gradient(grad_uz[:, :, 1:2], pos)[:, :, 1:2]
        + get_gradient(grad_uz[:, :, 2:3], pos)[:, :, 2:3]
    )
    laplacian_u = torch.cat([lap_ux, lap_uy, lap_uz], dim=-1)

    # 4. ∇(∇·u) 계산 (Divergence의 Gradient)
    grad_div_u = get_gradient(div_u, pos)

    # 5. 방정식 조립
    coeff = 1.0 / (1.0 - 2.0 * nu)
    pde_res = mu_pred * laplacian_u + (mu_pred * coeff) * grad_div_u

    return torch.mean(pde_res**2)