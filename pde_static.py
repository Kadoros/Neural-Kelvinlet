import torch

def get_gradient(y, x):
    return torch.autograd.grad(
        y, x, 
        grad_outputs=torch.ones_like(y), 
        create_graph=True, 
        retain_graph=True
    )[0]

def compute_static_pde_loss(pos, u_pred, mu_pred, nu=0.3):
    # 1. Gradients (변위의 공간 미분)
    grad_ux = get_gradient(u_pred[:, :, 0:1], pos)
    grad_uy = get_gradient(u_pred[:, :, 1:2], pos)
    grad_uz = get_gradient(u_pred[:, :, 2:3], pos)

    # 2. Strains (변형률 계산)
    eps_xx, eps_yy, eps_zz = grad_ux[:, :, 0:1], grad_uy[:, :, 1:2], grad_uz[:, :, 2:3]
    eps_xy = 0.5 * (grad_ux[:, :, 1:2] + grad_uy[:, :, 0:1])
    eps_xz = 0.5 * (grad_ux[:, :, 2:3] + grad_uz[:, :, 0:1])
    eps_yz = 0.5 * (grad_uy[:, :, 2:3] + grad_uz[:, :, 1:2])
    tr_eps = eps_xx + eps_yy + eps_zz

    # 3. Stress (응력 계산 - Lamé parameters 활용)
    lam_pred = (2.0 * mu_pred * nu) / (1.0 - 2.0 * nu)
    sig_xx = 2.0 * mu_pred * eps_xx + lam_pred * tr_eps
    sig_yy = 2.0 * mu_pred * eps_yy + lam_pred * tr_eps
    sig_zz = 2.0 * mu_pred * eps_zz + lam_pred * tr_eps
    sig_xy, sig_xz, sig_yz = 2.0 * mu_pred * eps_xy, 2.0 * mu_pred * eps_xz, 2.0 * mu_pred * eps_yz

    # 4. Divergence (평형 방정식: ∇·σ = 0)
    div_x = get_gradient(sig_xx, pos)[:, :, 0:1] + get_gradient(sig_xy, pos)[:, :, 1:2] + get_gradient(sig_xz, pos)[:, :, 2:3]
    div_y = get_gradient(sig_xy, pos)[:, :, 0:1] + get_gradient(sig_yy, pos)[:, :, 1:2] + get_gradient(sig_yz, pos)[:, :, 2:3]
    div_z = get_gradient(sig_xz, pos)[:, :, 0:1] + get_gradient(sig_yz, pos)[:, :, 1:2] + get_gradient(sig_zz, pos)[:, :, 2:3]

    res = torch.mean(div_x**2 + div_y**2 + div_z**2)
    
    mu_scale = torch.mean(mu_pred**2) + 0.1 
    return res / mu_scale

def compute_energy_loss(pos, u_pred, mu_pred, nu=0.3):
    """
    1차 미분(Strain)만 사용하는 탄성 에너지 최소화 로스
    W = μ * (ε:ε) + 0.5 * λ * (tr(ε))^2
    """
    # 1. Jacobian 계산 (∂u_i / ∂x_j) -> 1차 미분만 수행
    du_dx = get_gradient(u_pred[:, :, 0:1], pos) # [B, N, 3]
    du_dy = get_gradient(u_pred[:, :, 1:2], pos) # [B, N, 3]
    du_dz = get_gradient(u_pred[:, :, 2:3], pos) # [B, N, 3]

    # 2. 변형률 텐서(ε) 성분 추출
    eps_xx = du_dx[:, :, 0:1]
    eps_yy = du_dy[:, :, 1:2]
    eps_zz = du_dz[:, :, 2:3]
    
    eps_xy = 0.5 * (du_dx[:, :, 1:2] + du_dy[:, :, 0:1])
    eps_xz = 0.5 * (du_dx[:, :, 2:3] + du_dz[:, :, 0:1])
    eps_yz = 0.5 * (du_dy[:, :, 2:3] + du_dz[:, :, 1:2])

    # 3. 물리 파라미터 계산 (Lamé λ)
    lam_pred = (2.0 * mu_pred * nu) / (1.0 - 2.0 * nu)
    
    # 4. 탄성 에너지 밀도 W 계산
    trace_eps = eps_xx + eps_yy + eps_zz
    # ε:ε (Frobenius norm squared)
    eps_squared_norm = eps_xx**2 + eps_yy**2 + eps_zz**2 + \
                       2.0 * (eps_xy**2 + eps_xz**2 + eps_yz**2)

    # 에너지 밀도 수식
    energy_density = mu_pred * eps_squared_norm + 0.5 * lam_pred * (trace_eps**2)

    # 5. 전체 에너지 평균
    total_energy = torch.mean(energy_density)
    
    # [방어막] mu가 0으로 수렴하여 로스를 없애는 꼼수 방지
    mu_scale = torch.mean(mu_pred**2) + 0.1 
    
    return total_energy / mu_scale