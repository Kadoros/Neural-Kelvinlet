import torch

def get_gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), 
                               create_graph=True, retain_graph=True)[0]

def compute_static_pde_loss(pos, u_pred, mu_pred, nu=0.45):
    # 1. Gradients
    grad_ux = get_gradient(u_pred[:, :, 0:1], pos)
    grad_uy = get_gradient(u_pred[:, :, 1:2], pos)
    grad_uz = get_gradient(u_pred[:, :, 2:3], pos)

    # 2. Strains
    eps_xx, eps_yy, eps_zz = grad_ux[:, :, 0:1], grad_uy[:, :, 1:2], grad_uz[:, :, 2:3]
    eps_xy = 0.5 * (grad_ux[:, :, 1:2] + grad_uy[:, :, 0:1])
    eps_xz = 0.5 * (grad_ux[:, :, 2:3] + grad_uz[:, :, 0:1])
    eps_yz = 0.5 * (grad_uy[:, :, 2:3] + grad_uz[:, :, 1:2])
    tr_eps = eps_xx + eps_yy + eps_zz

    # 3. Lamé & Stress
    lam_pred = (2.0 * mu_pred * nu) / (1.0 - 2.0 * nu)
    sig_xx = 2.0 * mu_pred * eps_xx + lam_pred * tr_eps
    sig_yy = 2.0 * mu_pred * eps_yy + lam_pred * tr_eps
    sig_zz = 2.0 * mu_pred * eps_zz + lam_pred * tr_eps
    sig_xy, sig_xz, sig_yz = 2.0 * mu_pred * eps_xy, 2.0 * mu_pred * eps_xz, 2.0 * mu_pred * eps_yz

    # 4. Divergence (∇·σ)
    div_x = get_gradient(sig_xx, pos)[:, :, 0:1] + get_gradient(sig_xy, pos)[:, :, 1:2] + get_gradient(sig_xz, pos)[:, :, 2:3]
    div_y = get_gradient(sig_xy, pos)[:, :, 0:1] + get_gradient(sig_yy, pos)[:, :, 1:2] + get_gradient(sig_yz, pos)[:, :, 2:3]
    div_z = get_gradient(sig_xz, pos)[:, :, 0:1] + get_gradient(sig_yz, pos)[:, :, 1:2] + get_gradient(sig_zz, pos)[:, :, 2:3]

    # 분자: PDE 잔차 (Divergence의 제곱합)
    pde_residual = div_x**2 + div_y**2 + div_z**2
    
    # 🚀 [수정] 분모: 응력(Stress) 텐서의 크기 (Frobenius norm squared) 추가
    stress_norm = sig_xx**2 + sig_yy**2 + sig_zz**2 + 2.0 * (sig_xy**2 + sig_xz**2 + sig_yz**2)
    
    # 🚀 [수정] Scale-Invariant 수식 적용 (수학적 꼼수 원천 차단)
    # mu가 작아지면 분모인 stress_norm도 작아져서 전체 로스가 오히려 커짐!
    epsilon = 1e-8
    scale_invariant_loss = torch.mean(pde_residual / (stress_norm + epsilon))
    
    return scale_invariant_loss