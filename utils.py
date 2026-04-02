import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def log_visualization_to_tensorboard(writer, pos_raw, u_true, u_pred, mu_pred, epoch):
    """
    TensorBoard에 3D Point Cloud(Mesh)와 3D Scatter Plot(Figure)를 기록합니다.
    (배치 중 첫 번째 샘플[0]만 시각화)
    """
    # 1. 텐서 -> 넘파이 변환 (CPU)
    pos_np = pos_raw[0].detach().cpu().numpy()     # (N, 3)
    u_true_np = u_true[0].detach().cpu().numpy()   # (N, 3)
    u_pred_np = u_pred[0].detach().cpu().numpy()   # (N, 3)
    mu_np = mu_pred[0].detach().cpu().numpy()      # (N, 1)
    
    # 2. 오차 및 정규화
    u_error_np = np.linalg.norm(u_pred_np - u_true_np, axis=1) # (N,)
    
    mu_min, mu_max = mu_np.min(), mu_np.max()
    mu_norm = (mu_np.flatten() - mu_min) / (mu_max - mu_min + 1e-8)
    
    err_min, err_max = u_error_np.min(), u_error_np.max()
    err_norm = (u_error_np - err_min) / (err_max - err_min + 1e-8)

    # ==========================================
    # 방법 A: Matplotlib 3D Figure (한눈에 보기 편함)
    # ==========================================
    fig = plt.figure(figsize=(12, 5))
    
    # Mu Map
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], 
                      c=mu_norm, cmap='jet', s=10)
    ax1.set_title(f"Inferred Mu\nMin: {mu_min:.4f}, Max: {mu_max:.4f}")
    fig.colorbar(sc1, ax1=ax1, shrink=0.5)

    # Error Map
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], 
                      c=err_norm, cmap='magma', s=10)
    ax2.set_title(f"U Error\nMin: {err_min:.2e}, Max: {err_max:.2e}")
    fig.colorbar(sc2, ax2=ax2, shrink=0.5)

    plt.tight_layout()
    # TensorBoard에 이미지로 기록
    writer.add_figure("Eval/3D_Plot", fig, epoch)
    plt.close(fig)

    # ==========================================
    # 방법 B: TensorBoard 3D Mesh (마우스로 돌려볼 수 있음)
    # ==========================================
    # 컬러맵을 RGB(0~255) 텐서로 변환
    cmap_jet = cm.get_cmap('jet')
    cmap_magma = cm.get_cmap('magma')
    
    mu_colors = (cmap_jet(mu_norm)[:, :3] * 255).astype(np.uint8)
    err_colors = (cmap_magma(err_norm)[:, :3] * 255).astype(np.uint8)

    # TensorBoard add_mesh는 (Batch, N, 3) 형태를 요구함
    pos_tensor = torch.tensor(pos_np).unsqueeze(0)             # (1, N, 3)
    mu_color_tensor = torch.tensor(mu_colors).unsqueeze(0)     # (1, N, 3)
    err_color_tensor = torch.tensor(err_colors).unsqueeze(0)   # (1, N, 3)

    writer.add_mesh("Interactive_3D/Mu_Map", vertices=pos_tensor, colors=mu_color_tensor, global_step=epoch)
    writer.add_mesh("Interactive_3D/U_Error", vertices=pos_tensor, colors=err_color_tensor, global_step=epoch)