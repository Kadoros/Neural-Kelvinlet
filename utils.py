import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
def log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch):
    """3D 시각화 (Matplotlib CPU 연산이므로 가끔 호출해야 함)"""
    pos_np = pos_raw[0].detach().cpu().numpy()
    u_gt_np = u_gt[0].detach().cpu().numpy()
    u_pred_np = u_pred[0].detach().cpu().numpy()
    mu_np = mu_pred[0].detach().cpu().numpy().flatten()
    u_err = np.linalg.norm(u_pred_np - u_gt_np, axis=1)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=mu_np, cmap='jet', s=5)
    ax1.set_title(f"Mu Prediction (Mean: {mu_np.mean():.4f})")
    fig.colorbar(sc1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=u_err, cmap='magma', s=5)
    ax2.set_title(f"U Error (Max: {u_err.max():.2e})")
    fig.colorbar(sc2, ax=ax2, shrink=0.5)

    writer.add_figure("Visual/3D_Snapshot", fig, epoch)
    plt.close(fig)
    
    # Mesh 탭용 실시간 3D 데이터
    mu_norm = (mu_np - mu_np.min()) / (mu_np.max() - mu_np.min() + 1e-8)
    colors = (plt.cm.jet(mu_norm)[:, :3] * 255).astype(np.uint8)
    writer.add_mesh("Interactive/Mu_Map", vertices=pos_raw[0:1], 
                    colors=torch.tensor(colors).unsqueeze(0).to(pos_raw.device), global_step=epoch)
    writer.flush()