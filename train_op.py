import os
import torch
import torch.optim as optim

# [중요] GUI가 없는 서버 환경을 위해 Matplotlib 백엔드 고정 (import plt 보다 위에 있어야 함)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from OP import SimplePINO
from pde_static import compute_static_pde_loss, get_gradient

# --- Configuration ---
CONFIG = {
    "dataset_path": "data/nonlinear_graspers_ind.pt",
    "log_dir": "runs/liver_pino_v3", 
    "checkpoint_dir": "checkpoints",
    "batch_size": 8,
    "epochs": 300,
    "sample_size": 512,
    "pos_scale": 200.0,
    "lr_u": 1e-5,          # 변위 네트워크는 정밀하게 다듬기만 함
    "lr_mu": 5e-4,         # 강성 네트워크는 더 과감하게 탐색하게 함 (50배 차이)
    "target_pde_weight": 1e10, # PDE가 무시되지 않도록 대폭 상향
    "ramp_up_pde": 50,
}

def log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch):
    """3D Scatter Plot(이미지)과 Interactive Mesh(3D 객체)를 기록"""
    pos_np = pos_raw[0].detach().cpu().numpy()
    u_gt_np = u_gt[0].detach().cpu().numpy()
    u_pred_np = u_pred[0].detach().cpu().numpy()
    mu_np = mu_pred[0].detach().cpu().numpy().flatten()
    u_err = np.linalg.norm(u_pred_np - u_gt_np, axis=1)

    # 1. 3D Scatter Figure 생성 (IMAGES 탭으로 전송)
    fig = plt.figure(figsize=(15, 5))
    
    # Inferred Mu Map
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=mu_np, cmap='jet', s=5)
    ax1.set_title(f"Inferred Mu (Mean: {mu_np.mean():.4f})")
    fig.colorbar(sc1, ax=ax1, shrink=0.5)

    # Error Map
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=u_err, cmap='magma', s=5)
    ax2.set_title(f"Reconstruction Error (Max: {u_err.max():.2e})")
    fig.colorbar(sc2, ax=ax2, shrink=0.5)

    writer.add_figure("Visual/3D_Snapshot", fig, epoch)
    plt.close(fig)

    # 2. Interactive Mesh 기록 (MESH 탭으로 전송 - 색상 추가)
    # Mu 값을 Jet 컬러맵으로 변환하여 색상 입히기
    mu_min, mu_max = mu_np.min(), mu_np.max()
    mu_norm = (mu_np - mu_min) / (mu_max - mu_min + 1e-8)
    # matplotlib 컬러맵을 RGB(0-255)로 변환
    colors = (plt.cm.jet(mu_norm)[:, :3] * 255).astype(np.uint8)
    color_tensor = torch.tensor(colors).unsqueeze(0).to(pos_raw.device) # (1, N, 3)

    writer.add_mesh("Interactive/Mu_Map", 
                    vertices=pos_raw[0:1], 
                    colors=color_tensor, 
                    global_step=epoch)

    # [중요] 버퍼에 쌓인 데이터를 즉시 파일로 저장
    writer.flush()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(CONFIG["log_dir"])

    # 텐서보드 실행 명령 안내
    print("\n" + "="*70)
    print(f" 텐서보드 실행용 주소: \n tensorboard --logdir=runs --port=6006")
    print("="*70 + "\n")

    # --- Data Loading ---
    dataset = torch.load(CONFIG["dataset_path"])
    inputs_10ch = torch.cat([dataset["inputs"][0], dataset["outputs"][0]], dim=-1)
    loader = DataLoader(TensorDataset(inputs_10ch), batch_size=CONFIG["batch_size"], shuffle=True)

    model = SimplePINO().to(device)
    optimizer = optim.Adam([
        {'params': model.net_u.parameters(), 'lr': CONFIG["lr_u"]},
        {'params': model.net_mu.parameters(), 'lr': CONFIG["lr_mu"]}
    ])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_metrics = {"u_loss": 0.0, "pde_loss": 0.0, "total": 0.0, "grad_u_sum": 0.0}
        pde_weight = min(epoch / CONFIG["ramp_up_pde"], 1.0) * CONFIG["target_pde_weight"]

        for batch in loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()

            indices = torch.randperm(batch_data.shape[1])[:CONFIG["sample_size"]]
            pos_raw = batch_data[:, indices, 0:3].clone().detach().requires_grad_(True)
            tool_action = batch_data[:, 0, 3:7]
            u_gt = batch_data[:, indices, 7:10]

            u_pred, mu_pred = model(tool_action, pos_raw / CONFIG["pos_scale"])

            # 로깅용 미분 강도 계산
            du_dpos_val = get_gradient(u_pred, pos_raw).detach().abs().mean().item()

            loss_u = torch.mean((u_pred - u_gt) ** 2)
            loss_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
            
            weighted_pde = pde_weight * loss_pde
            total_loss = loss_u + weighted_pde

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_metrics["u_loss"] += loss_u.item()
            epoch_metrics["pde_loss"] += weighted_pde.item()
            epoch_metrics["total"] += total_loss.item()
            epoch_metrics["grad_u_sum"] += du_dpos_val
            
        # --- 통계 및 로깅 ---
        num_batches = len(loader)
        avg_u = epoch_metrics["u_loss"] / num_batches
        avg_pde = epoch_metrics["pde_loss"] / num_batches
        avg_total = epoch_metrics["total"] / num_batches
        avg_grad_u = epoch_metrics["grad_u_sum"] / num_batches
        
        u_ratio = avg_u / (avg_total + 1e-8)
        pde_ratio = avg_pde / (avg_total + 1e-8)

        writer.add_scalar("Loss/1_Total", avg_total, epoch)
        writer.add_scalars("Loss/Components", {"U_Data": avg_u, "Weighted_PDE": avg_pde}, epoch)
        writer.add_scalars("Ratio/Comparison", {"U_Data": u_ratio, "PDE": pde_ratio}, epoch)
        writer.add_scalar("Mu/Mean_Value", mu_pred.mean().item(), epoch)
        writer.add_scalar("Physics/Grad_U_Magnitude", avg_grad_u, epoch)

        # 시각화 (테스트를 위해 매 에폭마다 기록)
        if epoch % 1 == 0:
            # 히스토그램 (HISTOGRAMS 탭)
            writer.add_histogram("Dist/Mu_Prediction", mu_pred.detach().cpu(), epoch)
            # 3D 스냅샷 및 메쉬 (IMAGES, MESH 탭)
            log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch)
            
            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        print(f"Epoch [{epoch:03d}] Mu: {mu_pred.mean().item():.4f} | Total: {avg_total:.2e} | U: {u_ratio*100:4.1f}% | PDE: {pde_ratio*100:4.1f}%")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_dir']}/pino_v2_ep{epoch}.pth")

    writer.close()

if __name__ == "__main__":
    main()
