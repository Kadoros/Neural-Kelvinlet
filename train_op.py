import os
import torch
import torch.optim as optim
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
    "log_dir": "runs/liver_pino_v6_decoupled", 
    "checkpoint_dir": "checkpoints",
    "batch_size": 8,
    "epochs": 500,
    "phase1_epochs": 20,  # 20에포크까지는 u만 학습
    "sample_size": 1024,
    "pos_scale": 200.0,
    "lr_u": 1e-4,
    "lr_mu": 5e-4,
    "target_pde_weight": 1e1, # u가 고정된 후에는 큰 가중치가 필요 없음
}

def log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch):
    pos_np = pos_raw[0].detach().cpu().numpy()
    u_gt_np = u_gt[0].detach().cpu().numpy()
    u_pred_np = u_pred[0].detach().cpu().numpy()
    mu_np = mu_pred[0].detach().cpu().numpy().flatten()
    u_err = np.linalg.norm(u_pred_np - u_gt_np, axis=1)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=mu_np, cmap='jet', s=5)
    ax1.set_title(f"Inferred Mu (Mean: {mu_np.mean():.4f})")
    fig.colorbar(sc1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=u_err, cmap='magma', s=5)
    ax2.set_title(f"U Error (Max: {u_err.max():.2e})")
    fig.colorbar(sc2, ax=ax2, shrink=0.5)

    writer.add_figure("Visual/3D_Snapshot", fig, epoch)
    plt.close(fig)
    writer.flush()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(CONFIG["log_dir"])

    dataset = torch.load(CONFIG["dataset_path"])
    inputs_10ch = torch.cat([dataset["inputs"][0], dataset["outputs"][0]], dim=-1)
    loader = DataLoader(TensorDataset(inputs_10ch), batch_size=CONFIG["batch_size"], shuffle=True)

    model = SimplePINO().to(device)

    # 초기 바이어스: exp(-0.7) ~= 0.5 근처에서 시작하도록 설정
    with torch.no_grad():
        model.net_mu.trunk.net[-1].bias.fill_(-0.7)

    optimizer = optim.Adam([
        {'params': model.net_u.parameters(), 'lr': CONFIG["lr_u"]},
        {'params': model.net_mu.parameters(), 'lr': CONFIG["lr_mu"]}
    ])

    for epoch in range(CONFIG["epochs"]):
        # --- [단계별 학습 전략 핵심] ---
        if epoch < CONFIG["phase1_epochs"]:
            # Phase 1: u만 학습 (u_loss 집중)
            for p in model.net_u.parameters(): p.requires_grad = True
            for p in model.net_mu.parameters(): p.requires_grad = False
            pde_weight = 0.0
            phase_name = "Phase1: Train U"
        else:
            # Phase 2: u 고정, mu만 학습 (물리 법칙 강제 적용)
            for p in model.net_u.parameters(): p.requires_grad = False
            for p in model.net_mu.parameters(): p.requires_grad = True
            pde_weight = CONFIG["target_pde_weight"]
            phase_name = "Phase2: Infer Mu"

        model.train()
        epoch_metrics = {"u_scaled": 0.0, "pde_scaled": 0.0, "total": 0.0}

        for batch in loader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()

            indices = torch.randperm(batch_data.shape[1])[:CONFIG["sample_size"]]
            pos_raw = batch_data[:, indices, 0:3].clone().detach().requires_grad_(True)
            tool_action = batch_data[:, 0, 3:7]
            u_gt = batch_data[:, indices, 7:10]

            u_pred, mu_pred = model(tool_action, pos_raw / CONFIG["pos_scale"])

            # 1. Displacement Loss (1e4 스케일링 유지)
            loss_u = torch.mean((u_pred - u_gt) ** 2) * 1e4 

            # 2. PDE Loss (순정 PDE + Mu 정규화)
            raw_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
            # mu가 작아져도 로스가 줄지 않게 mu_pred의 크기로 나눔
            loss_pde = (raw_pde / (torch.mean(mu_pred).detach()**2 + 1e-8)) * 1e11
            
            total_loss = loss_u + pde_weight * loss_pde

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_metrics["u_scaled"] += loss_u.item()
            epoch_metrics["pde_scaled"] += (loss_pde.item() * pde_weight)
            epoch_metrics["total"] += total_loss.item()
            
        num_batches = len(loader)
        avg_u = epoch_metrics["u_scaled"] / num_batches
        avg_pde = epoch_metrics["pde_scaled"] / num_batches
        avg_total = epoch_metrics["total"] / num_batches
        
        writer.add_scalar("Loss/Total", avg_total, epoch)
        writer.add_scalars("Loss/Split", {"U": avg_u, "PDE": avg_pde}, epoch)
        writer.add_scalar("Mu/Mean_Value", mu_pred.mean().item(), epoch)

        if epoch % 1 == 0:
            log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch)

        print(f"Epoch [{epoch:03d}] {phase_name} | Mu: {mu_pred.mean().item():.4f} | Scaled_U: {avg_u:.2f} | Weighted_PDE: {avg_pde:.2f}")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_dir']}/pino_v6_ep{epoch}.pth")

    writer.close()

if __name__ == "__main__":
    main()