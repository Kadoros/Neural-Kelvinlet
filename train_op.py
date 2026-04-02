import os
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') # 서버 환경용 백엔드 설정
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from OP import SimplePINO
from pde_static import compute_static_pde_loss, get_gradient

# --- Configuration ---
CONFIG = {
    "dataset_path": "data/nonlinear_graspers_ind.pt",
    "log_dir": "runs/liver_pino_v13_optimized", # 경로 버전 관리
    "checkpoint_dir": "checkpoints",
    "batch_size": 64,
    "epochs": 500,
    "phase1_epochs": 50,
    "sample_size": 4096,  # 4096 샘플로 PDE 계산 (속도와 안정성 균형)
    "pos_scale": 200.0,
    "lr_u": 5e-4,
    "lr_mu": 5e-4,
    "target_pde_weight": 1,
    "vis_interval": 20,    # 시각화는 20회에 한 번 (CPU 병목 방지)
    "save_interval": 100,  # 모델 저장은 100회에 한 번
}

# SIREN 특수 초기화
def siren_init(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                num_input = m.weight.size(-1)
                # SIREN 공식에 따른 가중치 초기화
                m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))
                if m.bias is not None:
                    m.bias.zero_()

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(CONFIG["log_dir"])

    print(f"\n[INFO] 텐서보드 실행: tensorboard --logdir=runs\n")

    # --- 데이터 로더 최적화 ---
    dataset = torch.load(CONFIG["dataset_path"])
    inputs_10ch = torch.cat([dataset["inputs"][0], dataset["outputs"][0]], dim=-1)
    loader = DataLoader(
        TensorDataset(inputs_10ch), 
        batch_size=CONFIG["batch_size"], 
        shuffle=True,
        num_workers=12,        # 4 -> 8 (CPU 코어를 더 많이 써서 미리 준비)
        pin_memory=True,
        prefetch_factor=4     # 데이터를 미리 2배 더 가져다 놓기
    )

    model = SimplePINO().to(device)
    siren_init(model)

    optimizer = optim.Adam([
        {'params': model.net_u.parameters(), 'lr': CONFIG["lr_u"]},
        {'params': model.net_mu.parameters(), 'lr': CONFIG["lr_mu"]}
    ])

    for epoch in range(CONFIG["epochs"]):
        # 1. 단계별 학습 설정
        if epoch < CONFIG["phase1_epochs"]:
            # Phase 1: u만 학습 (mu 고정)
            for p in model.net_u.parameters(): p.requires_grad = True
            for p in model.net_mu.parameters(): p.requires_grad = False
            pde_weight = 0.0
            calc_pde = False # PDE 계산 생략으로 속도 향상
            phase_name = "P1:Train_U"
        else:
            # Phase 2: u 고정 (정답 고정), mu만 역산
            for p in model.net_u.parameters(): p.requires_grad = False
            for p in model.net_mu.parameters(): p.requires_grad = True
            pde_weight = CONFIG["target_pde_weight"]
            calc_pde = True
            phase_name = "P2:Infer_Mu"

        model.train()
        epoch_metrics = {"u_scaled": 0.0, "pde_scaled": 0.0, "total": 0.0}

        for batch in loader:
            batch_data = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            indices = torch.randperm(batch_data.shape[1])[:CONFIG["sample_size"]]
            pos_raw = batch_data[:, indices, 0:3].clone().detach().requires_grad_(True)
            tool_flag = batch_data[:, indices, 3:4] 
            tool_action = batch_data[:, 0, 3:7] 
            u_gt = batch_data[:, indices, 7:10]

            u_pred, mu_pred = model(tool_action, pos_raw / CONFIG["pos_scale"])

            # 3단계: Weighted MSE (도구 접촉부 강조) + 스케일링
            loss_weights = 1.0 + (tool_flag * 9.0)
            loss_u = torch.mean(loss_weights * (u_pred - u_gt) ** 2) * 1.0

            # 2단계: PDE Loss (Phase 2일 때만 연산)
            if calc_pde:
                raw_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
                loss_pde = (raw_pde / (torch.mean(mu_pred)**2 + 1e-8)) * 1e2
            else:
                loss_pde = torch.tensor(0.0).to(device)
            
            total_loss = loss_u + pde_weight * loss_pde

            total_loss.backward()
            # Gradient Clipping: SIREN 학습 안정화
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_metrics["u_scaled"] += loss_u.item()
            epoch_metrics["pde_scaled"] += (loss_pde.item() * pde_weight)
            epoch_metrics["total"] += total_loss.item()
            
        # --- 에폭 결과 정리 ---
        num_batches = len(loader)
        avg_u = epoch_metrics["u_scaled"] / num_batches
        avg_pde = epoch_metrics["pde_scaled"] / num_batches
        
        writer.add_scalar("Loss/Total", epoch_metrics["total"]/num_batches, epoch)
        writer.add_scalars("Loss/Split", {"U": avg_u, "PDE": avg_pde}, epoch)
        writer.add_scalar("Mu/Mean_Value", mu_pred.mean().item(), epoch)

        # 50 에폭마다 무거운 시각화 수행
        if epoch % CONFIG["vis_interval"] == 0:
            log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch)
            print(f"   [Snapshot] Visualization logged at epoch {epoch}")

        print(f"Epoch [{epoch:03d}] {phase_name} | Mu: {mu_pred.mean().item():.4f} | "
      f"Scaled_U: {avg_u:.4e} | Scaled_PDE: {avg_pde:.4e}") # .2f -> .4e 로 변경

        if epoch % CONFIG["save_interval"] == 0:
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_dir']}/pino_v10_ep{epoch}.pth")

    writer.close()

if __name__ == "__main__":
    main()