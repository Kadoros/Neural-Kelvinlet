import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from HyperPINO import HyperPINO
from pde_static import compute_static_pde_loss
from utils import log_3d_vis_to_tensorboard

CONFIG = {
    "dataset_path": "data/individual_graspers_linear.pt",
    "log_dir": "runs/linear_v25",
    "checkpoint_dir": "checkpoints",
    "lr_u": 5e-4,
    "lr_mu": 1e-5,  # 에너지 방식은 안정적이므로 mu 학습률을 조금 높임
    "sample_size": 1024,
    "target_energy_weight": 20.0,  
    "batch_size": 64,
    "epochs": 400,
    "phase1_epochs": 30,
    "vis_interval": 10,
    "save_interval": 20
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(CONFIG["log_dir"])

    dataset = torch.load(CONFIG["dataset_path"])
    inputs_10ch = torch.cat([dataset["inputs"][0], dataset["outputs"][0]], dim=-1)
    loader = DataLoader(TensorDataset(inputs_10ch), batch_size=CONFIG["batch_size"], shuffle=True)

    model = HyperPINO().to(device)

    u_params = list(model.hyper_net_u.parameters()) + list(model.param_proj_u.parameters())
    mu_params = list(model.hyper_net_mu.parameters()) + list(model.param_proj_mu.parameters())

    optimizer_u = optim.Adam(u_params, lr=CONFIG["lr_u"])
    optimizer_mu = optim.Adam(mu_params, lr=CONFIG["lr_mu"])

    for epoch in range(CONFIG["epochs"]):
        if epoch < CONFIG["phase1_epochs"]:
            for p in u_params: p.requires_grad = True
            for p in mu_params: p.requires_grad = False
            physics_weight, calc_phys = 1e-2, False
            active_optimizer = optimizer_u
            phase_name = "P1:Train_U "
        else:
            for p in u_params: p.requires_grad = False
            for p in mu_params: p.requires_grad = True
            calc_phys = True
            progress = min(1.0, (epoch - CONFIG["phase1_epochs"]) / 50.0) 
            physics_weight = CONFIG["target_energy_weight"] * progress
            active_optimizer = optimizer_mu
            phase_name = "P2:Infer_Mu"

        model.train()
        metrics = {"u_loss": 0.0, "phys_loss": 0.0, "total": 0.0}

        for batch in loader:
            batch_data = batch[0].to(device)
            active_optimizer.zero_grad()

            indices = torch.randperm(batch_data.shape[1])[:CONFIG["sample_size"]]
            pos_raw = batch_data[:, indices, 0:3].clone().detach()
            if calc_phys: pos_raw.requires_grad_(True) 

            pos_normalized = pos_raw / 200.0
            u_gt = batch_data[:, indices, 7:10]
            input_7ch = batch_data[:, indices, 0:7]

            u_pred, mu_pred = model(pos_normalized, input_7ch)

            loss_u = F.mse_loss(u_pred, u_gt)

            if calc_phys:
                loss_phys = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
                
                # 1. 평균을 1.0 근처로 유지하도록 유도 (Soft Penalty)
                mean_mu = torch.mean(mu_pred)
                loss_mean = F.mse_loss(mean_mu, torch.tensor(1.0).to(device)) * 1.0
                
            
    
                # 최종 Loss 조합
                total_loss = loss_u + (physics_weight * loss_phys) + loss_mean
            else:
                loss_phys = torch.tensor(0.0).to(device)
                total_loss = loss_u + (physics_weight * loss_phys)

            # 여기 있던 total_loss = ... 코드는 삭제합니다.
            
            total_loss.backward()
            
            if calc_phys:
                torch.nn.utils.clip_grad_norm_(mu_params, max_norm=0.01)

            active_optimizer.step()

            metrics["u_loss"] += loss_u.item()
            metrics["phys_loss"] += (loss_phys.item() * physics_weight)
            metrics["total"] += total_loss.item()

        # 로깅
        num_batches = len(loader)
        avg_u = metrics["u_loss"] / num_batches
        avg_phys = metrics["phys_loss"] / num_batches
        
        writer.add_scalars("Loss/Split", {"U_MSE": avg_u, "Phys_Energy": avg_phys}, epoch)
        writer.add_scalar("Mu/Mean", mu_pred.mean().item(), epoch)

        #  [추가] Mu 분포 히스토그램 및 통계
        mu_val = mu_pred.detach().cpu() # GPU 메모리 해제 및 CPU 복사
        writer.add_histogram("Mu/Distribution", mu_val, epoch) # 히스토그램 추가
        writer.add_scalar("Mu/Max", mu_val.max().item(), epoch) # 최대값 추적
        writer.add_scalar("Mu/Min", mu_val.min().item(), epoch) # 최소값 추적

        if epoch % CONFIG["vis_interval"] == 0:
            log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch)
            print(f"Epoch [{epoch:03d}] {phase_name} | Mu: {mu_pred.mean().item():.4f} | U_MSE: {avg_u:.4e}")

        if epoch % CONFIG["save_interval"] == 0 or epoch == CONFIG["epochs"] - 1:
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"model_epoch_v25_{epoch:03d}.pth")
            
            # 저장할 데이터 구성
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_u_state_dict': optimizer_u.state_dict(),
                'optimizer_mu_state_dict': optimizer_mu.state_dict(),
                'loss': total_loss.item(),
                'phase': phase_name
            }
            
            torch.save(save_dict, checkpoint_path)
            print(f"--- Checkpoint saved: {checkpoint_path} ---")

        # 2. [선택 사항] Phase 1이 끝나는 시점에 별도 저장 (U 학습 완료 시점)
        if epoch == CONFIG["phase1_epochs"] - 1:
            p1_path = os.path.join(CONFIG["checkpoint_dir"], "model_phase1_final.pth")
            torch.save(model.state_dict(), p1_path)
            print(f"*** Phase 1 Complete. Model saved to {p1_path} ***")

    # 3. 최종 학습 종료 후 저장
    final_path = os.path.join(CONFIG["checkpoint_dir"], "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model saved to {final_path}")

    writer.close()

if __name__ == "__main__":
    main()