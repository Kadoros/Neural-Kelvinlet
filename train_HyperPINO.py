import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from HyperPINO import HyperPINO
from energy_static import compute_energy_loss
from utils import log_3d_vis_to_tensorboard

CONFIG = {
    "dataset_path": "data/individual_graspers_linear.pt",
    "log_dir": "runs/linear_v9_energy",
    "checkpoint_dir": "checkpoints",
    "lr_u": 5e-4,
    "lr_mu": 1e-4,  # 에너지 방식은 안정적이므로 mu 학습률을 조금 높임
    "sample_size": 1024,
    "target_energy_weight": 1.0, 
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
            physics_weight, calc_phys = 0.0, False
            active_optimizer = optimizer_u
            phase_name = "P1:Train_U "
        else:
            for p in u_params: p.requires_grad = True
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
                loss_phys_raw = compute_energy_loss(pos_raw, u_pred, mu_pred)
                # 붕괴 방지용 앵커 (평균 1.0 유지)
                loss_anchor = torch.mean((torch.mean(mu_pred) - 1.2)**2) * 1.0
                loss_phys = loss_phys_raw + loss_anchor
            else:
                loss_phys = torch.tensor(0.0).to(device)

            total_loss = loss_u + (physics_weight * loss_phys)
            total_loss.backward()
            
            if calc_phys:
                torch.nn.utils.clip_grad_norm_(mu_params, max_norm=0.1) 

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

        if epoch % CONFIG["vis_interval"] == 0:
            log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch)
            print(f"Epoch [{epoch:03d}] {phase_name} | Mu: {mu_pred.mean().item():.4f} | U_MSE: {avg_u:.4e}")

    writer.close()

if __name__ == "__main__":
    main()