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
    "log_dir": "runs/linear_v7",
    "checkpoint_dir": "checkpoints",
    "lr_u": 5e-4,
    "lr_mu": 2e-5,  
    "sample_size": 1024,
    "pde_weight": 1e1,         # Phase 1에서 혹시 쓰일 기본값 (현재는 0.0으로 덮어씌워짐)
    "target_pde_weight": 5e2,  # Phase 2에서 사용할 실제 PDE 가중치
    "batch_size": 64,
    "epochs": 2200,
    "phase1_epochs": 30,
    "vis_interval": 10,
    "save_interval": 20
}

# [개선] STN(공간 변환 네트워크) 정규화 함수 추가 
# PointNet의 형태가 찌그러지지 않도록 유지해주는 필수 로스입니다.
def feature_transform_regularizer(trans):
    if trans is None:
        return 0.0
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :].to(trans.device) # [개선] 전역 device 변수 의존성 제거
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(CONFIG["log_dir"])

    print(f"\n[INFO] 텐서보드 실행: tensorboard --logdir={CONFIG['log_dir']}\n")

    # 1. 데이터 로드
    dataset = torch.load(CONFIG["dataset_path"])
    inputs_10ch = torch.cat([dataset["inputs"][0], dataset["outputs"][0]], dim=-1)
    loader = DataLoader(
        TensorDataset(inputs_10ch), 
        batch_size=CONFIG["batch_size"], 
        shuffle=True,
        num_workers=12,       
        pin_memory=True,
        prefetch_factor=4     
    )

    # 2. 모델 및 파라미터 분리
    model = HyperPINO().to(device)

    u_params = list(model.hyper_net_u.parameters()) + list(model.param_proj_u.parameters())
    mu_params = list(model.hyper_net_mu.parameters()) + list(model.param_proj_mu.parameters())

    # [개선] 옵티마이저를 2개로 분리 (모멘텀 충돌 방지)
    # Phase 1과 Phase 2에서 서로의 기울기 상태(Adam Momentum)가 간섭하지 않도록 합니다.
    optimizer_u = optim.Adam(u_params, lr=CONFIG["lr_u"])
    optimizer_mu = optim.Adam(mu_params, lr=CONFIG["lr_mu"])

    # 3. 학습 루프
    for epoch in range(CONFIG["epochs"]):
        
        # ---------------- Phase 설정 ----------------
        if epoch < CONFIG["phase1_epochs"]:
            # Phase 1: u만 학습 (mu 고정)
            for p in u_params: p.requires_grad = True
            for p in mu_params: p.requires_grad = False
            
            pde_weight = 0.0
            calc_pde = False
            active_optimizer = optimizer_u  # u 전용 옵티마이저 활성화
            phase_name = "P1:Train_U "
        else:
            for p in u_params: p.requires_grad = True
            for p in mu_params: p.requires_grad = True
    
            calc_pde = True
            active_optimizer = optimizer_mu
            phase_name = "P2:Infer_Mu"

            # [추가] PDE 가중치 서서히 증가 (예: 에포크 30~50 동안 0.0 -> target_pde_weight로 선형 증가)
            progress = min(1.0, (epoch - CONFIG["phase1_epochs"]) / 20.0) 
            pde_weight = CONFIG["target_pde_weight"] * progress
        # --------------------------------------------

        model.train()
        epoch_metrics = {"u_loss": 0.0, "pde_scaled": 0.0, "reg_loss": 0.0, "total": 0.0}

        for batch in loader:
            batch_data = batch[0].to(device, non_blocking=True)
            active_optimizer.zero_grad() # 현재 Phase에 맞는 옵티마이저 초기화

            # 데이터 샘플링
            indices = torch.randperm(batch_data.shape[1])[:CONFIG["sample_size"]]
            
            # [개선] Phase 1에서는 PDE 계산을 안 하므로 requires_grad_(True)를 끌 수 있어 메모리가 크게 절약됩니다.
            pos_raw = batch_data[:, indices, 0:3].clone().detach()
            if calc_pde:
                pos_raw.requires_grad_(True) 

            pos_normalized = pos_raw / 200.0
            
            # [개선] 사용하지 않는 tool_flag, tool_action 파이썬 변수 제거 (메모리 절약)
            u_gt = batch_data[:, indices, 7:10]
            input_7ch = batch_data[:, indices, 0:7] # 이 안에 Pos(3) + Flag(1) + Action(3) 이 모두 포함됨

            # [개선] 유연한 모델 출력 언패킹 (만약 HyperPINO가 trans 행렬을 반환하도록 수정되었을 경우 대응)
            outputs = model(pos_normalized, input_7ch)
            u_pred, mu_pred = outputs[0], outputs[1]
            
            # STN 정규화 로스 계산 (PointNetfeat가 trans_feat를 반환하도록 수정했다고 가정)
            loss_reg = torch.tensor(0.0).to(device)
            if len(outputs) > 2: 
                # outputs[2], outputs[3] 이 trans_u, trans_mu 라고 가정
                loss_reg += feature_transform_regularizer(outputs[2]) * 0.001
                if len(outputs) > 3:
                    loss_reg += feature_transform_regularizer(outputs[3]) * 0.001

            # Loss 계산
            loss_u = F.mse_loss(u_pred, u_gt)

            if calc_pde:
                raw_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
                loss_pde = raw_pde 
            else:
                loss_pde = torch.tensor(0.0).to(device)

            # 최종 Loss 및 역전파
            total_loss = loss_u + (pde_weight * loss_pde) + loss_reg

            
            total_loss.backward()
            
            if calc_pde:
                torch.nn.utils.clip_grad_norm_(mu_params, max_norm=0.05) 

            active_optimizer.step()

            # 메트릭 기록
            epoch_metrics["u_loss"] += loss_u.item()
            epoch_metrics["pde_scaled"] += (loss_pde.item() * pde_weight)
            epoch_metrics["reg_loss"] += loss_reg.item()
            epoch_metrics["total"] += total_loss.item()

        # 에포크 결산 및 로깅
        num_batches = len(loader)
        avg_u = epoch_metrics["u_loss"] / num_batches
        avg_pde = epoch_metrics["pde_scaled"] / num_batches
        avg_reg = epoch_metrics["reg_loss"] / num_batches
        
        writer.add_scalar("Loss/Total", epoch_metrics["total"]/num_batches, epoch)
        writer.add_scalars("Loss/Split", {"U_MSE": avg_u, "PDE_Scaled": avg_pde, "Reg": avg_reg}, epoch)
        writer.add_scalar("Mu/Mean_Value", mu_pred.mean().item(), epoch)

        if epoch % CONFIG["vis_interval"] == 0:
            log_3d_vis_to_tensorboard(writer, pos_raw, u_gt, u_pred, mu_pred, epoch)
            print(f"   [Snapshot] Visualization logged at epoch {epoch}")

        print(f"Epoch [{epoch:03d}] {phase_name} | Mu: {mu_pred.mean().item():.4f} | "
              f"U_MSE: {avg_u:.4e} | Scaled_PDE: {avg_pde:.4e} | Reg: {avg_reg:.4e}")

        if epoch % CONFIG["save_interval"] == 0:
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_dir']}/pino_v7_ep{epoch}.pth")

    writer.close()

if __name__ == "__main__":
    main()