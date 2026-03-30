import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from KelvinHyperPINO import KelvinHyperPINO
from pde_static import compute_static_pde_loss

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# [2] 데이터 로드
print("Loading Nonlinear dataset for Inverse Elasticity...")
dataset_path = "data/nonlinear/nonlinear_graspers_ind.pt" 
if not os.path.exists(dataset_path):
    dataset_path = "data/nonlinear_graspers_ind.pt" 

data = torch.load(dataset_path)
raw_inputs = data["inputs"][0]
u_nonlinear = data["outputs"][0]

inputs_10ch = torch.cat([raw_inputs, u_nonlinear], dim=-1)
dataset = TensorDataset(inputs_10ch)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# [3] 모델 및 최적화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KelvinHyperPINO(target_width=128).to(device)
# 🚀 학습률 상향 (5e-4)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# [4] 하이퍼파라미터
epochs = 500
SAMPLE_SIZE = 512    
# 🚀 좌표 스케일 정상화 (1.0): 미분값이 작아지는 현상 방지
POS_SCALE = 1.0    

RAMP_UP_EPOCHS = 100
# 🚀 최종 목표 가중치 상향: 1e25 (체급 맞추기)
TARGET_PDE_WEIGHT = 10.0 

print(f"🚀 Starting Inverse PINO training on {device}...")

for epoch in range(epochs):
    model.train()
    epoch_pde_loss = 0
    epoch_u_loss = 0
    
    if epoch < RAMP_UP_EPOCHS:
        current_pde_weight = (epoch / RAMP_UP_EPOCHS) * TARGET_PDE_WEIGHT
    else:
        current_pde_weight = TARGET_PDE_WEIGHT

    track_mu = (epoch % 10 == 0)
    all_mu_preds = [] if track_mu else None

    for batch in loader:
        batch_x = batch[0].to(device)
        optimizer.zero_grad()

        N = batch_x.shape[1]
        indices = torch.randperm(N)[:SAMPLE_SIZE]
        
        pos_raw = batch_x[:, indices, 0:3].clone().detach().requires_grad_(True)
        pos = pos_raw / POS_SCALE 
        
        u_pred, mu_pred = model(pos, batch_x)

        # PDE Loss 계산 (pos_raw 기준)
        loss_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)

        u_true_sampled = batch_x[:, indices, 7:10]
        loss_u = torch.mean((u_pred - u_true_sampled) ** 2)
        
        # mu 페널티 제거 (자유로운 역문제 풀이)
        total_loss = loss_u + (current_pde_weight * loss_pde)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        epoch_pde_loss += loss_pde.item()
        epoch_u_loss += loss_u.item()
        
        if track_mu:
            all_mu_preds.append(mu_pred.detach().cpu())

    if track_mu:
        full_mu = torch.cat(all_mu_preds)
        print(f"\n📊 DEBUG | [전체 mu 분포] Min: {full_mu.min().item():.4f} | Max: {full_mu.max().item():.4f} | Mean: {full_mu.mean().item():.4f}")

    print(f"Epoch [{epoch}/{epochs}] | PDE-Wt: {current_pde_weight:.1e} | U-L: {epoch_u_loss/len(loader):.6f} | PDE-L: {epoch_pde_loss/len(loader):.2e} | Tot: {total_loss.item():.6f}")

final_path = os.path.join(CHECKPOINT_DIR, "pino_inverse_final.pth")
torch.save(model.state_dict(), final_path)