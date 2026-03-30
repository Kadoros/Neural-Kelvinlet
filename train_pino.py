# train_pino.py
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from KelvinHyperPINO import KelvinHyperPINO
from pde_static import compute_static_pde_loss

# 1. 데이터 로드
print("Loading dataset...")
dataset_path = "data/individual_graspers_linear.pt"
data = torch.load(dataset_path)

raw_inputs = data["inputs"][0]  # (B, 10400, 7)
u_set = data["outputs"][0]      # (B, 10400, 3)
targets = data.get("mu_gt", [torch.ones((raw_inputs.shape[0], raw_inputs.shape[1], 1))])[0]

inputs_10ch = torch.cat([raw_inputs, u_set], dim=-1)
dataset = TensorDataset(inputs_10ch, targets)
loader = DataLoader(dataset, batch_size=2, shuffle=True) 

# 2. 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KelvinHyperPINO(target_width=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. 학습 설정
epochs = 500
SAMPLE_SIZE = 1024 # [에이스의 팁] 10400개 중 1024개만 무작위 샘플링! 연산 속도 폭발적 증가
print(f"Starting Optimized PINO training on {device}...")

for epoch in range(epochs):
    model.train()
    epoch_pde_loss = 0
    epoch_data_loss = 0

    for batch_x, batch_mu in loader:
        batch_x = batch_x.to(device)
        batch_mu = batch_mu.to(device)
        optimizer.zero_grad()

        # [핵심 수정] 포인트 샘플링 로직
        N = batch_x.shape[1]
        indices = torch.randperm(N)[:SAMPLE_SIZE] # 랜덤하게 점 선택
        
        # 미분용 pos와 전체 정보 분리 (샘플링된 점들에 대해서만)
        pos = batch_x[:, indices, 0:3].clone().detach().requires_grad_(True)
        
        # Forward: HyperNet은 전체를 보고 상황 파악, TargetNet은 샘플링된 점만 mu 예측
        u_pred, mu_pred = model(pos, batch_x) # KelvinHyperPINO가 pos에 대응하도록 수정됨

        # Loss 1: PDE Loss (샘플링된 1024개 점에 대해서만 계산)
        loss_pde = compute_static_pde_loss(pos, u_pred, mu_pred)

        # Loss 2 & 3: Data Loss
        u_true_sampled = batch_x[:, indices, 7:10]
        mu_true_sampled = batch_mu[:, indices, :]
        loss_u = torch.mean((u_pred - u_true_sampled) ** 2)
        loss_mu = torch.mean((mu_pred - mu_true_sampled) ** 2)

        total_loss = loss_mu + loss_u + (0.01 * loss_pde)
        total_loss.backward()
        optimizer.step()

        epoch_pde_loss += loss_pde.item()
        epoch_data_loss += (loss_mu.item() + loss_u.item())

    # [수정] 매 Epoch마다 로그를 찍어서 진행 상황 확인
    print(f"Epoch [{epoch}/{epochs}] | Loss: {epoch_data_loss/len(loader):.6f} | PDE: {epoch_pde_loss/len(loader):.6f}")

    if epoch % 50 == 0:
        torch.save(model.state_dict(), f"kelvin_pino_epoch_{epoch}.pth")

torch.save(model.state_dict(), "kelvin_pino_10ch_final.pth")
print("Training Complete.")