# train_pino.py
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from KelvinHyperPINO import KelvinHyperPINO
from pde_static import compute_static_pde_loss

# 1. 데이터 로드 (10채널 구조: Coords(3)+Flag(1)+ToolDisp(3)+u_set(3))
print("Loading 10-channel dataset...")
dataset_path = "data/individual_graspers_linear.pt"
data = torch.load(dataset_path)

inputs = data["inputs"]  # (B, N, 10)
targets = data["mu_gt"]  # (B, N, 1)

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2. 모델 및 최적화 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KelvinHyperPINO(target_width=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. 학습 루프
epochs = 500
print(f"Starting 10-channel PINO training on {device}...")

for epoch in range(epochs):
    epoch_pde_loss = 0
    epoch_data_loss = 0

    for batch_x, batch_mu in loader:
        batch_x = batch_x.to(device)
        batch_mu = batch_mu.to(device)

        optimizer.zero_grad()

        # [수정된 핵심 1] 모델에 들어가기 전에 미분용 pos를 먼저 빼서 추적(requires_grad)을 켭니다!!
        pos = batch_x[:, :, 0:3].clone().detach().requires_grad_(True)

        # [수정된 핵심 2] 모델에 미분용 pos와 전체 정보 batch_x를 같이 넘겨줍니다!!
        # (주의: KelvinHyperPINO의 forward 함수가 def forward(self, pos, full_input): 으로 되어있어야 합니다)
        u_pred, mu_pred = model(pos, batch_x)

        # [수정된 핵심 3] 이제 u_pred는 pos로부터 연산되었으므로 완벽하게 미분이 가능합니다.
        loss_pde = compute_static_pde_loss(pos, u_pred, mu_pred)

        # [추가된 핵심 4] u_pred가 농땡이 치지 못하게 강제하는 Loss (매우 중요!)
        # 데이터의 7~10번 채널에 있는 Kelvinlet 변위(u_set)를 정답으로 사용합니다.
        u_true = batch_x[:, :, 7:10]
        loss_u = torch.mean((u_pred - u_true) ** 2)

        # Loss 2: Data Loss (정답 mu와의 오차)
        loss_mu = torch.mean((mu_pred - batch_mu) ** 2)

        # 합산 Loss (u_loss를 반드시 더해줘야 합니다!)
        total_loss = loss_mu + loss_u + (0.01 * loss_pde)

        total_loss.backward()
        optimizer.step()

        epoch_pde_loss += loss_pde.item()
        epoch_data_loss += loss_data.item()

    if epoch % 10 == 0:
        print(
            f"Epoch [{epoch}/{epochs}] | Data Loss: {epoch_data_loss/len(loader):.6f} | PDE Loss: {epoch_pde_loss/len(loader):.6f}"
        )

# 4. 모델 저장
torch.save(model.state_dict(), "kelvin_pino_10ch_best.pth")
print("Training Complete. Model saved.")
