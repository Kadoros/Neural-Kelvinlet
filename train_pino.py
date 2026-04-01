import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
import os
from HyperPINO import HyperPINO
from pde_static import compute_static_pde_loss

# [1] 설정 및 텐서보드
LOG_DIR = "runs/liver_v10_Organ_Palette_Anatomical_Prior"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

# [2] 데이터 로드
print("🚀 Loading dataset...")

# 🚀 경로가 두 군데 중 어디에 있는지 체크합니다.
dataset_path = "data/nonlinear/nonlinear_graspers_ind.pt" 
if not os.path.exists(dataset_path):
    dataset_path = "data/nonlinear_graspers_ind.pt" 

if not os.path.exists(dataset_path):
    # 만약 둘 다 없으면 현재 폴더의 파일들을 출력해서 확인을 돕습니다.
    print(f"❌ Error: {dataset_path} 파일을 찾을 수 없습니다!")
    print("현재 폴더 내 'data' 폴더의 내용:", os.listdir('data') if os.path.exists('data') else "data 폴더 없음")
    exit()

data = torch.load(dataset_path)
inputs_10ch = torch.cat([data["inputs"][0], data["outputs"][0]], dim=-1)
# loader = DataLoader(TensorDataset(inputs_10ch), batch_size=4, shuffle=True)
loader = DataLoader(TensorDataset(inputs_10ch), batch_size=8, shuffle=True)

# [3] 모델 및 최적화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HyperPINO().to(device)
# optimizer = optim.Adam(model.parameters(), lr=2e-4)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# [4] 하이퍼파라미터 (참교육 모드)
# epochs = 500
# SAMPLE_SIZE = 512    
# RAMP_UP_EPOCHS = 30
# TARGET_PDE_WEIGHT = 3e5 # ⚡ 채찍질 강화

epochs = 300
SAMPLE_SIZE = 512    
RAMP_UP_EPOCHS = 10
TARGET_PDE_WEIGHT = 3e3 # ⚡ 채찍질 강화

print(f"🔥 Training started. View on: tensorboard --logdir={LOG_DIR}")

for epoch in range(epochs):
    model.train()
    epoch_pde, epoch_u = 0, 0
    current_pde_weight = (min(epoch / RAMP_UP_EPOCHS, 1.0)) * TARGET_PDE_WEIGHT

    for batch in loader:
        batch_x = batch[0].to(device)
        optimizer.zero_grad()
        
        indices = torch.randperm(batch_x.shape[1])[:SAMPLE_SIZE]
        pos_raw = batch_x[:, indices, 0:3].clone().detach().requires_grad_(True)
        
        u_pred, mu_pred = model(pos_raw, batch_x)

        loss_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
        loss_u = torch.mean((u_pred - batch_x[:, indices, 7:10]) ** 2)
        
        # --------------------------------------------------
        # 논문 기반 다중 닻 (Organ Palette Regularization)
        # --------------------------------------------------
        # 논문 에 명시된 4가지 주요 물성치
        target_mus = torch.tensor([0.03, 0.34, 0.69, 1.07], device=device)
        
        # 각 점의 예측값이 4가지 정답 중 하나에만 가까워도 OK!
        # (mu_pred: [512, 1] -> target_mus: [4]) -> diffs: [512, 4]
        diffs = (mu_pred - target_mus) ** 2
        min_diffs, _ = torch.min(diffs, dim=-1)
        
        # 평균값이 너무 튀지 않게 잡아주면서도, 4가지 옵션 중 하나를 선택할 자유를 줌
        loss_anchor = torch.mean(min_diffs)
        
        # 가중치는 1.0~10.0 정도로 조절 
        total_loss = loss_u + (current_pde_weight * loss_pde) + (1.0 * loss_anchor)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_pde += loss_pde.item()
        epoch_u += loss_u.item()

    # 🚀 텐서보드 기록
    avg_u = epoch_u / len(loader)
    avg_pde = epoch_pde / len(loader)
    mu_val = mu_pred.detach().cpu()
    
    writer.add_scalar("Loss/U-Loss", avg_u, epoch)
    writer.add_scalar("Loss/PDE-Loss", avg_pde, epoch)
    writer.add_scalar("Loss/Weighted-PDE", avg_pde * current_pde_weight, epoch)
    writer.add_scalar("Mu/Mean", mu_val.mean(), epoch)
    writer.add_scalar("Mu/Max", mu_val.max(), epoch)
    writer.add_histogram("Mu/Distribution", mu_val, epoch)

    if epoch % 1 == 0:
        print(f"Epoch [{epoch}] Mu-Avg: {mu_val.mean():.4f} | U-L: {avg_u:.6f} | PDE-Wt: {current_pde_weight:.1e}")

    if epoch % 10 == 0:  # 50에서 10으로 변경
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/pino_v10_Organ_Palette_Anatomical_Prior_ep{epoch}.pth")

writer.close()