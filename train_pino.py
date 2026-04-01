import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
import os
from HyperPINO import HyperPINO
from pde_static import compute_static_pde_loss

# [1] 설정 및 텐서보드
LOG_DIR = "runs/liver_v13_Dual_Path_Fourier_Scale_Invariant"
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
loader = DataLoader(TensorDataset(inputs_10ch), batch_size=8, shuffle=True)

# [3] 모델 및 최적화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 🚀 푸리에 주파수 5.0 추가 (로스 튀는 것 방지)
model = HyperPINO(freq=5.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# [4] 하이퍼파라미터 및 스케줄링 변수 추가
epochs = 300
SAMPLE_SIZE = 512    
RAMP_UP_PDE = 50        # PDE 가중치를 50에폭 동안 천천히 올림
RAMP_UP_ANCHOR = 20     # 앵커 로스는 20에폭 동안 천천히 올림
TARGET_PDE_WEIGHT = 3e3 
ANCHOR_WEIGHT = 0.1     # 0.69에 너무 빨리 갇히지 않게 비중을 낮춤
POS_SCALE = 200.0       # 좌표 정규화용 스케일

print(f"🔥 Training started. View on: tensorboard --logdir={LOG_DIR}")

for epoch in range(epochs):
    model.train()
    epoch_pde, epoch_u = 0, 0
    
    # 🚀 누락되었던 스케줄링 정의 추가 (에러 해결 핵심)
    current_pde_weight = (min(epoch / RAMP_UP_PDE, 1.0)) * TARGET_PDE_WEIGHT
    # 30에폭까지는 0, 그 이후부터 20에폭에 걸쳐 ANCHOR_WEIGHT까지 서서히 증가
    current_anchor_weight = (min(epoch / RAMP_UP_ANCHOR, 1.0)) * ANCHOR_WEIGHT

    for batch in loader:
        batch_x = batch[0].to(device)
        optimizer.zero_grad()
        
        indices = torch.randperm(batch_x.shape[1])[:SAMPLE_SIZE]
        pos_raw = batch_x[:, indices, 0:3].clone().detach().requires_grad_(True)

        # 좌표 정규화 (푸리에 입력이 너무 커서 로스가 발산하는 것 방지)
        pos_normalized = pos_raw / POS_SCALE
        
        # 모델 추론
        u_pred, mu_pred = model(pos_normalized, batch_x)

        # 로스 계산 (PDE에는 미분을 위해 pos_raw를 넣어야 함)
        loss_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
        loss_u = torch.mean((u_pred - batch_x[:, indices, 7:10]) ** 2)
        
        # Organ Palette Regularization (다중 닻)
        target_mus = torch.tensor([0.03, 0.34, 0.69, 1.07], device=device)
        diffs = (mu_pred - target_mus) ** 2
        min_diffs, _ = torch.min(diffs, dim=-1)
        loss_anchor = torch.mean(min_diffs)
        
        # 최종 로스 조합
        total_loss = loss_u + (current_pde_weight * loss_pde) + (current_anchor_weight * loss_anchor)
        
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
    writer.add_scalar("Loss/Weighted-Anchor", loss_anchor.item() * current_anchor_weight, epoch)
    writer.add_scalar("Mu/Mean", mu_val.mean(), epoch)
    writer.add_scalar("Mu/Max", mu_val.max(), epoch)
    writer.add_histogram("Mu/Distribution", mu_val, epoch)

    if epoch % 1 == 0:
        # 앵커 가중치도 얼마나 들어가는지 확인하기 위해 출력에 추가
        print(f"Epoch [{epoch}] Mu-Avg: {mu_val.mean():.4f} | U-L: {avg_u:.6f} | PDE-Wt: {current_pde_weight:.1e} | Anc-Wt: {current_anchor_weight:.3f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/pino_v13_Dual_Path_Fourier_Scale_Invariant_ep{epoch}.pth")

writer.close()