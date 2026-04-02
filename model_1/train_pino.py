import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
import os
from HyperPINO import HyperPINO
from pde_static import compute_static_pde_loss

# [1] 설정 및 텐서보드
LOG_DIR = "runs/liver_v14_Dual_Path_Fourier_Scale_Invariant_no_anchor"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

# [2] 데이터 로드
print("🚀 Loading dataset...")

dataset_path = "data/nonlinear/nonlinear_graspers_ind.pt" 
if not os.path.exists(dataset_path):
    dataset_path = "data/nonlinear_graspers_ind.pt" 

if not os.path.exists(dataset_path):
    print(f"❌ Error: {dataset_path} 파일을 찾을 수 없습니다!")
    print("현재 폴더 내 'data' 폴더의 내용:", os.listdir('data') if os.path.exists('data') else "data 폴더 없음")
    exit()

data = torch.load(dataset_path)
inputs_10ch = torch.cat([data["inputs"][0], data["outputs"][0]], dim=-1)
loader = DataLoader(TensorDataset(inputs_10ch), batch_size=8, shuffle=True)

# [3] 모델 및 최적화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HyperPINO(freq=5.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# [4] 하이퍼파라미터 및 스케줄링 변수
epochs = 300
SAMPLE_SIZE = 512    
RAMP_UP_PDE = 50        
RAMP_UP_ANCHOR = 20     
TARGET_PDE_WEIGHT = 3e3 
ANCHOR_WEIGHT = 0       # 앵커 오프 상태
POS_SCALE = 200.0       

print(f"🔥 Training started. View on: tensorboard --logdir={LOG_DIR}")

for epoch in range(epochs):
    model.train()
    
    # 🚀 비율 계산을 위해 가중치가 적용된 로스와 전체 로스를 담을 변수 추가
    epoch_u, epoch_weighted_pde, epoch_weighted_anchor, epoch_total = 0, 0, 0, 0
    
    current_pde_weight = (min(epoch / RAMP_UP_PDE, 1.0)) * TARGET_PDE_WEIGHT
    current_anchor_weight = (min(epoch / RAMP_UP_ANCHOR, 1.0)) * ANCHOR_WEIGHT

    for batch in loader:
        batch_x = batch[0].to(device)
        optimizer.zero_grad()
        
        indices = torch.randperm(batch_x.shape[1])[:SAMPLE_SIZE]
        pos_raw = batch_x[:, indices, 0:3].clone().detach().requires_grad_(True)

        pos_normalized = pos_raw / POS_SCALE
        
        u_pred, mu_pred = model(pos_normalized, batch_x)

        loss_pde = compute_static_pde_loss(pos_raw, u_pred, mu_pred)
        loss_u = torch.mean((u_pred - batch_x[:, indices, 7:10]) ** 2)
        
        target_mus = torch.tensor([0.03, 0.34, 0.69, 1.07], device=device)
        diffs = (mu_pred - target_mus) ** 2
        min_diffs, _ = torch.min(diffs, dim=-1)
        loss_anchor = torch.mean(min_diffs)
        
        # 🚀 가중치가 곱해진 실제 기여분 계산
        weighted_pde = current_pde_weight * loss_pde
        weighted_anchor = current_anchor_weight * loss_anchor
        total_loss = loss_u + weighted_pde + weighted_anchor
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 🚀 에폭별 누적 (순수 U-loss와 가중치 적용된 로스들)
        epoch_u += loss_u.item()
        epoch_weighted_pde += weighted_pde.item()
        epoch_weighted_anchor += weighted_anchor.item()
        epoch_total += total_loss.item()

    # 🚀 평균 로스 및 비율(Ratio) 계산
    avg_u = epoch_u / len(loader)
    avg_weighted_pde = epoch_weighted_pde / len(loader)
    avg_weighted_anchor = epoch_weighted_anchor / len(loader)
    avg_total = epoch_total / len(loader)
    
    # 0으로 나누기 방지
    eps = 1e-8
    ratio_u = avg_u / (avg_total + eps)
    ratio_pde = avg_weighted_pde / (avg_total + eps)
    ratio_anchor = avg_weighted_anchor / (avg_total + eps)

    mu_val = mu_pred.detach().cpu()
    
    # 텐서보드 기록 (기존 로스)
    writer.add_scalar("Loss_Abs/U-Loss", avg_u, epoch)
    writer.add_scalar("Loss_Abs/Weighted-PDE", avg_weighted_pde, epoch)
    writer.add_scalar("Loss_Abs/Weighted-Anchor", avg_weighted_anchor, epoch)
    writer.add_scalar("Loss_Abs/Total-Loss", avg_total, epoch)
    
    # 🚀 텐서보드 기록 (비율 - 0.0 ~ 1.0 사이 값)
    writer.add_scalar("Loss_Ratio/U_Ratio", ratio_u, epoch)
    writer.add_scalar("Loss_Ratio/PDE_Ratio", ratio_pde, epoch)
    writer.add_scalar("Loss_Ratio/Anchor_Ratio", ratio_anchor, epoch)

    writer.add_scalar("Mu/Mean", mu_val.mean(), epoch)
    writer.add_scalar("Mu/Max", mu_val.max(), epoch)
    writer.add_histogram("Mu/Distribution", mu_val, epoch)

    if epoch % 1 == 0:
        # 출력에도 비율을 %로 표시
        print(f"Epoch [{epoch}] Mu: {mu_val.mean():.4f} | U: {avg_u:.2e} ({ratio_u*100:.1f}%) | PDE: {avg_weighted_pde:.2e} ({ratio_pde*100:.1f}%)")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/pino_v14_ep{epoch}.pth")

writer.close()