import pyvista as pv
import torch
import os
import glob
import numpy as np
from tqdm import tqdm

# [1] 경로 설정: 선형 데이터셋 타겟
data_dir = "/home/cvmlserver5/Hyeon/dataset/FEM-simulations/linear_ind"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

vtk_files = glob.glob(os.path.join(data_dir, "*.vtk"))
print(f"🚀 Converting {len(vtk_files)} Linear files into 10-channel format...")

all_inputs = []
all_outputs = []

for f in tqdm(vtk_files):
    mesh = pv.read(f)
    
    # 1. 원본 좌표 추출 (3채널: x, y, z)
    coords = mesh.points
    
    # 2. 툴 데이터 추출
    flag = mesh.point_data['Tool flag']
    tool_disp = mesh.point_data['Tool displacement']
    
    # 3. [핵심] 비선형 정답 데이터 추출 (NonLinear Output displacement)
    # 선형 데이터(Output displacement)보다 장기별 탄성 특성이 훨씬 잘 반영되어 있음
    if 'NonLinear Output displacement' in mesh.point_data:
        out_disp = mesh.point_data['NonLinear Output displacement']
    else:
        # 혹시 키가 다를 경우를 대비한 예외 처리
        out_disp = mesh.point_data['Output displacement']
    
    # flag 차원 맞추기 (N,) -> (N, 1)
    if flag.ndim == 1:
        flag = flag[:, np.newaxis]
    
    # --- 채널 구성 전략 (Total 10ch) ---
    # Input Part 1 (7ch): [좌표(3) + 플래그(1) + 툴변위(3)]
    inp = np.hstack([coords, flag, tool_disp]) # (N, 7)
    
    all_inputs.append(torch.tensor(inp, dtype=torch.float32))
    all_outputs.append(torch.tensor(out_disp, dtype=torch.float32))

# 텐서 스택 (Batch, N, Channel)
inputs_tensor = torch.stack(all_inputs, dim=0)
outputs_tensor = torch.stack(all_outputs, dim=0)

# [4] 파일 저장: 기존 파일과 섞이지 않도록 이름을 'nonlinear'로 변경
# (나중에 train_pino.py에서도 이 파일명을 가리키도록 수정하세요)
data_save_name = 'nonlinear_graspers_ind.pt'
uset_save_name = 'u_set_nonlinear_ind.pth'

data = {
    'inputs': [inputs_tensor],
    'outputs': [outputs_tensor]
}

torch.save(data, os.path.join(output_dir, data_save_name))
torch.save(outputs_tensor, os.path.join(output_dir, uset_save_name))

print("\n" + "="*50)
print(f"✅ [완료] 비선형 데이터셋 재구성 성공!")
print(f"📍 저장 경로: {output_dir}")
print(f"📦 데이터 파일: {data_save_name}")
print(f"📦 uset 파일: {uset_save_name}")
print(f"📊 최종 입력 데이터 형태: (Batch, N, 7) + (Batch, N, 3) = 10 Channels")
print("="*50)