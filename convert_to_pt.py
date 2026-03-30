import pyvista as pv
import torch
import os
import glob
import numpy as np
from tqdm import tqdm

# 경로 설정
data_dir = "/home/cvmlserver5/Hyeon/dataset/FEM-simulations/linear_ind"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

vtk_files = glob.glob(os.path.join(data_dir, "*.vtk"))
print(f"Converting {len(vtk_files)} files into 10-channel compatible format...")

all_inputs = []
all_outputs = []

for f in tqdm(vtk_files):
    mesh = pv.read(f)
    
    # 1. 원본 좌표 추출 (3채널: x, y, z)
    coords = mesh.points
    
    # 2. 툴 데이터 추출
    flag = mesh.point_data['Tool flag']
    tool_disp = mesh.point_data['Tool displacement']
    
    # 3. 정답 데이터 추출 (Output displacement)
    out_disp = mesh.point_data['Output displacement']
    
    # flag 차원 맞추기 (N,) -> (N, 1)
    if flag.ndim == 1:
        flag = flag[:, np.newaxis]
    
    # --- 채널 구성 전략 ---
    # inputs_for_pt: [좌표(3) + 플래그(1) + 툴변위(3)] = 7채널
    # 나중에 preprocess.py에서 여기에 u_set(3)을 cat하여 총 10채널이 됨
    inp = np.hstack([coords, flag, tool_disp]) # (N, 7)
    
    all_inputs.append(torch.tensor(inp, dtype=torch.float32))
    all_outputs.append(torch.tensor(out_disp, dtype=torch.float32))

# 텐서 스택 (Batch, N, Channel)
inputs_tensor = torch.stack(all_inputs, dim=0)
outputs_tensor = torch.stack(all_outputs, dim=0)

# 1. individual_graspers_linear.pt 저장
# train.py의 preprocess 함수 내 torch.cat(data['inputs'], dim=0) 대응을 위해 리스트로 감쌈
data = {
    'inputs': [inputs_tensor],
    'outputs': [outputs_tensor]
}
torch.save(data, os.path.join(output_dir, 'individual_graspers_linear.pt'))

# 2. u_set_individual_linear.pth 저장 (3채널)
# 이것이 나중에 inputs의 7채널 뒤에 붙어서 10채널이 됨
torch.save(outputs_tensor, os.path.join(output_dir, 'u_set_individual_linear.pth'))

print("\n[완료] 데이터셋이 10채널 환경에 맞게 재구성되었습니다.")
print(f"Input shape: {inputs_tensor.shape} (7 channels)")
print(f"Output shape: {outputs_tensor.shape} (3 channels)")
print("최종적으로 모델에는 (7 + 3) = 10 채널이 입력됩니다.")