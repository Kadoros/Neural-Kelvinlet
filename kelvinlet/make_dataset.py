import numpy as np
import torch
import pyvista as pv
from data_generator import MeshDisplacementGenerator
import os

# 설정
data_dir = "/home/cvmlserver5/Hyeon/dataset/FEM-simulations/linear_ind"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 1. 원본 메시 읽기 (샘플 파일 하나를 읽어서 기반 데이터 구축)
# 주의: 실제로는 모든 vtk를 순회해야 함. 여기선 구조 잡기 예시
surface_mesh = pv.read(os.path.join(data_dir, "sol_0_221.vtk")) 
points = surface_mesh.points
normals = surface_mesh.point_data['Normals'] # vtk 파일 내 속성 이름 확인 필요
M_to_N_indices = surface_mesh.point_data['Map to 10K'] # vtk 파일 내 속성 이름 확인 필요

# 2. 제너레이터 초기화
generator = MeshDisplacementGenerator(points, normals, M_to_N_indices)

# 3. 데이터 생성 및 저장
print("Generating dataset...")
inputs, outputs = generator.generate_batch(n_displacements=100) # 일단 100개만 테스트

# 텐서로 변환
data = {
    'inputs': [torch.tensor(inputs, dtype=torch.float32)],
    'outputs': [torch.tensor(outputs, dtype=torch.float32)]
}

# 파일 저장
torch.save(data, os.path.join(output_dir, 'individual_graspers_linear.pt'))
print("Done! Saved to data/individual_graspers_linear.pt")