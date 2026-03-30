import pyvista as pv
import os
import numpy as np

# 베이스 경로
base_path = "/home/cvmlserver5/Hyeon/dataset/FEM-simulations"
folders = ["linear_ind", "linear_combined", "nonlinear_ind", "nonlinear_combined"]

print("\n" + "="*70)
print(f"{'Folder Name':<20} | {'Point Data Keys'}")
print("-"*70)

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"{folder:<20} | ❌ 폴더를 찾을 수 없음")
        continue
        
    files = [f for f in os.listdir(folder_path) if f.endswith('.vtk')]
    if not files:
        print(f"{folder:<20} | ❌ 파일 없음")
        continue
        
    # 첫 번째 파일 읽기
    sample_file = os.path.join(folder_path, files[0])
    mesh = pv.read(sample_file)
    keys = list(mesh.point_data.keys())
    
    # 출력
    print(f"{folder:<20} | {keys}")
    
    # 만약 특별한 키가 보인다면 구체적으로 분석
    for key in keys:
        if any(word in key.lower() for word in ['label', 'id', 'material', 'organ', 'type']):
            unique_vals = np.unique(mesh.point_data[key])
            print(f"   ㄴ 🚩 발견! [{key}]: Unique Values = {unique_vals}")

print("="*70 + "\n")