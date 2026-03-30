# eval_pino.py
import torch
import pyvista as pv
import numpy as np
import os
from KelvinHyperPINO import KelvinHyperPINO

def visualize_visual_haptics(pos_np, mu_pred_np, mu_gt_np=None):
    # 1. PyVista Point Cloud 객체 생성
    cloud = pv.PolyData(pos_np)

    # 2. 예측된 탄성도(mu) 시각화 최적화 (대비 강조)
    # 모델 출력 mu가 너무 균일할 경우를 대비해 0~1 사이로 정규화해서 봅니다.
    mu_min, mu_max = mu_pred_np.min(), mu_pred_np.max()
    print(f"Pred Mu Range: {mu_min:.6f} ~ {mu_max:.6f}")
    
    mu_vis = (mu_pred_np - mu_min) / (mu_max - mu_min + 1e-8)
    cloud["Predicted Elasticity (mu)"] = mu_vis.flatten()

    if mu_gt_np is not None:
        cloud["Ground Truth Elasticity"] = mu_gt_np.flatten()
        p = pv.Plotter(shape=(1, 2), window_size=(1600, 600))

        # 왼쪽: 예측값
        p.subplot(0, 0)
        p.add_mesh(cloud, scalars="Predicted Elasticity (mu)", cmap="jet", point_size=5.0, render_points_as_spheres=True)
        p.add_text(f"PINO Prediction (Range: {mu_min:.4f}-{mu_max:.4f})", font_size=12)

        # 오른쪽: 정답(GT)
        p.subplot(0, 1)
        p.add_mesh(cloud, scalars="Ground Truth Elasticity", cmap="jet", point_size=5.0, render_points_as_spheres=True)
        p.add_text("Ground Truth", font_size=12)
    else:
        p = pv.Plotter()
        p.add_mesh(cloud, scalars="Predicted Elasticity (mu)", cmap="jet", point_size=5.0, render_points_as_spheres=True)
        p.add_text("PINO Predicted Relative Elasticity", font_size=14)

    p.link_views()
    p.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 초기화 및 가중치 로드 (가장 최근 저장된 100 epoch 파일)
    model = KelvinHyperPINO(target_width=128).to(device)
    checkpoint_path = "kelvin_pino_epoch_100.pth" # 만약 파일명이 다르면 확인하세요!
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Check your current directory.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Successfully loaded model from {checkpoint_path}")

    # 2. 데이터 로드 (train_pino.py 로직과 동일하게 10채널 구성)
    dataset_path = "data/individual_graspers_linear.pt"
    data = torch.load(dataset_path)

    # 데이터 구조에 따라 인덱싱 (리스트 형태 대응)
    raw_inputs = data["inputs"][0]  # (B, N, 7)
    u_set = data["outputs"][0]      # (B, N, 3)
    
    # 10채널 합치기
    inputs_10ch = torch.cat([raw_inputs, u_set], dim=-1)
    
    # mu_gt 데이터가 있는지 확인 (없으면 1로 채운 가짜 데이터 생성)
    if "mu_gt" in data:
        targets = data["mu_gt"][0]
    else:
        print("Warning: mu_gt not found in dataset. Showing prediction only.")
        targets = torch.ones((inputs_10ch.shape[0], inputs_10ch.shape[1], 1))

    # 테스트할 샘플 선택 (0번 배치)
    idx = 0
    test_input = inputs_10ch[idx:idx+1].to(device)
    test_target = targets[idx:idx+1]

    # 3. 모델 추론
    print("Running Inference...")
    with torch.no_grad():
        # PINO 모델 구조에 맞춰 분리해서 넣어줌
        pos = test_input[:, :, 0:3]
        u_pred, mu_pred = model(pos, test_input)

    # 4. Numpy 변환 및 시각화
    pos_np = pos[0].cpu().numpy()
    mu_pred_np = mu_pred[0].cpu().numpy()
    mu_gt_np = test_target[0].numpy()

    print("Rendering 3D Visual Haptics...")
    visualize_visual_haptics(pos_np, mu_pred_np, mu_gt_np)

if __name__ == "__main__":
    main()