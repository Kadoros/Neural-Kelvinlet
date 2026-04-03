import torch
import pyvista as pv
import numpy as np
import os
from HyperPINO import HyperPINO

def visualize_inverse_results(pos_np, mu_pred_np, u_error_np):
    cloud = pv.PolyData(pos_np)
    
    # 1. 역산된 mu 시각화 (대비 강조)
    mu_min, mu_max = mu_pred_np.min(), mu_pred_np.max()
    print(f"📊 Inferred Mu Range: {mu_min:.6f} ~ {mu_max:.6f}")
    
    # 시각화를 위한 정규화 (0~1 사이로 변환하여 대비 극대화)
    mu_vis = (mu_pred_np - mu_min) / (mu_max - mu_min + 1e-8)
    cloud["Inferred Elasticity (mu)"] = mu_vis.flatten()
    cloud["Displacement Error"] = u_error_np.flatten()

    p = pv.Plotter(shape=(1, 2), window_size=(1400, 600))
    
    # 왼쪽: 추론된 탄성 계수 맵
    p.subplot(0, 0)
    p.add_mesh(cloud, scalars="Inferred Elasticity (mu)", cmap="jet", point_size=4.0, render_points_as_spheres=True)
    p.add_text(f"Inferred Elasticity Map\nMin:{mu_min:.4f}, Max:{mu_max:.4f}", font_size=10)

    # 오른쪽: 변위 복원 오차
    p.subplot(0, 1)
    p.add_mesh(cloud, scalars="Displacement Error", cmap="magma", point_size=4.0, render_points_as_spheres=True)
    p.add_text("Displacement Reconstruction Error", font_size=10)

    p.link_views()
    p.show()

def main():
    device = torch.device("cpu") 
    model = HyperPINO(target_width=128).to(device)
    
    # [1] 가중치 로드
    checkpoint_path = "checkpoints/pino_v10_ep140.pth" 
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} 가 없습니다.")
        return

    # weights_only=True는 보안 권장사항이며, 최신 PyTorch 경고를 방지합니다.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()

    # [2] 데이터 로드
    data_path = "data/individual_graspers_ind.pt"
    if not os.path.exists(data_path):
        data_path = "data/nonlinear_graspers_ind.pt"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} 가 없습니다.")
        return

    data = torch.load(data_path, weights_only=False)
    
    # inputs(7ch)와 outputs(3ch)를 합쳐서 관리 (총 10ch)
    # inputs[0]의 형태가 [N, 7]이고 outputs[0]이 [N, 3]이라고 가정
    full_data = torch.cat([data["inputs"][0], data["outputs"][0]], dim=-1)

    idx = 0 
    test_sample = full_data[idx:idx+1].to(device) # [1, N, 10]

    # [3] 추론
    POS_SCALE = 200.0
    with torch.no_grad():
        # 좌표 데이터 (0, 1, 2번 채널)
        pos_raw = test_sample[:, :, 0:3]
        pos_normalized = pos_raw / POS_SCALE
        
        # 🔥 중요: 모델 입력으로는 앞쪽 7개 채널만 전달
        model_input = test_sample[:, :, :7] 
        u_pred, mu_pred = model(pos_normalized, model_input)
    
    # [4] 오차 계산
    # 정답 변위(u_true)는 10개 채널 중 마지막 3개 (7, 8, 9번 채널)
    u_true_np = test_sample[0, :, 7:10].cpu().numpy()
    u_pred_np = u_pred[0].cpu().numpy()
    
    # 각 점에서의 L2 Norm (Euclidean Distance) 계산
    u_error = np.linalg.norm(u_pred_np - u_true_np, axis=1)

    # 결과 시각화
    visualize_inverse_results(pos_raw[0].cpu().numpy(), mu_pred[0].cpu().numpy(), u_error)

if __name__ == "__main__":
    main()