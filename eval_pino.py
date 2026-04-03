# eval_pino.py
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
    
    mu_vis = (mu_pred_np - mu_min) / (mu_max - mu_min + 1e-8)
    cloud["Inferred Elasticity (mu)"] = mu_vis.flatten()
    cloud["Displacement Error"] = u_error_np.flatten()

    p = pv.Plotter(shape=(1, 2), window_size=(1400, 600))
    
    p.subplot(0, 0)
    p.add_mesh(cloud, scalars="Inferred Elasticity (mu)", cmap="jet", point_size=4.0, render_points_as_spheres=True)
    p.add_text(f"Inferred Elasticity Map\nMin:{mu_min:.4f}, Max:{mu_max:.4f}", font_size=10)

    p.subplot(0, 1)
    p.add_mesh(cloud, scalars="Displacement Error", cmap="magma", point_size=4.0, render_points_as_spheres=True)
    p.add_text("Displacement Reconstruction Error", font_size=10)

    p.link_views()
    p.show()

def main():
    device = torch.device("cpu") 
    model = HyperPINO(target_width=128).to(device)
    
    # [1] 가중치 로드
    checkpoint_path = "checkpoints/pino_v10_epoch_100.pth" 
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: {checkpoint_path} 가 없습니다.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # [2] 데이터 로드
    data_path = "data/nonlinear/nonlinear_graspers_ind.pt"
    if not os.path.exists(data_path):
        data_path = "data/nonlinear_graspers_ind.pt"
    
    data = torch.load(data_path)
    inputs_10ch = torch.cat([data["inputs"][0], data["outputs"][0]], dim=-1)

    idx = 0 
    test_input = inputs_10ch[idx:idx+1].to(device)

    # [3] 추론 (학습과 동일한 POS_SCALE 적용)
    POS_SCALE = 200.0
    with torch.no_grad():
        pos_raw = test_input[:, :, 0:3]
        pos_normalized = pos_raw / POS_SCALE
        u_pred, mu_pred = model(pos_normalized, test_input)
    
    # 오차 계산
    u_true_np = test_input[0, :, 7:10].cpu().numpy()
    u_pred_np = u_pred[0].cpu().numpy()
    u_error = np.linalg.norm(u_pred_np - u_true_np, axis=1)

    visualize_inverse_results(pos_raw[0].cpu().numpy(), mu_pred[0].cpu().numpy(), u_error)

if __name__ == "__main__":
    main()