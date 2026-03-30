# eval_pino.py
import torch
import pyvista as pv
import numpy as np
from KelvinHyperPINO import KelvinHyperPINO


def visualize_visual_haptics(pos_np, mu_pred_np, mu_gt_np=None):
    """
    pos_np: (N, 3) 3D 좌표
    mu_pred_np: (N, 1) 모델이 예측한 상대적 탄성도
    mu_gt_np: (N, 1) 정답 탄성도 (비교용, 생략 가능)
    """
    # 1. PyVista Point Cloud 객체 생성
    cloud = pv.PolyData(pos_np)

    # 2. 예측된 탄성도(mu)를 Point 속성으로 추가
    cloud["Predicted Elasticity (mu)"] = mu_pred_np.flatten()

    if mu_gt_np is not None:
        cloud["Ground Truth Elasticity"] = mu_gt_np.flatten()

        # 두 개를 나란히 비교해서 그리기
        p = pv.Plotter(shape=(1, 2), window_size=(1600, 600))

        # 왼쪽: 예측값
        p.subplot(0, 0)
        p.add_mesh(
            cloud,
            scalars="Predicted Elasticity (mu)",
            cmap="jet",
            point_size=5.0,
            render_points_as_spheres=True,
        )
        p.add_text("PINO Prediction (Visual Haptics)", font_size=12)

        # 오른쪽: 정답(GT)
        p.subplot(0, 1)
        p.add_mesh(
            cloud,
            scalars="Ground Truth Elasticity",
            cmap="jet",
            point_size=5.0,
            render_points_as_spheres=True,
        )
        p.add_text("Ground Truth", font_size=12)

    else:
        # 예측값만 그리기
        p = pv.Plotter()
        p.add_mesh(
            cloud,
            scalars="Predicted Elasticity (mu)",
            cmap="jet",
            point_size=5.0,
            render_points_as_spheres=True,
        )
        p.add_text("PINO Predicted Relative Elasticity", font_size=14)

    p.link_views()  # 카메라 시점 동기화
    p.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 학습된 모델 불러오기
    print("Loading Trained PINO Model...")
    model = KelvinHyperPINO(target_width=128).to(device)

    # 학습이 아직 안 끝났다면 에러가 날 테니, try-except로 처리해둡니다.
    try:
        model.load_state_dict(
            torch.load("kelvin_pino_10ch_best.pth", map_location=device)
        )
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return

    # 2. 테스트용 데이터 1개 불러오기
    print("Loading Test Data...")
    dataset_path = "data/individual_graspers_linear.pt"
    data = torch.load(dataset_path)

    inputs = data["inputs"]  # (B, N, 10)
    targets = data["mu_gt"]  # (B, N, 1)

    # 첫 번째 배치(인덱스 0) 샘플만 가져와서 테스트
    test_input = inputs[0:1].to(device)  # (1, N, 10)
    test_target = targets[0:1]  # (1, N, 1)

    # 3. 모델 추론 (Inference) - PINO의 핵심! 단 한 번의 Forward Pass로 끝!
    print("Running Inference...")
    with torch.no_grad():  # 평가할 때는 미분(그래프)을 추적하지 않습니다.
        u_pred, mu_pred = model(test_input)

    # 4. 시각화를 위해 Numpy로 변환
    pos = test_input[0, :, 0:3].cpu().numpy()
    mu_pred_np = mu_pred[0].cpu().numpy()
    mu_gt_np = test_target[0].numpy()

    # 5. 시각적 촉각(Visual Haptics) 렌더링
    print("Rendering 3D Visual Haptics...")
    visualize_visual_haptics(pos, mu_pred_np, mu_gt_np)


if __name__ == "__main__":
    main()
