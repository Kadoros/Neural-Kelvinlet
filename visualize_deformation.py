import torch
import pyvista as pv
import numpy as np

def visualize_pt_data(file_path):
    # 1. 데이터 로드 (.pt 파일)
    # 데이터 구조가 dict 형태라고 가정합니다. (사용자님의 terminal 출력 기준)
    data = torch.load(file_path)
    
    # 만약 리스트 형태라면 첫 번째 데이터를 가져옵니다.
    if isinstance(data, list):
        sample = data[0]
    else:
        sample = data

    # 2. 필수 필드 추출
    # 논문 데이터셋 구조: pos(원래 좌표), out_disp(변위)
    # 'pos'가 데이터에 포함되어 있지 않다면, 학습 시 사용한 base_mesh가 필요할 수 있습니다.
    pos = sample.get('pos').numpy() if 'pos' in sample else None
    disp = sample.get('Output displacement').numpy()
    
    if pos is None:
        print("경고: 원본 좌표('pos')를 찾을 수 없습니다. 임의의 그리드를 생성합니다.")
        pos = np.zeros_like(disp) # 실제 환경에선 원본 mesh.nodes를 넣어야 합니다.

    # 3. 변형된 좌표 계산 (원래 위치 + 변위)
    deformed_pos = pos + disp
    
    # 4. 변위 크기(Magnitude) 계산 (색상 표현용)
    mag = np.linalg.norm(disp, axis=1)

    # 5. PyVista 객체 생성 (Point Cloud)
    # 만약 면(Face) 정보가 있다면 PolyData에 faces를 추가하여 Surface로 볼 수 있습니다.
    point_cloud = pv.PolyData(deformed_pos)
    point_cloud["Displacement Magnitude"] = mag

    # 6. 플로팅 설정
    plotter = pv.Plotter(title="Soft Tissue Deformation Visualization")
    plotter.add_mesh(
        point_cloud, 
        scalars="Displacement Magnitude", 
        cmap="jet",          # 논문과 유사한 무지개색 테마
        point_size=5.0, 
        render_points_as_spheres=True,
        scalar_bar_args={'title': "Displacement (mm)"}
    )
    
    # 배경색 및 카메라 설정
    plotter.set_background("white")
    plotter.add_axes()
    print("시각화 창을 띄웁니다...")
    plotter.show()

if __name__ == "__main__":
    # 파일 경로를 본인의 환경에 맞게 수정하세요.
    PATH = "data/individual_graspers_linear.pt"
    visualize_pt_data(PATH)