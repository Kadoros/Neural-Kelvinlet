import torch
import torch.nn as nn
from pointnet import PointNetfeat


class FCLayer_batch(nn.Module):
    """
    Fully Connected Layer with dynamically generated weights.
    Supports batched inputs: (batch, pred_num, input_dim)
    """

    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.weight_size = torch.tensor([num_out, num_in])  # Shape of weight matrix
        self.bias_size = torch.tensor([num_out])  # Shape of bias vector

    def forward(self, x, param):
        """
        Forward pass with dynamic weight and bias.

        Args:
            x: Tensor of shape (batch, pred_num, input_dim)
            param: Tensor of shape (batch, param_size) containing weights and biases.

        Returns:
            Output Tensor of shape (batch, pred_num, output_dim)
        """
        B, P, _ = x.shape  # (batch, pred_num, input_dim)

        # Extract weights and biases for each batch
        w = param[:, : torch.prod(self.weight_size)]  # (batch, weight_size)
        b = param[:, torch.prod(self.weight_size) :]  # (batch, bias_size)

        # Reshape parameters
        w = w.reshape(
            B, self.weight_size[0], self.weight_size[1]
        )  # (batch, output_dim, input_dim)
        b = b.reshape(B, self.bias_size[0])  # (batch, output_dim)

        # Apply affine transformation (batch matrix multiplication)
        return torch.einsum("bpi, bio->bpo", x, w.transpose(1, 2)) + b.unsqueeze(
            1
        )  # (batch, pred_num, output_dim)

    def get_param_size(self):
        return torch.prod(self.weight_size) + self.bias_size


class KelvinHyperPINO(nn.Module):
    def __init__(self, target_width=128):
        super().__init__()

        # 1. HyperNet: 10채널 데이터를 훑어서 1024차원 특징 추출
        self.hyper_net = PointNetfeat(global_feat=True)

        # 2. TargetNet (Decoder)
        self.layer1 = FCLayer_batch(3, target_width)
        self.layer2 = FCLayer_batch(target_width, 4)  # 출력: u(3) + mu(1)

        self.size1 = self.layer1.get_param_size()
        self.size2 = self.layer2.get_param_size()
        self.param_proj = nn.Linear(1024, self.size1 + self.size2)

    # [수정됨] 미분을 위한 pos를 외부에서 명시적으로 받도록 인자 추가
    def forward(self, pos, full_input):
        """
        pos: (B, N, 3) -> requires_grad=True 가 걸려있는 미분용 좌표 텐서
        full_input: (B, N, 10) -> HyperNet용 전체 텐서
        """
        # A. HyperNet: 상황 파악 (기울기 추적 불필요)
        global_feat = self.hyper_net(full_input)

        # B. 가중치 생성
        all_params = self.param_proj(global_feat)
        p1 = all_params[:, : self.size1]
        p2 = all_params[:, self.size1 :]

        # C. TargetNet: 외부에서 전달받은 'pos'를 직접 사용!!! (매우 중요)
        x = torch.sin(self.layer1(pos, p1))
        out = self.layer2(x, p2)  # (B, N, 4)

        u_pred = out[:, :, 0:3]  # 변위 예측값
        mu_pred = out[:, :, 3:4]  # 탄성 예측값

        return u_pred, mu_pred
