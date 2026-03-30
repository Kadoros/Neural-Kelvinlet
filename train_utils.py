import torch
import torch.nn as nn


def cuda2numpy(Tensor, device):
    return Tensor.detach().cpu().numpy()

def df2cuda(df):
    return torch.from_numpy(np.array(df)).to(device)

def deformation_capture_mean(predicted, ground_truth):

    error_vectors = predicted - ground_truth
    error_magnitudes = torch.norm(error_vectors, dim=1)
    d_error = torch.mean(error_magnitudes)
    measured_magnitudes = torch.norm(ground_truth, dim=1)
    d_meas = torch.mean(measured_magnitudes)
    percentage_correction = 100 * (1 - (d_error / d_meas))
    
    return percentage_correction.item()

def feature_transform_regularizer(trans, device):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


def deformation_capture_max(predicted, ground_truth):
    error_vectors = predicted - ground_truth
    error_magnitudes = torch.norm(error_vectors, dim=1)
    d_error = torch.max(error_magnitudes)
    measured_magnitudes = torch.norm(ground_truth, dim=1)
    d_meas = torch.max(measured_magnitudes)
    
    # 0으로 나누기 방지
    if d_meas == 0: return 0.0
    
    percentage_correction = 100 * (1 - (d_error / d_meas))
    return percentage_correction.item()

def deformation_capture_median(predicted, ground_truth):
    error_vectors = predicted - ground_truth
    error_magnitudes = torch.norm(error_vectors, dim=1)
    d_error = torch.median(error_magnitudes)
    measured_magnitudes = torch.norm(ground_truth, dim=1)
    d_meas = torch.median(measured_magnitudes)
    
    # 0으로 나누기 방지
    if d_meas == 0: return 0.0
    
    percentage_correction = 100 * (1 - (d_error / d_meas))
    return percentage_correction.item()