import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.nn.functional as F
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, cuda_use=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.cuda_use = cuda_use

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H)
        :return: out
        """
        batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
        xx_range = torch.arange(dim_x, dtype=torch.int32).cuda()
        xx_channel = xx_range[None, None, :]

        xx_channel = xx_channel.float() / (dim_x - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)
        out = torch.cat([input_tensor, xx_channel], dim=1)

        # if self.cuda_use:
        #     xx_channel = xx_channel.cuda()
        #     out = out.cuda()

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out
class CoordConv1d(conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, cuda_use=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H)
        output_tensor_shape: N,C_out,H_out）
        :return: CoordConv1d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class ProteinCoordconv(nn.Module):
    def __init__(self, dim, kernels=[3, 6, 9]):
        super(ProteinCoordconv, self).__init__()
        self.conv1 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = CoordConv1d(in_channels=dim, out_channels=dim, kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        return x

#DRUG:304
#PROTEIN:1171
if __name__ == "__main__":
    x = torch.randn(64, 1171, 128)
    print(x.shape)
    m = ProteinCoordconv(128)
    print(m(x).shape)
