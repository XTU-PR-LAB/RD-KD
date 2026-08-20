import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))
        attn = torch.sigmoid(avg_out + max_out)
        return attn


class CBAMBlock(nn.Module):
    def __init__(self, num_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_mask = self.channel_attention(x)
        x = x * c_mask
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s_mask = self.spatial_attention(torch.cat([avg, mx], dim=1))
        x = x * s_mask
        return x


class SharedExtractor(nn.Module):

    def __init__(self, C_in=384, C_common=128):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(C_in, C_common, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(C_common),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            self._res_block(C_common),
            self._res_block(C_common),
            self._res_block(C_common)
        )

    def _res_block(self, C):
        return nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C)
        )

    def forward(self, x):
        x = self.reduce(x)
        residual = x
        for block in self.res_blocks:
            out = block(x)
            x = F.relu(out + residual)
            residual = x
        return x


def build_feature_connector(s_channel, t_channel):
    connector = nn.Sequential(
        nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(t_channel)
    )
    for m in connector.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return connector


class HeteroFeatureDistiller(nn.Module):


    def __init__(self,
                 student_channels=(32,64,128),
                 teacher_channels=(384, 384, 384, 384),
                 teacher_hw=(28, 48),
                 feature_pairs=((0, 1), (1, 2), (2, 3)),
                 distill_weight=1.0,
                 C_common=128,
                 sigma=1.0,
                 recon_weight=0.3):
        super().__init__()

        self.feature_pairs = feature_pairs
        self.teacher_height, self.teacher_width = teacher_hw
        self.distill_weight = distill_weight
        self.C_common = C_common
        self.sigma = sigma
        self.recon_weight = recon_weight

        self.teacher_recon_heads = nn.ModuleList([
            ReconHead(C_in=C_common, C_out=t)
            for t in teacher_channels
        ])

        self.student_cbam = nn.ModuleList([CBAMBlock(c) for c in student_channels])
        self.teacher_cbam = nn.ModuleList([CBAMBlock(c) for c in teacher_channels])

        self._init_cbam_weights(self.teacher_cbam)
        for m in self.teacher_cbam:
            for p in m.parameters():
                p.requires_grad = False

        self.Connectors = nn.ModuleList([
            build_feature_connector(s, t)
            for s, t in zip(student_channels, teacher_channels[:len(student_channels)])
        ])

        self.shared_projectors = nn.ModuleList([
            SharedExtractor(C_in=t, C_common=C_common)
            for t in teacher_channels
        ])

    def _init_cbam_weights(self, module_list):
        for m in module_list.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _tokens_to_feature_map(self, t_feat):
        B, N, C = t_feat.shape
        return t_feat.view(B, C, self.teacher_height, self.teacher_width)

    def forward(self, student_feats, teacher_feats):
        total_loss, cnt = 0.0, 0

        hw = teacher_feats[1]
        self.teacher_height, self.teacher_width = teacher_feats[1][1], teacher_feats[1][2]

        refined_teacher = []
        for i, t_feat in enumerate(teacher_feats[0]):
            t_map = self._tokens_to_feature_map(t_feat[:, 5:, :])
            t_refined = self.teacher_cbam[i](t_map).detach()
            refined_teacher.append(t_refined)

        recon_loss = 0.0
        for si, ti in self.feature_pairs:
            s_map = self.student_cbam[si](student_feats[si])
            t_map = refined_teacher[ti]

            s_aligned = self.Connectors[si](s_map)

            if s_aligned.shape[-2:] != t_map.shape[-2:]:
                s_aligned = F.interpolate(s_aligned, size=t_map.shape[-2:], mode='bilinear', align_corners=False)

            t_projected = self.shared_projectors[ti](t_map).detach()
            s_projected = self.shared_projectors[ti](s_aligned)

            t_recon = self.teacher_recon_heads[ti](t_projected)
            if t_recon.shape[2:] != t_map.shape[2:]:
                t_recon = F.interpolate(t_recon, size=t_map.shape[2:],
                                        mode='bilinear', align_corners=False)
            recon_loss += F.mse_loss(t_recon, t_map.detach())


            loss = compute_mmd(t_projected.detach(), s_projected, sigma=self.sigma)


            total_loss += loss
            total_loss += self.recon_weight * recon_loss
            cnt += 1

        return self.distill_weight * (total_loss / max(cnt, 1))


class ReconHead(nn.Module):
    def __init__(self, C_in=128, C_out=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_out, C_out, 3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


def compute_mmd(f_t, f_s, sigma=1.0):
    B, C, H, W = f_t.shape
    n = H * W
    ft = f_t.flatten(2).transpose(1, 2)
    fs = f_s.flatten(2).transpose(1, 2)
    ft = F.normalize(ft, dim=-1)
    fs = F.normalize(fs, dim=-1)

    def gaussian_kernel(x, y):
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm = (y ** 2).sum(dim=-1, keepdim=True)
        dist = x_norm + y_norm.transpose(1, 2) - 2 * torch.bmm(x, y.transpose(1, 2))
        return torch.exp(-dist / (2 * sigma ** 2))

    K_tt = gaussian_kernel(ft, ft)
    K_ss = gaussian_kernel(fs, fs)
    K_ts = gaussian_kernel(ft, fs)
    mmd = K_tt.mean() - 2 * K_ts.mean() + K_ss.mean()
    return mmd