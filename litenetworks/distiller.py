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
                 student_channels,
                 teacher_channels,
                 teacher_hw,
                 feature_pairs,
                 distill_weight):
        super().__init__()

        self.feature_pairs = feature_pairs
        self.teacher_height, self.teacher_width = teacher_hw
        self.distill_weight = distill_weight

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

        refined_teacher = []
        for i, (t_feat, _) in enumerate(teacher_feats):
            t_map = self._tokens_to_feature_map(t_feat)
            t_refined = self.teacher_cbam[i](t_map).detach()
            refined_teacher.append(t_refined)

        for si, ti in self.feature_pairs:
            s_map = self.student_cbam[si](student_feats[si])
            t_map = refined_teacher[ti]

            s_aligned = self.Connectors[si](s_map)

            if s_aligned.shape[-2:] != t_map.shape[-2:]:
                s_aligned = F.interpolate(s_aligned, size=t_map.shape[-2:], mode='bilinear', align_corners=False)

            s_norm = s_aligned / (s_aligned.norm(p=2, dim=1, keepdim=True) + 1e-6)
            t_norm = t_map / (t_map.norm(p=2, dim=1, keepdim=True) + 1e-6)
            loss = F.mse_loss(s_norm, t_norm)

            total_loss += loss
            cnt += 1

        return self.distill_weight * (total_loss / max(cnt, 1))
