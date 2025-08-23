import torch
import torch.nn as nn
import torch.nn.functional as F
# -----------------------------
# Backbone: ResNet (as in repo)
# -----------------------------


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        depth=50,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        inplanes=64,
        shrink_ch_ratio=1,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif depth == 152:
            block = Bottleneck
            layers = [3, 4, 36, 3]
        elif depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        else:
            raise ValueError("only support depth in [18, 50, 101, 152]")

        shrink_input_ch = int(inplanes * shrink_ch_ratio)
        self.inplanes = int(inplanes * shrink_ch_ratio)
        if shrink_ch_ratio == 0.125:
            layers = [2, 3, 3, 3]

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, shrink_input_ch, layers[0])
        self.layer2 = self._make_layer(
            block,
            shrink_input_ch * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            shrink_input_ch * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            shrink_input_ch * 8,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        four_conv_layer = []
        x = self.layer1(x)
        four_conv_layer.append(x)
        x = self.layer2(x)
        four_conv_layer.append(x)
        x = self.layer3(x)
        four_conv_layer.append(x)
        x = self.layer4(x)
        four_conv_layer.append(x)
        return four_conv_layer


# -----------------------------------------
# Neck: LFPN (Light FPN) with DSFD features
# -----------------------------------------


class FeatureFusion(nn.Module):
    def __init__(self, lat_ch=256, **channels):
        super(FeatureFusion, self).__init__()
        self.main_conv = nn.Conv2d(channels["main"], lat_ch, kernel_size=1)

    def forward(self, up, main):
        main = self.main_conv(main)
        _, _, H, W = main.size()
        res = F.interpolate(up, scale_factor=2, mode="bilinear", align_corners=False)
        if res.size(2) != main.size(2) or res.size(3) != main.size(3):
            res = res[:, :, 0:H, 0:W]
        res = res + main
        return res


class LFPN(nn.Module):
    def __init__(
        self,
        c2_out_ch=256,
        c3_out_ch=512,
        c4_out_ch=1024,
        c5_out_ch=2048,
        c6_mid_ch=512,
        c6_out_ch=512,
        c7_mid_ch=128,
        c7_out_ch=256,
        out_dsfd_ft=True,
    ):
        super(LFPN, self).__init__()
        self.out_dsfd_ft = out_dsfd_ft
        if self.out_dsfd_ft:
            dsfd_module = []
            dsfd_module.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))  # c2
            dsfd_module.append(nn.Conv2d(512, 256, kernel_size=3, padding=1))  # c3
            dsfd_module.append(nn.Conv2d(1024, 256, kernel_size=3, padding=1))  # c4
            dsfd_module.append(nn.Conv2d(2048, 256, kernel_size=3, padding=1))  # c5
            dsfd_module.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))  # c6
            dsfd_module.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))  # c7
            self.dsfd_modules = nn.ModuleList(dsfd_module)

        c6_input_ch = c5_out_ch
        self.c6 = nn.Sequential(
            *[
                nn.Conv2d(c6_input_ch, c6_mid_ch, kernel_size=1),
                nn.BatchNorm2d(c6_mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(c6_mid_ch, c6_out_ch, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c6_out_ch),
                nn.ReLU(inplace=True),
            ]
        )
        self.c7 = nn.Sequential(
            *[
                nn.Conv2d(c6_out_ch, c7_mid_ch, kernel_size=1),
                nn.BatchNorm2d(c7_mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(c7_mid_ch, c7_out_ch, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c7_out_ch),
                nn.ReLU(inplace=True),
            ]
        )

        self.p2_lat = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.p3_lat = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.p4_lat = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.c5_lat = nn.Conv2d(c6_input_ch, 256, kernel_size=3, padding=1)
        self.c6_lat = nn.Conv2d(c6_out_ch, 256, kernel_size=3, padding=1)
        self.c7_lat = nn.Conv2d(c7_out_ch, 256, kernel_size=3, padding=1)

        self.ff_c5_c4 = FeatureFusion(main=c4_out_ch)
        self.ff_c4_c3 = FeatureFusion(main=c3_out_ch)
        self.ff_c3_c2 = FeatureFusion(main=c2_out_ch)

    def forward(self, feature_list):
        c2, c3, c4, c5 = feature_list
        c6 = self.c6(c5)
        c7 = self.c7(c6)

        c5 = self.c5_lat(c5)
        c6 = self.c6_lat(c6)
        c7 = self.c7_lat(c7)

        if self.out_dsfd_ft:
            dsfd_fts = []
            dsfd_fts.append(self.dsfd_modules[0](c2))
            dsfd_fts.append(self.dsfd_modules[1](c3))
            dsfd_fts.append(self.dsfd_modules[2](c4))
            dsfd_fts.append(self.dsfd_modules[3](feature_list[-1]))
            dsfd_fts.append(self.dsfd_modules[4](c6))
            dsfd_fts.append(self.dsfd_modules[5](c7))

        p4 = self.ff_c5_c4(c5, c4)
        p3 = self.ff_c4_c3(p4, c3)
        p2 = self.ff_c3_c2(p3, c2)

        p2 = self.p2_lat(p2)
        p3 = self.p3_lat(p3)
        p4 = self.p4_lat(p4)

        if self.out_dsfd_ft:
            return ([p2, p3, p4, c5, c6, c7], dsfd_fts)
        else:
            return [p2, p3, p4, c5, c6, c7]


# ------------------------------------
# Head: MogPredNet (deep head + DSFD)
# ------------------------------------


class conv_bn(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(
            in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)


class CPM(nn.Module):
    def __init__(self, in_plane):
        super(CPM, self).__init__()
        self.branch1 = conv_bn(in_plane, 512, 1, 1, 0)
        self.branch2a = conv_bn(in_plane, 128, 1, 1, 0)
        self.branch2b = conv_bn(128, 128, 3, 1, 1)
        self.branch2c = conv_bn(128, 512, 1, 1, 0)

        self.ssh_1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_dimred = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.ssh_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ssh_3a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ssh_3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_residual = self.branch1(x)
        x = F.relu(self.branch2a(x), inplace=True)
        x = F.relu(self.branch2b(x), inplace=True)
        x = self.branch2c(x)
        rescomb = F.relu(x + out_residual, inplace=True)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = F.relu(self.ssh_dimred(rescomb), inplace=True)
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = F.relu(self.ssh_3a(ssh_dimred), inplace=True)
        ssh_3b = self.ssh_3b(ssh_3a)
        ssh_out = torch.cat([ssh1, ssh_2, ssh_3b], dim=1)
        ssh_out = F.relu(ssh_out, inplace=True)
        return ssh_out


class SSHContext(nn.Module):
    def __init__(self, channels, Xchannels=256):
        super(SSHContext, self).__init__()
        self.conv1 = nn.Conv2d(channels, Xchannels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channels, Xchannels // 2, kernel_size=3, dilation=2, stride=1, padding=2
        )
        self.conv2_1 = nn.Conv2d(
            Xchannels // 2, Xchannels // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2_2 = nn.Conv2d(
            Xchannels // 2,
            Xchannels // 2,
            kernel_size=3,
            dilation=2,
            stride=1,
            padding=2,
        )
        self.conv2_2_1 = nn.Conv2d(
            Xchannels // 2, Xchannels // 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x), inplace=True)
        x2_1 = F.relu(self.conv2_1(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2), inplace=True)
        return torch.cat([x1, x2_1, x2_2], 1)


class DeepHead(nn.Module):
    def __init__(self, in_channel=256, out_channel=256, use_gn=False, num_conv=4):
        super(DeepHead, self).__init__()
        self.use_gn = use_gn
        self.num_conv = num_conv
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        if self.use_gn:
            self.gn1 = nn.GroupNorm(16, out_channel)
            self.gn2 = nn.GroupNorm(16, out_channel)
            self.gn3 = nn.GroupNorm(16, out_channel)
            self.gn4 = nn.GroupNorm(16, out_channel)

    def forward(self, x):
        if self.use_gn:
            x1 = F.relu(self.gn1(self.conv1(x)), inplace=True)
            x2 = F.relu(self.gn2(self.conv1(x1)), inplace=True)
            x3 = F.relu(self.gn3(self.conv1(x2)), inplace=True)
            x4 = F.relu(self.gn4(self.conv1(x3)), inplace=True)
        else:
            x1 = F.relu(self.conv1(x), inplace=True)
            x2 = F.relu(self.conv1(x1), inplace=True)
            if self.num_conv == 2:
                return x2
            x3 = F.relu(self.conv1(x2), inplace=True)
            x4 = F.relu(self.conv1(x3), inplace=True)
        return x4


class MogPredNet(nn.Module):
    def __init__(
        self,
        num_anchor_per_pixel=1,
        num_classes=1,
        input_ch_list=(256, 256, 256, 256, 256, 256),
        use_deep_head=True,
        deep_head_with_gn=True,
        use_ssh=False,
        use_cpm=True,
        use_dsfd=True,
        phase="test",
        deep_head_ch=256,
    ):
        super(MogPredNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.use_ssh = use_ssh
        self.use_cpm = use_cpm
        self.use_dsfd = use_dsfd
        self.use_deep_head = use_deep_head
        self.deep_head_with_gn = deep_head_with_gn
        self.deep_head_ch = deep_head_ch

        if self.use_dsfd:
            self.dsfd_loc = nn.Conv2d(
                input_ch_list[0], 4, kernel_size=3, stride=1, padding=1
            )
            self.dsfd_conf = nn.Conv2d(
                input_ch_list[0], 1, kernel_size=3, stride=1, padding=1
            )

        if self.use_ssh:
            self.conv_SSH = SSHContext(input_ch_list[0], self.deep_head_ch // 2)
        if self.use_cpm:
            self.conv_CPM = CPM(input_ch_list[0])

        if self.use_deep_head:
            if self.deep_head_with_gn:
                self.deep_loc_head = DeepHead(
                    self.deep_head_ch, self.deep_head_ch, use_gn=True
                )
                self.deep_cls_head = DeepHead(
                    self.deep_head_ch, self.deep_head_ch, use_gn=True
                )
            else:
                self.deep_loc_head = DeepHead(self.deep_head_ch, self.deep_head_ch)
                self.deep_cls_head = DeepHead(self.deep_head_ch, self.deep_head_ch)
            self.pred_cls = nn.Conv2d(
                self.deep_head_ch, 1 * num_anchor_per_pixel, 3, 1, 1
            )
            self.pred_loc = nn.Conv2d(
                self.deep_head_ch, 4 * num_anchor_per_pixel, 3, 1, 1
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, pyramid_feature_list, dsfd_ft_list=None):
        loc = []
        conf = []

        if dsfd_ft_list is not None:
            assert self.use_dsfd, "should set 'use_dsfd = True'."
            dsfd_conf = []
            dsfd_loc = []
            for x in dsfd_ft_list:
                dsfd_conf.append(self.dsfd_conf(x).permute(0, 2, 3, 1).contiguous())
                dsfd_loc.append(self.dsfd_loc(x).permute(0, 2, 3, 1).contiguous())

        if self.use_deep_head:
            for x in pyramid_feature_list:
                if self.use_ssh:
                    x = self.conv_SSH(x)
                if self.use_cpm:
                    x = self.conv_CPM(x)
                x_cls = self.deep_cls_head(x)
                x_loc = self.deep_loc_head(x)
                conf.append(self.pred_cls(x_cls).permute(0, 2, 3, 1).contiguous())
                loc.append(self.pred_loc(x_loc).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

        if self.phase == "training":
            if self.use_dsfd:
                dsfd_loc = torch.cat([o.view(o.size(0), -1, 4) for o in dsfd_loc], 1)
                dsfd_conf = torch.cat(
                    [o.view(o.size(0), -1, self.num_classes) for o in dsfd_conf], 1
                )
                return (
                    conf.view(conf.size(0), -1, self.num_classes),
                    loc.view(loc.size(0), -1, 4),
                    dsfd_conf.view(dsfd_conf.size(0), -1, self.num_classes),
                    dsfd_loc.view(dsfd_loc.size(0), -1, 4),
                )
            else:
                return (
                    conf.view(conf.size(0), -1, self.num_classes),
                    loc.view(loc.size(0), -1, 4),
                )
        else:
            return (
                self.sigmoid(conf.view(conf.size(0), -1, self.num_classes)),
                loc.view(loc.size(0), -1, 4),
            )


# --------------------------------
# Architecture: WiderFaceBaseNet
# --------------------------------


class WiderFaceBaseNet(nn.Module):
    def __init__(self, backbone, fpn, pred_net, phase="test", out_bb_ft=False):
        super(WiderFaceBaseNet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pred_net = pred_net
        self.phase = phase
        self.out_bb_ft = out_bb_ft
        if self.phase == "training":
            pass

    def forward(self, x):
        feature_list = self.backbone(x)
        fpn_list = self.fpn(feature_list)
        if isinstance(fpn_list, (list, tuple)) and len(fpn_list) == 2:
            pyramid_feature_list = fpn_list[0]
            dsfd_ft_list = fpn_list[1]
        else:
            pyramid_feature_list = fpn_list
            dsfd_ft_list = None
        if self.phase == "training":
            if dsfd_ft_list is not None:
                conf, loc, dsfd_conf, dsfd_loc = self.pred_net(
                    pyramid_feature_list, dsfd_ft_list
                )
                return conf, loc, dsfd_conf, dsfd_loc
            conf, loc = self.pred_net(pyramid_feature_list)
            if self.out_bb_ft:
                return conf, loc, feature_list
            else:
                return conf, loc
        else:
            if dsfd_ft_list is not None:
                conf, loc = self.pred_net(pyramid_feature_list, dsfd_ft_list)
                return conf, loc
            conf, loc = self.pred_net(pyramid_feature_list)
            if self.out_bb_ft:
                return conf, loc, feature_list
            else:
                return conf, loc


def build_mogface_e():
    backbone = ResNet(depth=152)
    fpn = LFPN(
        c2_out_ch=256,
        c3_out_ch=512,
        c4_out_ch=1024,
        c5_out_ch=2048,
        c6_mid_ch=512,
        c6_out_ch=512,
        c7_mid_ch=128,
        c7_out_ch=256,
        out_dsfd_ft=True,
    )
    pred_net = MogPredNet(
        num_anchor_per_pixel=1,
        num_classes=1,
        input_ch_list=(256, 256, 256, 256, 256, 256),
        use_deep_head=True,
        deep_head_with_gn=True,
        deep_head_ch=256,
        use_ssh=False,
        use_cpm=True,
        use_dsfd=True,
        phase="test",
    )
    model = WiderFaceBaseNet(
        backbone=backbone, fpn=fpn, pred_net=pred_net, phase="test"
    )
    return model
