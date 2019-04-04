from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import ChainList

from chainercv.links.model.ssd import Multibox
from chainercv.links.model.ssd import Normalize
from chainercv.links.model.ssd import SSD
from chainercv.links.model.ssd import MultiboxCoder
from chainercv.links import Conv2DBNActiv
from chainercv.links import SEBlock
from chainercv import transforms
from chainercv import utils


# RGB, (C, 1, 1) format
_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))


class VGG16(chainer.Chain):
    """An extended VGG-16 model for M2Det.
    """

    def __init__(self):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(64, 3, pad=1)
            self.conv1_2 = L.Convolution2D(64, 3, pad=1)

            self.conv2_1 = L.Convolution2D(128, 3, pad=1)
            self.conv2_2 = L.Convolution2D(128, 3, pad=1)

            self.conv3_1 = L.Convolution2D(256, 3, pad=1)
            self.conv3_2 = L.Convolution2D(256, 3, pad=1)
            self.conv3_3 = L.Convolution2D(256, 3, pad=1)

            self.conv4_1 = L.Convolution2D(512, 3, pad=1)
            self.conv4_2 = L.Convolution2D(512, 3, pad=1)
            self.conv4_3 = L.Convolution2D(512, 3, pad=1)

            self.conv5_1 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_2 = L.DilatedConvolution2D(512, 3, pad=1)
            self.conv5_3 = L.DilatedConvolution2D(512, 3, pad=1)

            self.conv6 = L.DilatedConvolution2D(1024, 3, pad=6, dilate=6)
            self.conv7 = L.Convolution2D(1024, 1)

    def forward(self, x):
        ys = []

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        ys.append(h)
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        ys.append(h)

        return ys


class M2Det300(SSD):
    _models = {
        'imagenet': {
            'url': 'https://chainercv-models.preferred.jp/'
            'ssd_vgg16_imagenet_converted_2017_06_09.npz',
            'cv2': True
        },
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(M2Det300, self).__init__(
            extractor=VGG16MLFPN300(),
            multibox=Multibox(
                n_class=param['n_fg_class'] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=(30, 60, 111, 162, 213, 264, 315),
            mean=_imagenet_mean)

        if path:
            chainer.serializers.load_npz(path, self, strict=False)


class M2Det320(SSD):
    _models = {
        'imagenet': {
            'url': 'https://chainercv-models.preferred.jp/'
            'ssd_vgg16_imagenet_converted_2017_06_09.npz',
            'cv2': True
        },
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(M2Det320, self).__init__(
            extractor=VGG16MLFPN320(),
            multibox=Multibox(
                n_class=param['n_fg_class'] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 107, 320),
            sizes=(25.6, 48, 105.6, 163.2, 220.8, 278.4, 336),
            mean=_imagenet_mean)

        if path:
            chainer.serializers.load_npz(path, self, strict=False)


class M2Det512(SSD):
    _models = {
        'imagenet': {
            'url': 'https://chainercv-models.preferred.jp/'
            'ssd_vgg16_imagenet_converted_2017_06_09.npz',
            'cv2': True
        },
    }

    def __init__(self, n_fg_class=None, pretrained_model=None):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(M2Det512, self).__init__(
            extractor=VGG16MLFPN512(),
            multibox=Multibox(
                n_class=param['n_fg_class'] + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 128, 256),
            sizes=(30.72, 76.8, 168.96, 261.12, 353.28, 445.44, 537.6),
            mean=_imagenet_mean)

        if path:
            chainer.serializers.load_npz(path, self, strict=False)


class VGG16MLFPN300(VGG16):
    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self):
        super(VGG16MLFPN300, self).__init__()

        with self.init_scope():
            self.mlfpn = MLFPN()

    def forward(self, x):
        x = super(VGG16MLFPN300, self).forward(x)
        ys = self.mlfpn(x)

        return ys


class VGG16MLFPN320(VGG16):
    insize = 320
    grids = (40, 20, 10, 5, 3, 1)

    def __init__(self):
        super(VGG16MLFPN300, self).__init__()

        with self.init_scope():
            self.mlfpn = MLFPN()

    def forward(self, x):
        x = super(VGG16MLFPN320, self).forward(x)
        ys = self.mlfpn(x)

        return ys


class VGG16MLFPN512(VGG16):
    insize = 512
    grids = (64, 32, 16, 8, 4, 2)

    def __init__(self):
        super(VGG16MLFPN512, self).__init__()

        with self.init_scope():
            self.mlfpn = MLFPN(mingrid=2)

    def forward(self, x):
        x = super(VGG16MLFPN512, self).forward(x)
        ys = self.mlfpn(x)

        return ys


class FFMv1(chainer.Chain):
    def __init__(self):
        super(FFMv1, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(256, 3, pad=1)
            self.bn1 = L.BatchNormalization(256)
            self.conv2 = L.Convolution2D(512, 1)
            self.bn2 = L.BatchNormalization(512)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = F.resize_images(x2, (x1.shape[2], x1.shape[3]))
        x = F.concat((x1, x2), axis=1)

        return x


class FFMv2(chainer.Chain):
    def __init__(self):
        super(FFMv2, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(128, 1)
            self.bn1 = L.BatchNormalization(128)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x = F.concat((x1, x2), axis=1)

        return x


class TUM(chainer.Chain):
    def __init__(self, inplanes, scales=6, mingrid=1):
        super(TUM, self).__init__()

        self.scales = scales

        with self.init_scope():
            ecs = []
            for s in range(scales-1):
                if s == 0:
                    conv = Conv2DBNActiv(inplanes, 256, 3, 2, pad=1, nobias=True)
                elif s == scales-2 and mingrid == 1:
                    conv = Conv2DBNActiv(256, 256, 3, 2, nobias=True)
                else:
                    conv = Conv2DBNActiv(256, 256, 3, 2, pad=1, nobias=True)
                ecs.append(conv)
            self.ecs = ChainList(*ecs)

            dcs = []
            for s in range(scales):
                if s == scales-1:
                    conv = Conv2DBNActiv(inplanes, 256, 3, pad=1, nobias=True)
                else:
                    conv = Conv2DBNActiv(256, 256, 3, pad=1, nobias=True)
                dcs.append(conv)
            self.dcs = ChainList(*dcs)

            self.scs = ChainList(*[Conv2DBNActiv(256, 128, 1, nobias=True) for _ in range(scales)])

    def __call__(self, x):
        e = x
        es = [e]
        for conv in self.ecs.children():
            e = conv(e)
            es.append(e)

        d = es[-1]
        ds = [d]
        for s in range(self.scales-2):
            d = F.resize_images(self.dcs[s](d), (es[-(s+2)].shape[2], es[-(s+2)].shape[3])) + es[-(s+2)]
            ds.append(d)
        d = F.resize_images(self.dcs[self.scales-2](d), (x.shape[2], x.shape[3])) + self.dcs[self.scales-1](x)
        ds.append(d)

        ys = []
        for s in range(self.scales):
            ys.append(self.scs[s](ds[s]))

        return ys[::-1]


class SFAM(chainer.Chain):
    def __init__(self, levels=8, scales=6, planes=1024):
        super(SFAM, self).__init__()

        self.levels = levels
        self.scales = scales

        with self.init_scope():
            self.ses = ChainList(*[SEBlock(planes) for _ in range(scales)])

    def __call__(self, x):
        ys = []
        for s in range(self.scales):
            h = F.concat((x[l][s] for l in range(self.levels)), axis=1)
            ys.append(self.ses[s](h))

        return ys


class MLFPN(chainer.Chain):
    def __init__(self, levels=8, scales=6, mingrid=1):
        super(MLFPN, self).__init__()

        self.levels = levels

        with self.init_scope():
            self.ffmv1 = FFMv1()
            self.tums = ChainList(*[TUM(768, scales, mingrid) if l == 0 else TUM(256, scales, mingrid) for l in range(levels)])
            self.ffmv2s = ChainList(*[FFMv2() for _ in range(levels-1)])
            self.sfam = SFAM(levels, scales)

    def forward(self, x):
        base = self.ffmv1(x[0], x[1])
        ys = []
        h = self.tums[0](base)
        ys.append(h)
        for l in range(self.levels-1):
            h = self.tums[l+1](self.ffmv2s[l](base, h[0]))
            ys.append(h)
        ys = self.sfam(ys)

        return ys
