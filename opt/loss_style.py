# import kornia
import torch
from torchvision import models, transforms
import torch.nn.functional as F


class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(weights='DEFAULT').eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_feats(self, x, layers=[], supress_assert=True):
        # Layer indexes:
        # Conv1_*: 1,3
        # Conv2_*: 6,8
        # Conv3_*: 11, 13, 15
        # Conv4_*: 18, 20, 22
        # Conv5_*: 25, 27, 29

        if not supress_assert:
            assert x.min() >= 0.0 and x.max() <= 1.0, "input is expected to be an image scaled between 0 and 1"

        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs


def cos_distance(a, b, center=True):
    """a: [b, c, hw],
    b: [b, c, h2w2]
    """
    # """cosine distance
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()

    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)

    d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)
    # """"

    """
    a_norm_sq = (a * a).sum(1).unsqueeze(2)
    b_norm_sq = (b * b).sum(1).unsqueeze(1)

    d_mat = a_norm_sq + b_norm_sq - 2.0 * torch.matmul(a.transpose(2, 1), b)
    """
    return d_mat


def cos_loss(a, b):
    # """cosine loss
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()
    # """

    # return ((a - b) ** 2).mean()


def feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    for i in range(n):
        z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

        z_best = torch.argmin(z_dist, 2)
        del z_dist

        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)

        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new



def nn_loss(outputs, styles, vgg, blocks=[2]):

    blocks.sort()
    block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
    total_loss = 0.0

    all_layers = []
    for block in blocks:
        all_layers += block_indexes[block]

    x_feats_all = vgg.get_feats(outputs, all_layers)
    with torch.no_grad():
        s_feats_all = vgg.get_feats(styles, all_layers)

    ix_map = {}
    for a, b in enumerate(all_layers):
        ix_map[b] = a

    for block in blocks:
        layers = block_indexes[block]
        x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
        s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

        target_feats = feat_replace(x_feats, s_feats)
        total_loss += cos_loss(x_feats, target_feats)

    return total_loss


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G


class LossStyle(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGG().to(device)

    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nn_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
        contw=1e-3,
        gramw=1e-9,
    ):
        blocks.sort()
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.vgg.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.vgg.get_feats(styles, all_layers)
            if contents is not None:
                content_feats_all = self.vgg.get_feats(contents, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        nn_loss = 0.0
        gram_loss = 0.0
        content_loss = 0.0
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if "nn_loss" in loss_names:
                target_feats = feat_replace(x_feats, s_feats)
                nn_loss += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                gram_loss += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)

            if contents is not None:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                content_loss += torch.mean((content_feats - x_feats) ** 2)

        return contw * content_loss + gramw * gram_loss + nn_loss