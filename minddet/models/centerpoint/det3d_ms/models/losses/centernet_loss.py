import mindspore as ms
from mindspore import nn, ops


def _gather_feat(feat, ind, mask=None):
    expand_dims = ops.ExpandDims()
    dim = feat.shape[2]
    ind = ops.BroadcastTo((ind.shape[0], ind.shape[1], dim))(
        expand_dims(ind, 2)
    )  # ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = ops.GatherD()(feat, 1, ind)  # feat.gather(1, ind)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = ops.Transpose()(feat, (0, 2, 3, 1))  # feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat


class RegLoss(nn.Cell):
    """Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
    """

    def __init__(self):
        super(RegLoss, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.cast = ops.Cast()

    def construct(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = self.expand_dims(
            self.cast(mask, ms.float32), 2
        )  # mask.float().unsqueeze(2)

        loss = nn.L1Loss(reduction="none")(pred * mask, target * mask)
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 1, 0).sum(axis=2).sum(axis=1)
        return loss


class FastFocalLoss(nn.Cell):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.expand_dims = ops.ExpandDims()
        self.cast = ops.Cast()

    def construct(self, out, target, ind, mask, cat):
        """
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        """
        mask = self.cast(mask, ms.float32)  # mask.float()
        gt = self.pow(1 - target, 4)
        neg_loss = self.log(1 - out) * self.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = ops.GatherD()(
            pos_pred_pix, 2, self.expand_dims(cat, 2)
        )  # pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
        num_pos = mask.sum()
        pos_loss = (
            self.log(pos_pred) * self.pow(1 - pos_pred, 2) * self.expand_dims(mask, 2)
        )  # mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        return ops.Select()(num_pos == 0, -neg_loss, -(pos_loss + neg_loss) / num_pos)
