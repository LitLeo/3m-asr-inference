from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from layer.decoder import TransformerDecoder
from layer.decoder import BiTransformerDecoder
from loss.loss_compute import LabelSmoothingLoss
from utils.common import add_sos_eos, reverse_pad_list
from utils.mask import make_pad_mask

from model.conformer2 import Net as ConformerEncoder
from model.ctc_aed import JointCtcAedModel


class Net(JointCtcAedModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_conf: Optional[Dict[str, Any]] = None,
        decoder_type: str = "transformer",
        decoder_conf: Optional[Dict[str, Any]] = None,
        ignore_id: int = -1,
        ctc_weight: float = 0.3,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalize_loss: bool = False
    ):
        super().__init__(
            input_dim, output_dim, encoder_conf, decoder_type,
            decoder_conf, ignore_id, ctc_weight, reverse_weight,
            lsm_weight, length_normalize_loss)
        # use default config for construction function
        if encoder_conf is None:
            encoder_conf = {}
        if decoder_conf is None:
            decoder_conf = {}
        self.encoder = ConformerEncoder(input_dim, output_dim, **encoder_conf)

        encoder_out_dim = self.encoder.encoder_embed_dim
        self.build_decoder(self.output_dim, encoder_out_dim,
                decoder_type, decoder_conf)
        # self.build_criterion()

    def build_decoder(
        self,
        vocab_size: int,
        encoder_out_dim: int,
        decoder_type: str = "transformer",
        decoder_conf: Optional[Dict[str, Any]] = None,
    ):
        if decoder_conf is None:
            # use default config of construction function
            decoder_conf = {}
        if decoder_type == "transformer":
            self.decoder = TransformerDecoder(
                vocab_size, encoder_out_dim, **decoder_conf)
            self.decoder2 = TransformerDecoder(
                vocab_size, encoder_out_dim, **decoder_conf)
        else:
            assert 0.0 < self.reverse_weight < 1.0
            assert "r_num_blocks" in decoder_conf and \
                    decoder_conf["r_num_blocks"] > 0
            self.decoder = BiTransformerDecoder(
                    vocab_size, encoder_out_dim, **decoder_conf)
            self.decoder2 = BiTransformerDecoder(
                    vocab_size, encoder_out_dim, **decoder_conf)

    @property
    def metric_tags(self):
        tags = []
        if self.ctc_weight > 0.0:
            tags += ['ctc_loss']
        if self.ctc_weight < 1.0:
            tags += ['aed_loss', 'aed_loss2']
        return tags

    def forward(
        self,
        feats: torch.Tensor,
        feat_lens: torch.Tensor,
        target: torch.Tensor,
        target_lens: torch.Tensor
    ) -> Dict[str, Any]:
        # ctc branch
        res = self.encoder(feats, feat_lens)
        encoder_out = res['hidden']
        out_lens = res['out_lens']
        branch_out = res['branch_hidden']
        max_step = encoder_out.size(1)
        encoder_mask = ~make_pad_mask(out_lens, max_step)
        encoder_mask = encoder_mask.unsqueeze(1)
        # aed branch
        ys_in_pad, ys_out_pad = add_sos_eos(target, self.sos, self.eos, self.ignore_id)
        ys_in_lens = target_lens + 1
        # reverse the seq, used for right-to-left decoder
        r_ys_pad = reverse_pad_list(target, target_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        # forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, ys_in_pad,
            ys_in_lens, r_ys_in_pad, self.reverse_weight)
        decoder_out2, r_decoder_out2, _ = self.decoder2(
            encoder_out, encoder_mask, ys_in_pad,
            ys_in_lens, r_ys_in_pad, self.reverse_weight)
        res['decoder_out'] = decoder_out
        res['r_decoder_out'] = r_decoder_out
        res['ys_out_pad'] = ys_out_pad
        res['r_ys_out_pad'] = r_ys_out_pad
        res['decoder_out2'] = decoder_out2
        res['r_decoder_out2'] = r_decoder_out2
        return res

    def cal_loss(self, res, target, target_lens):
        out_nosm = res['out_nosm']
        out_lens = res['out_lens']
        decoder_out, ys_out_pad = res['decoder_out'], res['ys_out_pad']
        r_decoder_out, r_ys_out_pad = res['r_decoder_out'], res['r_ys_out_pad']
        decoder_out2 = res['decoder_out2']
        r_decoder_out2 = res['r_decoder_out2']
        loss, metric, count = 0.0, (), ()
        if self.ctc_weight > 0.0:
            loss_ctc, metric_ctc, count_ctc = self.ctc_criterion(
                out_nosm, out_lens, target, target_lens)
            loss += self.ctc_weight * loss_ctc
            metric += metric_ctc
            count += count_ctc
        if self.ctc_weight < 1.0:
            loss_att, metric_att, count_att = self.att_criterion(
                decoder_out, ys_out_pad)
            loss_att2, metric_att2, count_att2 = self.att_criterion(
                decoder_out2, ys_out_pad)
            # ignore the metric of reverse decoder
            if hasattr(self.decoder, "right_decoder"):
                r_loss_att, _, _ = self.att_criterion(
                    r_decoder_out, r_ys_out_pad)
                r_loss_att2, _, _ = self.att_criterion(
                    r_decoder_out2, r_ys_out_pad)
                loss_att = loss_att * (1 - self.reverse_weight) + \
                            self.reverse_weight * r_loss_att
                loss_att2 = loss_att2 * (1 - self.reverse_weight) + \
                            self.reverse_weight * r_loss_att2
            loss += (1 - self.ctc_weight) * (loss_att + loss_att2)
            metric += metric_att + metric_att2
            count += count_att + count_att2
        return loss, metric, count
