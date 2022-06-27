import torch
import torch.nn as nn

from model.conformer import Net as ConformerEncoder
from utils.common import add_sos_eos, reverse_pad_list
from layer.att_decoder import TransformerDecoder
from layer.att_decoder import BiTransformerDecoder

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, encoder_conf=None,
                 decoder_type="transformer", decoder_conf=None,
                 reverse_weight=0.0, padding_idx=None):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = ConformerEncoder(input_dim, output_dim, **encoder_conf)
        vocab_size = output_dim
        encoder_embed_dim = encoder_conf['attention_dim']
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.reverse_weight = reverse_weight
        self.padding_idx = padding_idx
        if padding_idx is None:
            self.padding_idx = output_dim
        if decoder_type == "transformer":
            self.decoder = TransformerDecoder(vocab_size, encoder_embed_dim,
                                              **decoder_conf)
        else:
            assert 0.0 < reverse_weight < 1.0
            assert decoder_conf['r_num_blocks'] > 0
            self.decoder = BiTransformerDecoder(vocab_size, encoder_embed_dim,
                                                **decoder_conf)

    def forward(self, data, lens, aed_target, aed_lens):
        out, out_lens, out_embed, out_mask = self.encoder(
                data, lens, output_embed=True)
        # aed branch
        ys_in_pad, ys_out_pad = add_sos_eos(aed_target, self.sos, self.eos,
                                            self.padding_idx)
        ys_in_lens = aed_lens + 1
        # reverse the seq, used for right-to-left decoder
        r_ys_pad = reverse_pad_list(aed_target, aed_lens, float(self.padding_idx))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.padding_idx)
        # forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(out_embed, out_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad, self.reverse_weight)
        return out, out_lens, decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad

    def forward_encoder(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
