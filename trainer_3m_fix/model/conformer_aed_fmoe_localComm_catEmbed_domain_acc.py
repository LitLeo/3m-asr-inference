import torch
import torch.nn as nn
from collections import OrderedDict
import torch.distributed as dist

from model.conformer_fmoe_localComm_catEmbed_domain_acc import Net as ConformerEncoder
###from model.conformer_fmoeExMarc_localComm_catEmbed import Net as ConformerEncoder
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
            ###self.decoder_1 = TransformerDecoder(vocab_size, encoder_embed_dim,
            ###                                  **decoder_conf)
            ###self.decoder_2 = TransformerDecoder(vocab_size, encoder_embed_dim,
            ###                                  **decoder_conf)
        else:
            assert 0.0 < reverse_weight < 1.0
            assert decoder_conf['r_num_blocks'] > 0
            self.decoder = BiTransformerDecoder(vocab_size, encoder_embed_dim,
                                                **decoder_conf)
            ##self.decoder_1 = BiTransformerDecoder(vocab_size, encoder_embed_dim,
            ##                                    **decoder_conf)
            ##self.decoder_2 = BiTransformerDecoder(vocab_size, encoder_embed_dim,
            ##                                    **decoder_conf)

    def init_embed_model(self, load_path):
        return self.encoder.init_embed_model(load_path)

    def forward(self, data, lens, aed_target, aed_lens):
        out, out_lens, aux_embed, aux_loss, out_embed, out_mask = self.encoder(
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
       ## decoder_out_6, r_decoder_out_6, _ = self.decoder_1(out_embed_6, out_mask,
       ##                                              ys_in_pad, ys_in_lens,
       ##                                              r_ys_in_pad, self.reverse_weight)
       ## decoder_out_12, r_decoder_out_12, _ = self.decoder_2(out_embed_12, out_mask,
       ##                                              ys_in_pad, ys_in_lens,
       ##                                              r_ys_in_pad, self.reverse_weight)
        return out, out_lens, decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad, aux_embed, aux_loss

    def forward_encoder(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def state_dict_comm(self):
        local_state_dict = self.state_dict()
        rank = self.encoder.moe_conf['rank']
        world_size = self.encoder.moe_conf['world_size']
        num_experts = self.encoder.moe_conf['num_experts']
        comm = self.encoder.moe_conf['comm']
        if world_size <= 1:
            return local_state_dict
        else:
            new_state_dict = OrderedDict()
            all_experts_num = world_size * num_experts
            for k, v in local_state_dict.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    new_size = list(v.size())
                    new_size[0] = all_experts_num
                    experts_weight = v.data.new_zeros(*new_size)
                    experts_weight[rank * num_experts: (rank + 1) * num_experts] = v
                    dist.all_reduce(experts_weight, group=comm, async_op=False)
                    new_state_dict[k] = experts_weight
            return new_state_dict

    def load_state_dict_comm(self, whole_model_state):
        rank = self.encoder.moe_conf['rank']
        world_size = self.encoder.moe_conf['world_size']
        num_experts = self.encoder.moe_conf['num_experts']
        if world_size <= 1:
            return self.load_state_dict(whole_model_state)
        else:
            ###new_state_dict = OrderedDict()
            new_state_dict = self.state_dict()
            for k, v in whole_model_state.items():
                if "experts" not in k:
                    if k in new_state_dict and v.size()==new_state_dict[k].size():
                        new_state_dict[k] = v
                    ###new_state_dict[k] = v
                else:
                    assert v.size(0) == num_experts * world_size
                    new_state_dict[k] = v[rank * num_experts: (rank + 1) * num_experts]
            return self.load_state_dict(new_state_dict)

