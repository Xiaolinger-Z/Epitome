import math
import torch

from torch.autograd import Variable
from base_layer import MultiHeadAttention,  PoswiseFeedForwardNet
import torch.nn as nn
from modules import _get_clones

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.layers[i].reset_parameters()

    def forward(self, src,  enc_self_attn_mask = None):
        r"""Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output =  self.layers[i](output, self_attn_mask=enc_self_attn_mask)
        if self.norm:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.layers[i].reset_parameters()

    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_pad_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output, dec_self_attn, dec_enc_attn = self.layers[i](output, memory, self_attn_mask=dec_self_attn_mask,
                                                         enc_attn_mask=dec_enc_attn_pad_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def reset_parameters(self):
        self.enc_self_attn.reset_parameters()
        #self.tgt_emb.reset_parameters()
        self.pos_ffn.reset_parameters()

    def forward(self, enc_inputs, self_attn_mask=None):
        enc_outputs, enc_self_attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                                        enc_inputs, attn_mask=self_attn_mask)

        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.0):
        super(TransformerDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def reset_parameters(self):
        self.dec_self_attn.reset_parameters()
        self.dec_enc_attn.reset_parameters()
        self.pos_ffn.reset_parameters()

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn

class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model
        self.pos = PositionalEncoding(d_model, dropout)

    def reset_parameters(self):
        self.embed.reset_parameters()

    def forward(self, x):
        return self.pos(self.embed(x))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    # noinspection PyArgumentList
    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0)], requires_grad=False)
        return self.dropout(x)



