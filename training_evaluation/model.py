from __future__ import print_function
import torch
import operator
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import *
from graph_layers import k_hop_GraphNN
from modules import NormGenerator
from transformer_layers import TransformerEmbedding, TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
from model_util import cal_performance, preprocess_adj
from transformers import BertModel
import random
from sklearn.metrics import accuracy_score
from triplet_loss import TripletLoss

def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask

class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, preseq, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''

        self.prevNode = previousNode
        self.wordid = wordId
        self.preseq = preseq
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=0.7):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class Model(nn.Module):
    def __init__(self, config, node_vocab, target_vocab):
        '''
        Args:
            config: Hyper Parameters
            node_vocab_size: vocabulary size of encoder language
            target_vocab_size: vacabulary size of decoder language
            label_size: optimization level size
        '''
        super(Model, self).__init__()
        self.config = config
        self.target_vocab_size = len(target_vocab)
        self.node_vocab_size = len(node_vocab)
        self.pad_idx = target_vocab.pad_index
        self.sos_index = target_vocab.sos_index
        self.eos_index = target_vocab.eos_index


        d_k = d_v = int(self.config.hidden_dim / self.config.num_heads)
        d_ff = self.config.hidden_dim * 4
        self.pad_idx =0

        # encoder
        pretrain_model_path = self.config.pre_train_model
        self.embedd = BertModel.from_pretrained(pretrain_model_path, add_pooling_layer=False)
        print(self.embedd.config)
        self.embedd.cuda()

        self.dec_embedding = TransformerEmbedding(self.target_vocab_size, self.config.emb_dim, self.config.dropout)
        self.dencoder_proj_layer = nn.Linear(self.config.emb_dim, self.config.hidden_dim)

        encoder_layer = TransformerEncoderLayer(d_k, d_v, self.config.hidden_dim,  d_ff, self.config.num_heads,
                                                self.config.dropout)
        encoder_norm = LayerNorm(self.config.hidden_dim)
        self.encoder = TransformerEncoder(encoder_layer, self.config.num_blocks, encoder_norm)


        decoder_layer = TransformerDecoderLayer(d_k, d_v, self.config.hidden_dim,  d_ff, self.config.num_heads,
                                                self.config.dropout)
        decoder_norm = LayerNorm(self.config.hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, self.config.num_blocks, decoder_norm)

        self.conv_layer = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.config.emb_dim,
                                    out_channels=self.config.conv_feature_dim,
                                    kernel_size=h),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.AvgPool1d(kernel_size=self.config.node_len - h + 1))
            for h in [1, 2, 3, 4]
        ])

        self.ast_encoder = k_hop_GraphNN(self.config.conv_feature_dim*4, self.config.hidden_dim, self.node_vocab_size, self.config.dropout,
                                         self.config.radius).cuda()

        self.encoder_proj_layer = nn.Sequential(nn.Linear((self.config.hidden_dim* 2), self.config.hidden_dim),
                                                nn.GELU(),
                                                nn.Dropout(self.config.dropout))

        self.enc_dropout = nn.Dropout(self.config.dropout)
        self.dec_dropout = nn.Dropout(self.config.dropout)


        self.generator = NormGenerator(self.target_vocab_size, self.config.hidden_dim)

        self.sim_layer = nn.Linear(self.config.hidden_dim, self.config.emb_dim)
        self.sim_drop = nn.Dropout(self.config.dropout)

        #the margin value is depand experiment
        self.loss_fct1 = TripletLoss(margin=0.1)

        self.reset_parameters()


    def reset_parameters(self):
        self.dec_embedding.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.dencoder_proj_layer.reset_parameters()
        self.inst_emb_layer.reset_parameters()
        self.generator.reset_parameters()
        self.sim_layer.reset_parameters()

        for m in self.conv_layer.modules():
            if isinstance(m, nn.Conv1d):
                m.reset_parameters()

        self.ast_encoder.reset_parameters()

        for m in self.encoder_proj_layer.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        #self.proj_layer.reset_parameters()

    def encode(self, input):

        inst, inst_seg, adj, features, segment, idx = input
        if self.config.multi_gpu:
            adj, features, segment = preprocess_adj(adj, features, segment)

            inst = inst.cuda()
            inst_seg = inst_seg.cuda()
            idx = idx.cuda()

        insts_features = list()
        final_insts_features = list()
        cfg_node_features = list()
        final_cfg_node_features = list()
        with torch.no_grad():
            for i, attention_mask in zip(features, segment):
                tmp = self.embedd(input_ids=i, attention_mask=attention_mask)
                result = tmp['last_hidden_state'].detach()
                cfg_node_features.append(result)
                del tmp

        for fea in cfg_node_features:
            embed_node = fea.permute(0, 2, 1)
            enc_nodes = [conv(embed_node) for conv in self.conv_layer]

            enc_node = torch.cat(enc_nodes, dim=1)

            enc_node = enc_node.permute(0, 2, 1)  # embed_node

            final_cfg_node_features.append(enc_node.squeeze())

        ast_outputs = self.ast_encoder(adj, final_cfg_node_features, segment, idx)
        ast_outputs = ast_outputs.unsqueeze(1)
        ast = ast_outputs.expand(ast_outputs.shape[0], self.config.node_num, ast_outputs.shape[2])
        with torch.no_grad():
            for i, attention_mask in zip(inst, inst_seg):
                tmp = self.embedd(input_ids=i, attention_mask=attention_mask)
                result = tmp['last_hidden_state'].detach()
                insts_features.append(result)
                del tmp

        for fea in insts_features:
            embed_inst = fea.permute(0, 2, 1)
            enc_insts= [conv(embed_inst) for conv in self.conv_layer]

            enc_inst = torch.cat(enc_insts, dim=1)

            enc_inst = enc_inst.permute(0, 2, 1)  # embed_node

            final_insts_features.append(enc_inst.squeeze())

        emb_out = torch.stack(final_insts_features, 0)
        emb_out = self.emb_dropout(emb_out)
        emb_out = self.inst_emb_layer(emb_out)
        enc_out = self.encoder(emb_out)

        enc_outputs = torch.cat([enc_out, ast], dim=-1)
        outputs = self.encoder_proj_layer(enc_outputs)

        return outputs

    def decode(self, tgt, memory):

        dec_self_attn_pad_mask = get_attn_pad_mask(tgt, tgt)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(tgt)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # dec_enc_attn_pad_mask = get_attn_pad_mask(tgt, None)
        tgt = self.dec_embedding(tgt)
        tgt = self.dencoder_proj_layer(tgt)

        output = self.decoder(tgt, memory, dec_self_attn_mask=dec_self_attn_mask, dec_enc_attn_pad_mask=None)

        return self.generator(output)

    def decode_sim(self,encoder_output, batch_label=None):

        out1 = torch.mean(encoder_output, dim=1)

        out1 = self.sim_layer(out1)

        if batch_label is not None:
            loss = self.loss_fct1(batch_label, out1)

            return loss
        else:
            return out1


    def evaluate(self, inst, inst_seg, adj, features, segment, idx, select_feature, y):
        bs, _ = y.size()
        if self.config.multi_gpu:
            y = y.cuda()
        decoder_inputs = y[:, :-1]
        encoder_output = self.encode(input)

        if self.config.beam > 0:
            logits = self.beam_decode(decoder_inputs, encoder_output)

            preds = []
            for i in logits:
                preds.append(i[0].data.cpu().numpy()[0])
        else:
            logits = self.decode(decoder_inputs, encoder_output)

            _, preds = torch.max(logits, -1)

            return preds

        return preds

    def beam_decode(self, tgt, encoder_outputs):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        beam_width = self.config.beam
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        # decoding goes sentence by sentence
        for idx in range(tgt.size(0)):
            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)
            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[self.sos_index]]).cuda()
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))
            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(None, decoder_input, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 20: break
                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.preseq

                if n.wordid.item() == self.eos_index and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue
                # output [batch_size, vocab_size]
                # hidden [num_layers * num_directions, batch_size, hidden_size]
                # decode for one step using decoder
                decoder_output = self.decode(decoder_input, encoder_output)
                decoder_output = decoder_output[:, -1, :]
                # PUT HERE REAL BEAM SEARCH OF TOP
                # log_prov, indexes: [batch_size, beam_width] = [1, beam_width]
                decoder_output = F.log_softmax(decoder_output, dim=-1)
                log_prob, indexes = torch.topk(decoder_output, beam_width, dim=-1)
                nextnodes = []
                tmp_beam_width = min(len(log_prob), beam_width)
                for new_k in range(tmp_beam_width):

                    decoded_t = indexes[0][new_k].view(1, -1)
                    preseq_t = torch.cat([decoder_input, decoded_t], dim=-1)
                    # log_p, int
                    log_p = log_prob[0][new_k].item()
                    node = BeamSearchNode(n, decoded_t, preseq_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))
                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1
            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]
            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterances.append(n.preseq)
            decoded_batch.append(utterances)
        return decoded_batch



    def forward(self, inputs, targets, batch_label=None):
        # define decoder inputs
        bs, _ = targets.size()
        if self.config.multi_gpu:
            targets = targets.cuda()
            # targets = targets.cuda()
        decoder_inputs = targets[:, :-1]
        target = targets[:, 1:]

        encoder_output1 = self.encode(inputs)

        logits = self.decode(decoder_inputs, encoder_output1)
        _, preds = torch.max(logits, -1)

        probs = F.softmax(logits, dim=-1).view(-1, self.target_vocab_size)
        istarget = (1. - target.eq(0.).float()).view(-1)
        # Loss
        y_onehot = torch.zeros(logits.size()[0] * logits.size()[1], self.target_vocab_size).cuda()
        y_onehot = Variable(y_onehot.scatter_(1, target.contiguous().view(-1, 1).data, 1))
        y_smoothed = y_onehot

        loss = - torch.sum(y_smoothed * torch.log(probs), dim=-1)
        mean_loss = torch.sum(loss * istarget) / torch.sum(istarget)

        tp_correct, fp_correct, fn_correct = cal_performance(preds, target)

        sim_loss = 0

        if batch_label is not None:
            if self.config.multi_gpu:
                batch_label = batch_label.cuda()
            sim_loss = self.decode_classify(encoder_output1, batch_label)


        final_loss = mean_loss + sim_loss

        return final_loss, preds, tp_correct, fp_correct, fn_correct


    def clip_grad(self, max_norm):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        total_norm = total_norm ** (0.5)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return total_norm
