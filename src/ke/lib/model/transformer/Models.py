from json import decoder
import torch
from torch.autograd import variable
import torch.nn as nn

from ....lib.data.Constants import BOS, EOS, PAD
from .Layers import EncoderLayer, DecoderLayer
from .Embed import Embedder, PositionalEncoder
from .Sublayers import Norm
import copy
import math
from .Batch import create_masks, nopeak_mask
import torch.nn.functional as F
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

    def init_vars(self,src, opt):

        init_tok = BOS
        src_mask = (src != PAD).unsqueeze(-2)
        e_output = self.encoder(src, src_mask)

        outputs = torch.LongTensor([[init_tok]])
        if opt.device == 0:
            outputs = outputs.cuda()

        trg_mask = nopeak_mask(1, opt)

        out = self.out(self.decoder(outputs,
        e_output, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)

        probs, ix = out[:, -1].data.topk(opt.k)
        log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

        outputs = torch.zeros(opt.k, opt.max_len).long()
        if opt.device == 0:
            outputs = outputs.cuda()
        outputs[:, 0] = init_tok
        outputs[:, 1] = ix[0]

        e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
        if opt.device == 0:
            e_outputs = e_outputs.cuda()
        e_outputs[:, :] = e_output[0]

        return outputs, e_outputs, log_scores

    def k_best_outputs(self, outputs, out, log_scores, i, k):

        probs, ix = out[:, -1].data.topk(k)
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
        k_probs, k_ix = log_probs.view(-1).topk(k)

        row = k_ix // k
        col = k_ix % k

        outputs[:, :i] = outputs[row, :i]
        outputs[:, i] = ix[row, col]

        log_scores = k_probs.unsqueeze(0)

        return outputs, log_scores

    def beam_search(self, src, opt):
        outputs, e_outputs, log_scores = self.init_vars(src, opt)
        eos_tok = EOS
        src_mask = (src != PAD).unsqueeze(-2)
        ind = None
        outs = []
        for i in range(2, opt.max_len):

            trg_mask = nopeak_mask(i, opt)

            decoder_out = self.decoder(outputs[:,:i],e_outputs, src_mask, trg_mask)
            out = self.out(decoder_out)
            outs.append(decoder_out)

            out = F.softmax(out, dim=-1)

            outputs, log_scores = self.k_best_outputs(outputs, out, log_scores, i, opt.k)

            ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
            sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
            for vec in ones:
                i = vec[0]
                if sentence_lengths[i]==0: # First end symbol has not been found yet
                    sentence_lengths[i] = vec[1] # Position of first end symbol

            num_finished_sentences = len([s for s in sentence_lengths if s > 0])

            if num_finished_sentences == opt.k:
                alpha = 0.7
                div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        outs = torch.cat(outs, dim=1)[0]
        if ind is None:
            try:
                length = (outputs[0]==eos_tok).nonzero()[0]
            except:
                length = 10
            return outputs[0][1:length], outs

        else:
            try:
                length = (outputs[0]==eos_tok).nonzero()[0]
            except:
                length = 10
            return outputs[ind][1:length], outs

def get_model(opt, src_vocab, trg_vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    if opt.device == 0:
        model = model.cuda()

    return model

class fake_opt(object):
    def __init__(self):
        self.src_pad = PAD
        self.trg_pad = PAD
        self.device = 0
        self.max_len = 20
        self.k = 1

class TransformerWapper(Transformer):
    def __init__(self, *args,**kwargs):
        super().__init__(10000, 10000, 512, 6, 8, 0.1)

    def forward(self, batch, *args,**kwargs):
        src = batch[0][0].t()
        tgt = batch[2].t()
        src_mask, trg_mask = create_masks(src, tgt, fake_opt())

        return super().forward(src, tgt, src_mask, trg_mask)
    def super_forward(self, *args,**kwargs):
        return super().forward(*args,**kwargs)

    # def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
    #     logits = outputs
    #     loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1))
    def translate_sentence(self, sentence, opt = fake_opt()):

        self.eval()
        # indexed = []
        # sentence = SRC.preprocess(sentence)
        # for tok in sentence:
        #     if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
        #         indexed.append(SRC.vocab.stoi[tok])
        #     else:
        #         indexed.append(get_synonym(tok, SRC))
        sentence = sentence.reshape([1]+[i for i in sentence.shape])
        if opt.device == 0:
            sentence = sentence.cuda()

        sentence, outs = self.beam_search(sentence, opt)

        return sentence, outs

    def translate(self, batch, max_length):
        srcs = batch[0][0].t()
        pred = []
        for src in srcs:
            p, _ = self.translate_sentence(src)

            if len(p) > max_length:
                p = p[:max_length]
            elif len(p) < max_length:
                p = torch.cat((p, torch.LongTensor([PAD] *(max_length-len(p))).cuda()), axis=0)


            pred.append(p)
        return torch.stack(pred).t()

    def sample(self, batch, max_length):
        srcs = batch[0][0].t()
        pred = []
        outs = []
        for src in srcs:
            p, o = self.translate_sentence(src)
            if len(p) > max_length:
                p = p[:max_length]
            elif len(p) < max_length:
                p = torch.cat((p, torch.LongTensor([PAD] *(max_length-len(p))).cuda()), axis=0)
            pred.append(p)
            outs.append(o)
        return torch.stack(pred).t(), outs

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        from torch.autograd import Variable
        # pool = torch.nn.MaxPool1d(10000)
        outputs = Variable(torch.cat(outputs, dim=0).data, requires_grad=True) # 8741  *512

        logits = outputs.contiguous().view(-1) if regression else self.out(outputs) #8741 * 10000
        # logits = logits.reshape((logits.shape[0] * logits.shape[1], logits.shape[2]))
        # logits = logits.reshape(logits.shape[:2]).t()
        # logits = logits[:targets.shape[0], :]

        loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1))
        loss.div(normalizer).backward() # calculate gradient
        loss = loss.data.item()

        if outputs.grad is None:
            grad_output = torch.zeros(outputs.size())
        else:
            grad_output = outputs.grad.data
        outputs.backward(grad_output)
        return loss