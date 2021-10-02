import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torch.nn.functional import log_softmax
from .tools import batch_sequences
from .modules.state import State
from ...data.Constants import UNK, PAD, EOS, BOS
from .tools.beam_search import SequenceGenerator


class Seq2Seq(nn.Module):

    def __init__(self, encoder=None, decoder=None, bridge=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pool = torch.nn.MaxPool1d(10000)

        if bridge is not None:
            self.bridge = bridge

    def bridge(self, context):
        return State(context=context,
                     batch_first=getattr(self.decoder, 'batch_first', context.batch_first))

    def encode(self, inputs, hidden=None, device_ids=None):
        if isinstance(device_ids, tuple):
            return data_parallel(self.encoder, (inputs, hidden),
                                 device_ids=device_ids,
                                 dim=0 if self.encoder.batch_first else 1)
        else:
            return self.encoder(inputs, hidden)

    def decode(self, *kargs, **kwargs):
        device_ids = kwargs.pop('device_ids', None)
        if isinstance(device_ids, tuple):
            return data_parallel(self.decoder, *kargs, **kwargs,
                                 device_ids=device_ids,
                                 dim=0 if self.decoder.batch_first else 1)
        else:
            return self.decoder(*kargs, **kwargs)

    def forward_old(self, input_encoder, input_decoder, encoder_hidden=None, device_ids=None):
        if not isinstance(device_ids, dict):
            device_ids = {'encoder': device_ids, 'decoder': device_ids}
        context = self.encode(input_encoder, encoder_hidden,
                              device_ids=device_ids.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        output, state = self.decode(
            input_decoder, state, device_ids=device_ids.get('decoder', None))
        return output

    def forward(self, inputs, eval=None, regression=False):
        src = inputs[0][0]
        tgt = inputs[2]
        outputs =  self.forward_old(src, tgt)
        if regression:
            outputs = self.pool(outputs)
            outputs = outputs.reshape(outputs.shape[:2])
        return outputs

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        if regression:
            logits = outputs.view(-1)
            loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1))
        else:
            loss = criterion(outputs.reshape((outputs.shape[0] * outputs.shape[1], outputs.shape[2])), targets.contiguous().view(-1), weights.contiguous().view(-1))
        loss.div(normalizer).backward() # calculate gradient
        # for name, parms in self.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
        #     ' -->grad_value:',parms.grad.abs().max())
        return loss

    def predict(self, outputs, targets, weights, criterion):
        logits = outputs.reshape((outputs.shape[0] * outputs.shape[1], outputs.shape[2]))
        preds = logits.data.max(1)[1].view(outputs.size(0), -1)

        loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1)).data.item()

        return preds, loss

    def translate(self, inputs, max_length):
        src = inputs[0][0]
        bos = [[BOS]] * inputs[0][0].shape[1]
        output = self.generate(src, bos, beam_size=1, get_attention=False, max_sequence_length=max_length)
        output = [i.output for i in output]
        output = [i + ([PAD] * (max_length - len(i))) for i in output]
        return torch.LongTensor(output).t()

    def sample(self, inputs, max_length):
        return self.translate(inputs, max_length), self.forward(inputs)
        # import torch.nn.functional as F

        # outputs = self.forward(inputs) # 120 * 50 * 10000
        # samples = outputs.max(2)[1]
        # for i in range(samples.shape[-1]):
        #     for j in range(samples.shape[0]):
        #         if samples[j][i] == EOS:
        #             samples[j+1:, i] = PAD
        #             break

        # for i in range(max_length):
        #     logit = outputs[:, i, :]

        #     dist = F.softmax(logit, dim=-1)
        #     sample = dist.multinomial(1, replacement=False).view(-1).data

        #     samples.append(sample)

        # samples = torch.stack(samples)
        return samples, outputs


    def _decode_step(self, input_list, state_list, args_dict={},
                     k=1,
                     feed_all_timesteps=False,
                     keep_all_timesteps=False,
                     time_offset=0,
                     time_multiply=1,
                     apply_lsm=True,
                     remove_unknown=False,
                     get_attention=False,
                     device_ids=None):

        view_shape = (-1, 1) if self.decoder.batch_first else (1, -1)
        time_dim = 1 if self.decoder.batch_first else 0
        device = next(self.decoder.parameters()).device

        # For recurrent models, the last input frame is all we care about,
        # use feed_all_timesteps whenever the whole input needs to be fed
        if feed_all_timesteps:
            inputs = [torch.tensor(inp, device=device, dtype=torch.long)
                      for inp in input_list]
            inputs = batch_sequences(
                inputs, device=device, batch_first=self.decoder.batch_first)[0]

        else:
            last_tokens = [inputs[-1] for inputs in input_list]
            inputs = torch.stack(last_tokens).view(*view_shape)

        states = State().from_list(state_list)
        decode_inputs = dict(get_attention=get_attention,
                             device_ids=device_ids, **args_dict)
        if time_multiply > 1:
            decode_inputs['time_multiply'] = time_multiply
        logits, new_states = self.decode(inputs, states, **decode_inputs)

        if not keep_all_timesteps:
            # use only last prediction
            logits = logits.select(time_dim, -1).contiguous()
        if remove_unknown:
            # Remove possibility of unknown
            logits[:, UNK].fill_(-float('inf'))
        if apply_lsm:
            logprobs = log_softmax(logits, dim=-1)
        else:
            logprobs = logits
        logprobs, words = logprobs.topk(k, dim=-1)
        new_states_list = [new_states[i] for i in range(len(input_list))]
        return words, logprobs, new_states_list

    def generate(self, input_encoder, input_decoder, beam_size=None,
                 max_sequence_length=None, length_normalization_factor=0,
                 get_attention=False, device_ids=None, autoregressive=True):
        if not isinstance(device_ids, dict):
            device_ids = {'encoder': device_ids, 'decoder': device_ids}
        context = self.encode(input_encoder,
                              device_ids=device_ids.get('encoder', None))
        if hasattr(self, 'bridge'):
            state = self.bridge(context)
        state_list = state.as_list()
        params = dict(decode_step=self._decode_step,
                      beam_size=beam_size,
                      max_sequence_length=max_sequence_length,
                      get_attention=get_attention,
                      length_normalization_factor=length_normalization_factor,
                      device_ids=device_ids.get('encoder', None))
        generator = SequenceGenerator(**params)
        return generator.beam_search(input_decoder, state_list)
