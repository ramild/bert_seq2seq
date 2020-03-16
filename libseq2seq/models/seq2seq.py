import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models


class seq2seq(nn.Module):
    def __init__(
        self,
        config,
        src_vocab_size,
        tgt_vocab_size,
        use_cuda,
        score_fn=None,
    ):
        super(seq2seq, self).__init__()
        self.encoder = models.rnn_encoder(config, src_vocab_size)
        self.decoder = models.rnn_decoder(
            config,
            tgt_vocab_size,
            score_fn=score_fn,
        )
        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.criterion = models.criterion(tgt_vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax()

    def compute_loss(self, hidden_outputs, targets):
        return models.cross_entropy_loss(
            hidden_outputs,
            self.decoder,
            targets,
            self.criterion,
            self.config,
        )

    def forward(self, src, src_len, tgt, tgt_len):
        lengths, indices = torch.sort(
            src_len.squeeze(0),
            dim=0,
            descending=True,
        )
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)
        contexts, state = self.encoder(src, lengths.data.tolist())
        outputs, final_state = self.decoder(
            tgt[:-1],
            state,
            contexts.transpose(0, 1),
        )
        return outputs, tgt[1:]

    def beam_sample(self, src, src_len, beam_size=1):
        batch_size = src.size(1)
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(
            torch.index_select(src, dim=1, index=indices),
            volatile=True,
        )
        contexts, encState = self.encoder(src, lengths.tolist())

        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        contexts = rvar(contexts.transpose(0, 1).data).transpose(0, 1)
        decState = (rvar(encState[0].data), rvar(encState[1].data))
        beam = [
            models.Beam(beam_size, n_best=1, cuda=self.use_cuda)
            for _ in range(batch_size)
        ]
        mask = None
        soft_score = None
        for i in range(self.config.max_tgt_len):
            if all((b.done() for b in beam)):
                break
            inp = var(
                torch.stack([b.getCurrentState() for b in beam])
                .t()
                .contiguous()
                .view(-1),
            )
            output, decState, attn = self.decoder.sample_one(
                inp,
                soft_score,
                decState,
                contexts,
                mask,
            )
            soft_score = F.softmax(output)
            predicted = output.max(1)[1]
            if self.config.mask:
                if mask is None:
                    mask = predicted.unsqueeze(1).long()
                else:
                    mask = torch.cat((mask, predicted.unsqueeze(1)), 1)
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        allHyps, allScores, allAttn = [], [], []
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        return allHyps, allAttn
