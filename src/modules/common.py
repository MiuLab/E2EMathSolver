import math
import torch
import pdb


class Encoder(torch.nn.Module):
    """ Simple RNN encoder.

    Args:
        dim_embed (int): Dimension of input embedding.
        dim_hidden (int): Dimension of encoder RNN.
        dim_last (int): Dimension of the last state will be transformed to.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, dim_embed, dim_hidden, dim_last, dropout_rate):
        super(Encoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embed,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 2, dim_last),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Tanh())
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 2, dim_last),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Tanh())

    def forward(self, inputs, lengths):
        """

        Args:
            inputs (tensor): Indices of words. The shape is `B x T x 1`.
            length (list of int): Length of inputs.

        Return:
            outputs (tensor): Encoded sequence. The shape is
                `B x T x dim_hidden`.
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True)
        hidden_state = None
        outputs, hidden_state = self.rnn(packed, hidden_state)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                            batch_first=True)

        # reshape (2, batch, dim_hidden) to (batch, dim_hidden)
        hidden_state = \
            (hidden_state[0].transpose(1, 0).contiguous()
             .view(hidden_state[0].size(1), -1),
             hidden_state[1].transpose(1, 0).contiguous()
             .view(hidden_state[1].size(1), -1))
        hidden_state = \
            (self.mlp1(hidden_state[0]).unsqueeze(0),
             self.mlp2(hidden_state[1]).unsqueeze(0))

        return outputs, hidden_state


class AttnEncoder(torch.nn.Module):
    """ Simple RNN encoder with attention which also extract variable embedding.

    Args:
        dim_embed (int): Dimension of input embedding.
        dim_hidden (int): Dimension of encoder RNN.
        dim_last (int): Dimension of the last state will be transformed to.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, dim_embed, dim_hidden, dim_last, dropout_rate,
                 dim_attn_hidden=256):
        super(AttnEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embed,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 2, dim_last),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Tanh())
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 2, dim_last),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Tanh())
        self.attn = Attention(dim_hidden * 2, dim_hidden * 2,
                              dim_attn_hidden)
        self.embedding_one = torch.nn.Parameter(
            torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.embedding_pi = torch.nn.Parameter(
            torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.register_buffer('padding',
                             torch.zeros(dim_hidden * 2))
        self.embeddings = torch.nn.Parameter(
            torch.normal(torch.zeros(20, 2 * dim_hidden), 0.01))

    def forward(self, inputs, lengths, constant_indices):
        """

        Args:
            inputs (tensor): Indices of words. The shape is `B x T x 1`.
            length (list of int): Length of inputs.
            constant_indices (list of int): Each list contains list

        Return:
            outputs (tensor): Encoded sequence. The shape is
                `B x T x dim_hidden`.
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True)
        hidden_state = None
        outputs, hidden_state = self.rnn(packed, hidden_state)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                            batch_first=True)

        # reshape (2, batch, dim_hidden) to (batch, dim_hidden)
        hidden_state = \
            (hidden_state[0].transpose(1, 0).contiguous()
             .view(hidden_state[0].size(1), -1),
             hidden_state[1].transpose(1, 0).contiguous()
             .view(hidden_state[1].size(1), -1))
        hidden_state = \
            (self.mlp1(hidden_state[0]).unsqueeze(0),
             self.mlp2(hidden_state[1]).unsqueeze(0))

        batch_size = outputs.size(0)
        operands = [[self.embedding_one, self.embedding_pi] +
                    [outputs[b][i]
                     for i in constant_indices[b]]
                    for b in range(batch_size)]
        # operands = [[self.embedding_one, self.embedding_pi] +
        #             [self.embeddings[i]
        #              for i in range(len(constant_indices[b]))]
        #             for b in range(batch_size)]
        # n_operands, operands = pad_and_cat(operands, self.padding)

        # attns = []
        # for i in range(operands.size(1)):
        #     attn = self.attn(outputs, operands[:, i], lengths)
        #     attns.append(attn)

        # operands = [[self.embedding_one, self.embedding_pi]
        #             + [attns[i][b]
        #                for i in range(len(constant_indices[b]))]
        #             for b in range(batch_size)]

        return outputs, hidden_state, operands


class GenVar(torch.nn.Module):
    """ Module to generate variable embedding.

    Args:
        dim_encoder_state (int): Dimension of the last cell state of encoder
            RNN (output of Encoder module).
        dim_context (int): Dimension of RNN in GenVar module.
        dim_attn_hidden (int): Dimension of hidden layer in attention.
        dim_mlp_hiddens (int): Dimension of hidden layers in the MLP
            that transform encoder state to query of attention.
        dropout_rate (int): Dropout rate for attention and MLP.
    """
    def __init__(self, dim_encoder_state, dim_context,
                 dim_attn_hidden=256, dropout_rate=0.5):
        super(GenVar, self).__init__()
        self.attention = Attention(
            dim_context, dim_encoder_state,
            dim_attn_hidden, dropout_rate)

    def forward(self, encoder_state, context, context_lens):
        """ Generate embedding for an unknown variable.

        Args:
            encoder_state (FloatTensor): Last cell state of the encoder
                (output of Encoder module).
            context (FloatTensor): Encoded context, with size
                (batch_size, text_len, dim_hidden).

        Return:
            var_embedding (FloatTensor): Embedding of an unknown variable,
                with size (batch_size, dim_context)
        """
        attn = self.attention(context, encoder_state.squeeze(0), context_lens)
        return attn


class Transformer(torch.nn.Module):
    def __init__(self, dim_hidden):
        super(Transformer, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * dim_hidden, dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.Tanh()
        )
        self.ret = torch.nn.Parameter(torch.zeros(dim_hidden))
        torch.nn.init.normal_(self.ret.data)

    def forward(self, top2):
        return self.mlp(top2)
        # return torch.stack([self.ret] * top2.size(0), 0)


class Attention(torch.nn.Module):
    """ Calculate attention

    Args:
        dim_value (int): Dimension of value.
        dim_query (int): Dimension of query.
        dim_hidden (int): Dimension of hidden layer in attention calculation.
    """
    def __init__(self, dim_value, dim_query,
                 dim_hidden=256, dropout_rate=0.5):
        super(Attention, self).__init__()
        self.relevant_score = \
            MaskedRelevantScore(dim_value, dim_query, dim_hidden)

    def forward(self, value, query, lens):
        """ Generate variable embedding with attention.

        Args:
            query (FloatTensor): Current hidden state, with size
                (batch_size, dim_query).
            value (FloatTensor): Sequence to be attented, with size
                (batch_size, seq_len, dim_value).
            lens (list of int): Lengths of values in a batch.

        Return:
            FloatTensor: Calculated attention, with size
                (batch_size, dim_value).
        """
        relevant_scores = self.relevant_score(value, query, lens)
        e_relevant_scores = torch.exp(relevant_scores)
        weights = e_relevant_scores / e_relevant_scores.sum(-1, keepdim=True)
        attention = (weights.unsqueeze(-1) * value).sum(1)
        return attention


class MaskedRelevantScore(torch.nn.Module):
    """ Relevant score masked by sequence lengths.

    Args:
        dim_value (int): Dimension of value.
        dim_query (int): Dimension of query.
        dim_hidden (int): Dimension of hidden layer in attention calculation.
    """
    def __init__(self, dim_value, dim_query, dim_hidden=256,
                 dropout_rate=0.0):
        super(MaskedRelevantScore, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relevant_score = RelevantScore(dim_value, dim_query,
                                            dim_hidden,
                                            dropout_rate)

    def forward(self, value, query, lens):
        """ Choose candidate from candidates.

        Args:
            query (FloatTensor): Current hidden state, with size
                (batch_size, dim_query).
            value (FloatTensor): Sequence to be attented, with size
                (batch_size, seq_len, dim_value).
            lens (list of int): Lengths of values in a batch.

        Return:
            tensor: Activation for each operand, with size
                (batch, max([len(os) for os in operands])).
        """
        relevant_scores = self.relevant_score(value, query)

        # make mask to mask out padding embeddings
        mask = torch.zeros_like(relevant_scores)
        for b, n_c in enumerate(lens):
            mask[b, n_c:] = -math.inf

        # apply mask
        relevant_scores += mask

        return relevant_scores


class RelevantScore(torch.nn.Module):
    def __init__(self, dim_value, dim_query, hidden1, dropout_rate=0):
        super(RelevantScore, self).__init__()
        self.lW1 = torch.nn.Linear(dim_value, hidden1, bias=False)
        self.lW2 = torch.nn.Linear(dim_query, hidden1, bias=False)
        self.b = torch.nn.Parameter(
            torch.normal(torch.zeros(1, 1, hidden1), 0.01))
        self.tanh = torch.nn.Tanh()
        self.lw = torch.nn.Linear(hidden1, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, value, query):
        """
        Args:
            value (FloatTensor): (batch, seq_len, dim_value).
            query (FloatTensor): (batch, dim_query).
        """
        u = self.tanh(self.dropout(
            self.lW1(value)
            + self.lW2(query).unsqueeze(1)
            + self.b))
        # u.size() == (batch, seq_len, dim_hidden)
        return self.lw(u).squeeze(-1)


def pad_and_cat(tensors, padding):
    """ Pad lists to have same number of elements, and concatenate
    those elements to a 3d tensor.

    Args:
        tensors (list of list of Tensors): Each list contains
            list of operand embeddings. Each operand embedding is of
            size (dim_element,).
        padding (Tensor):
            Element used to pad lists, with size (dim_element,).

    Return:
        n_tensors (list of int): Length of lists in tensors.
        tensors (Tensor): Concatenated tensor after padding the list.
    """
    n_tensors = [len(ts) for ts in tensors]
    pad_size = max(n_tensors)

    # pad to has same number of operands for each problem
    tensors = [ts + (pad_size - len(ts)) * [padding]
               for ts in tensors]

    # tensors.size() = (batch_size, pad_size, dim_hidden)
    tensors = torch.stack([torch.stack(t)
                           for t in tensors], dim=0)

    return n_tensors, tensors
