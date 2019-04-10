import pdb
import torch
from define import OPERATIONS
from .common import (AttnEncoder, Transformer, Attention,
                     MaskedRelevantScore, pad_and_cat)


class NeuralSolverMachineV1(torch.nn.Module):
    """ Neural Math Word Problem Solver Machine Version 1.

    Args:
        dim_embed (int): Dimension of text embeddings.
        dim_hidden (int): Dimension of encoder decoder hidden state.
    """
    def __init__(self, dim_embed=300, dim_hidden=300, dropout_rate=0.5):
        super(NeuralSolverMachineV1, self).__init__()
        self.encoder = AttnEncoder(dim_embed,
                                   dim_hidden,
                                   dim_hidden,
                                   dropout_rate)
        self.decoder = Decoder(dim_hidden, dropout_rate)
        self.embedding_one = torch.nn.Parameter(
            torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.embedding_pi = torch.nn.Parameter(
            torch.normal(torch.zeros(2 * dim_hidden), 0.01))


class Decoder(torch.nn.Module):
    def __init__(self, dim_hidden=300, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.transformer_add = Transformer(2 * dim_hidden)
        self.transformer_sub = Transformer(2 * dim_hidden)
        self.transformer_mul = Transformer(2 * dim_hidden)
        self.transformer_div = Transformer(2 * dim_hidden)
        self.transformers = {
            OPERATIONS.ADD: self.transformer_add,
            OPERATIONS.SUB: self.transformer_sub,
            OPERATIONS.MUL: self.transformer_mul,
            OPERATIONS.DIV: self.transformer_div,
            OPERATIONS.EQL: None}
        self.gen_var = Attention(2 * dim_hidden,
                                 dim_hidden,
                                 dropout_rate=0.0)
        self.attention = Attention(2 * dim_hidden,
                                   dim_hidden,
                                   dropout_rate=dropout_rate)
        self.choose_arg = MaskedRelevantScore(
            dim_hidden * 2,
            dim_hidden * 7,
            dropout_rate=dropout_rate)
        self.arg_gate = torch.nn.Linear(
            dim_hidden * 7,
            3,
            torch.nn.Sigmoid()
        )
        self.rnn = torch.nn.LSTM(2 * dim_hidden,
                                 dim_hidden,
                                 1,
                                 batch_first=True)
        self.op_selector = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 8))
        self.op_gate = torch.nn.Linear(
            dim_hidden * 7,
            3,
            torch.nn.Sigmoid()
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.register_buffer('noop_padding_return',
                             torch.zeros(dim_hidden * 2))
        self.register_buffer('padding_embedding',
                             torch.zeros(dim_hidden * 2))

    def forward(self, context, text_len, operands, stacks,
                prev_op, prev_output, prev_state):
        """
        Args:
            context (FloatTensor): Encoded context, with size
                (batch_size, text_len, dim_hidden).
            text_len (LongTensor): Text length for each problem in the batch.
            operands (list of FloatTensor): List of operands embeddings for
                each problem in the batch. Each element in the list is of size
                (n_operands, dim_hidden).
            stacks (list of StackMachine): List of stack machines used for each
                problem.
            prev_op (LongTensor): Previous operation, with size (batch, 1).
            prev_arg (LongTensor): Previous argument indices, with size
                (batch, 1). Can be None for the first step.
            prev_output (FloatTensor): Previous decoder RNN outputs, with size
                (batch, dim_hidden). Can be None for the first step.
            prev_state (FloatTensor): Previous decoder RNN state, with size
                (batch, dim_hidden). Can be None for the first step.

        Returns:
            op_logits (FloatTensor): Logits of operation selection.
            arg_logits (FloatTensor): Logits of argument choosing.
            outputs (FloatTensor): Outputs of decoder RNN.
            state (FloatTensor): Hidden state of decoder RNN.
        """
        batch_size = context.size(0)

        # collect stack states
        stack_states = \
            torch.stack([stack.get_top2().view(-1,) for stack in stacks],
                        dim=0)

        # skip the first step (all NOOP)
        if prev_output is not None:
            # result calculated batch-wise
            batch_result = {
                OPERATIONS.GEN_VAR: self.gen_var(
                    context, prev_output, text_len),
                OPERATIONS.ADD: self.transformer_add(stack_states),
                OPERATIONS.SUB: self.transformer_sub(stack_states),
                OPERATIONS.MUL: self.transformer_mul(stack_states),
                OPERATIONS.DIV: self.transformer_div(stack_states)
            }

        prev_returns = []
        # apply previous op on stacks
        for b in range(batch_size):
            # no op
            if prev_op[b].item() == OPERATIONS.NOOP:
                ret = self.noop_padding_return

            # generate variable
            elif prev_op[b].item() == OPERATIONS.GEN_VAR:
                variable = batch_result[OPERATIONS.GEN_VAR][b]
                operands[b].append(variable)
                stacks[b].add_variable(variable)
                ret = variable

            # OPERATIONS.ADD, SUB, MUL, DIV
            elif prev_op[b].item() in [OPERATIONS.ADD, OPERATIONS.SUB,
                                       OPERATIONS.MUL, OPERATIONS.DIV]:
                transformed = batch_result[prev_op[b].item()][b]
                ret = stacks[b].apply(
                    prev_op[b].item(),
                    transformed)

            elif prev_op[b].item() == OPERATIONS.EQL:
                ret = stacks[b].apply(prev_op[b].item(), None)

            # push operand
            else:
                stacks[b].push(prev_op[b].item() - OPERATIONS.N_OPS)
                ret = operands[b][prev_op[b].item() - OPERATIONS.N_OPS]
            prev_returns.append(ret)

        # collect stack states (after applied op)
        stack_states = \
            torch.stack([stack.get_top2().view(-1,) for stack in stacks],
                        dim=0)

        # collect previous returns
        prev_returns = torch.stack(prev_returns)
        prev_returns = self.dropout(prev_returns)

        # decode
        outputs, hidden_state = self.rnn(prev_returns.unsqueeze(1),
                                         prev_state)
        outputs = outputs.squeeze(1)

        # attention
        attention = self.attention(context, outputs, text_len)

        # collect information for op selector
        gate_in = torch.cat([outputs, stack_states, attention], -1)
        op_gate_in = self.dropout(gate_in)
        op_gate = self.op_gate(op_gate_in)
        arg_gate_in = self.dropout(gate_in)
        arg_gate = self.arg_gate(arg_gate_in)
        op_in = torch.cat([op_gate[:, 0:1] * outputs,
                           op_gate[:, 1:2] * stack_states,
                           op_gate[:, 2:3] * attention], -1)
        arg_in = torch.cat([arg_gate[:, 0:1] * outputs,
                            arg_gate[:, 1:2] * stack_states,
                            arg_gate[:, 2:3] * attention], -1)
        # op_in = arg_in = torch.cat([outputs, stack_states, attention], -1)

        op_logits = self.op_selector(op_in)

        n_operands, cated_operands = \
            pad_and_cat(operands, self.padding_embedding)
        arg_logits = self.choose_arg(
            cated_operands, arg_in, n_operands)

        return op_logits, arg_logits, outputs, hidden_state
