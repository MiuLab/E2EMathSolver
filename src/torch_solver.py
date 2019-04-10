import math
import pdb
import torch
from define import OPERATIONS
from pytorch_base import TorchBase
from modules.neural_solver_machine_v1 import NeuralSolverMachineV1
from stack_machine import StackMachine


class TorchSolver(TorchBase):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """
    def __init__(self, dim_embed, dim_hidden,
                 decoder_use_state=True, **kwargs):
        super(TorchSolver, self).__init__(**kwargs)
        self._dim_embed = dim_embed
        self._dim_hidden = dim_hidden
        self.use_state = decoder_use_state
        self._model = NeuralSolverMachineV1(dim_embed, dim_hidden, 0.1)

        # make class weights to ignore loss for padding operations
        class_weights = torch.ones(8)
        class_weights[OPERATIONS.NOOP] = 0

        # use cuda
        class_weights = class_weights.to(self._device)
        self._model = self._model.to(self._device)

        # make loss
        self._op_loss = torch.nn.CrossEntropyLoss(class_weights,
                                                  size_average=False,
                                                  reduce=False)
        self._arg_loss = torch.nn.CrossEntropyLoss()

        # make optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=self._learning_rate)

    def _run_iter(self, batch, training):
        order = torch.sort(batch['text_len'] * -1)[1]
        for k in batch:
            if type(batch[k]) is list:
                batch[k] = [batch[k][i] for i in order]
            else:
                batch[k] = batch[k][order]
        batch_size = len(order)

        # zero embedding for the stack bottom
        bottom = torch.zeros(self._dim_hidden * 2)
        bottom.requires_grad = False

        # deal with device
        text, ops, bottom = \
            batch['text'].to(self._device), \
            batch['operations'].to(self._device), \
            bottom.to(self._device)

        # encode
        context, state, operands = \
            self._model.encoder.forward(text, batch['text_len'],
                                        batch['constant_indices'])

        # extract constant embeddings
        # operands = [[self._model.embedding_one, self._model.embedding_pi] +
        #             [context[b][i]
        #              for i in batch['constant_indices'][b]]
        #             for b in range(batch_size)]

        # initialize stacks
        stacks = [StackMachine(batch['constants'][b], operands[b], bottom,
                  dry_run=True)
                  for b in range(batch_size)]

        loss = torch.zeros(batch_size).to(self._device)
        prev_op = torch.zeros(batch_size).to(self._device)
        prev_output = None

        if self.use_state:
            prev_state = state
        else:
            prev_state = None
        for t in range(max(batch['op_len'])):
            # step one
            op_logits, arg_logits, prev_output, prev_state = \
                self._model.decoder(
                    context, batch['text_len'], operands, stacks,
                    prev_op, prev_output, prev_state)

            # accumulate op loss
            op_target = torch.tensor(ops[:, t])
            op_target[op_target >= OPERATIONS.N_OPS] = OPERATIONS.N_OPS
            op_target.require_grad = False
            loss += self._op_loss(op_logits, torch.tensor(op_target))

            # accumulate arg loss
            for b in range(batch_size):
                if ops[b, t] < OPERATIONS.N_OPS:
                    continue

                loss[b] += self._arg_loss(
                    arg_logits[b].unsqueeze(0),
                    ops[b, t].unsqueeze(0) - OPERATIONS.N_OPS)

            prev_op = ops[:, t]

        # if training:
        #     weights = 1 / torch.tensor(batch['op_len']).to(self._device).float()
        # else:
        weights = 1

        loss = (loss * weights).mean()
        predicts = [stack.get_solution() for stack in stacks]

        return predicts, loss

    def _predict_batch(self, batch, max_len=30):
        order = torch.sort(batch['text_len'] * -1)[1]
        for k in batch:
            if type(batch[k]) is list:
                batch[k] = [batch[k][i] for i in order]
            else:
                batch[k] = batch[k][order]
        batch_size = len(order)

        # for constants, cindices, operations in zip(batch['constants'],
        #                                            batch['constant_indices'],
        #                                            batch['operations']):
        #     used = set()
        #     for op in operations:
        #         if op >= OPERATIONS.N_OPS:
        #             used.add(op - OPERATIONS.N_OPS)

        #     for i in range(len(cindices) - 1, -1, -1):
        #         if i not in used:
        #             del constants[i + 2]
        #             del cindices[i]

        # zero embedding for the stack bottom
        bottom = torch.zeros(self._dim_hidden * 2)
        bottom.requires_grad = False

        # deal with device
        text, bottom = \
            batch['text'].to(self._device), \
            bottom.to(self._device)

        # encode
        context, state, operands = \
            self._model.encoder.forward(text, batch['text_len'],
                                        batch['constant_indices'])

        # extract constant embeddings
        # operands = [[self._model.embedding_one, self._model.embedding_pi]
        #             + [context[b][i]
        #                for i in batch['constant_indices'][b]]
        #             for b in range(batch_size)]

        # initialize stacks
        stacks = [StackMachine(batch['constants'][b], operands[b], bottom)
                  for b in range(batch_size)]

        loss = torch.zeros(batch_size).to(self._device)
        prev_op = torch.zeros(batch_size).to(self._device)
        prev_output = None
        prev_state = state
        finished = [False] * batch_size
        for t in range(40):
            op_logits, arg_logits, prev_output, prev_state = \
                self._model.decoder(
                    context, batch['text_len'], operands, stacks,
                    prev_op, prev_output, prev_state)

            n_finished = 0
            for b in range(batch_size):
                if stacks[b].get_solution() is not None:
                    finished[b] = True

                if finished[b]:
                    op_logits[b, OPERATIONS.NOOP] = math.inf
                    n_finished += 1

                if stacks[b].get_height() < 2:
                    op_logits[b, OPERATIONS.ADD] = -math.inf
                    op_logits[b, OPERATIONS.SUB] = -math.inf
                    op_logits[b, OPERATIONS.MUL] = -math.inf
                    op_logits[b, OPERATIONS.DIV] = -math.inf
                    op_logits[b, OPERATIONS.EQL] = -math.inf

            op_loss, prev_op = torch.log(
                torch.nn.functional.softmax(op_logits, -1)
            ).max(-1)
            arg_loss, prev_arg = torch.log(
                torch.nn.functional.softmax(arg_logits, -1)
            ).max(-1)

            for b in range(batch_size):
                if prev_op[b] == OPERATIONS.N_OPS:
                    prev_op[b] += prev_arg[b]
                    loss[b] += arg_loss[b]

                if prev_op[b] != OPERATIONS.NOOP:
                    loss[b] += op_loss[b]

            if n_finished == batch_size:
                break

        predicts = [None] * batch_size
        for i, o in enumerate(order):
            predicts[o] = {
                'ans': stacks[i].get_solution(),
                'equations': stacks[i].stack_log,
                'confidence': loss[i].item()
            }

        return predicts
