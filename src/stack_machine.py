import sympy
import torch
import logging
from define import OPERATIONS


class StackMachine:
    """

    Args:
        constants (list): Value of numbers.
        embeddings (tensor): Tensor of shape [len(constants), dim_embedding].
            Embedding of the constants.
        bottom_embedding (teonsor): Tensor of shape (dim_embedding,). The
            embeding to return when stack is empty.
    """
    def __init__(self, constants, embeddings, bottom_embedding, dry_run=False):
        self._operands = list(constants)
        self._embeddings = [embedding for embedding in embeddings]

        # number of unknown variables
        self._n_nuknown = 0

        # stack which stores (val, embed) tuples
        self._stack = []

        # equations got from applying `=` on the stack
        self._equations = []
        self.stack_log = []

        # functions operate on value
        self._val_funcs = {
            OPERATIONS.ADD: sympy.Add,
            OPERATIONS.SUB: lambda a, b: sympy.Add(-a, b),
            OPERATIONS.MUL: sympy.Mul,
            OPERATIONS.DIV: lambda a, b: sympy.Mul(1/a, b)
        }
        self._op_chars = {
            OPERATIONS.ADD: '+',
            OPERATIONS.SUB: '-',
            OPERATIONS.MUL: '*',
            OPERATIONS.DIV: '/',
            OPERATIONS.EQL: '='
        }

        self._bottom_embed = bottom_embedding

        if dry_run:
            self.apply = self.apply_embed_only

    def add_variable(self, embedding):
        """ Tell the stack machine to increase the number of nuknown variables
            by 1.

        Args:
            embedding (tensor): Tensor of shape (dim_embedding). Embedding
                of the unknown varialbe.
        """
        var = sympy.Symbol('x{}'.format(self._n_nuknown))
        self._operands.append(var)
        self._embeddings.append(embedding)
        self._n_nuknown += 1

    def push(self, operand_index):
        """ Push var to stack.

        Args:
            operand_index (int): Index of the operand. If
                index >= number of constants, then it implies a variable is
                pushed.
        Return:
            tensor: Simply return the pushed embedding.
        """
        self._stack.append((self._operands[operand_index],
                            self._embeddings[operand_index]))
        self.stack_log.append(str(self._operands[operand_index]))
        return self._embeddings[operand_index]

    def apply(self, operation, embed_res):
        """ Apply operator on stack.

        Args:
            operator (OPERATION): One of
                - OPERATIONS.ADD
                - OPERATIONS.SUB
                - OPERATIONS.MUL
                - OPERATIONS.DIV
                - OPERATIONS.EQL
            embed_res (FloatTensor): Resulted embedding after transformation,
                with size (dim_embedding,).
        Return:
            tensor: embeding on the top of the stack.
        """
        val1, embed1 = self._stack.pop()
        val2, embed2 = self._stack.pop()
        if operation != OPERATIONS.EQL:
            try:
                # calcuate values in the equation
                val_res = self._val_funcs[operation](val1, val2)
                # transform embedding
                self._stack.append((val_res, embed_res))
            except ZeroDivisionError:
                logging.warn('WARNING: zero division error, skip operation')
        else:
            self._equations.append(val1 - val2)
            # pass
        self.stack_log.append(self._op_chars[operation])

        if len(self._stack) > 0:
            return self._stack[-1][1]
        else:
            return self._bottom_embed

    def apply_embed_only(self, operation, embed_res):
        """ Apply operator on stack with embedding operation only.

        Args:
            operator (OPERATION): One of
                - OPERATIONS.ADD
                - OPERATIONS.SUB
                - OPERATIONS.MUL
                - OPERATIONS.DIV
                - OPERATIONS.EQL
            embed_res (FloatTensor): Resulted embedding after transformation,
                with size (dim_embedding,).
        Return:
            tensor: embeding on the top of the stack.
        """
        val1, embed1 = self._stack.pop()
        val2, embed2 = self._stack.pop()
        if operation != OPERATIONS.EQL:
            # calcuate values in the equation
            val_res = None
            # transform embedding
            self._stack.append((val_res, embed_res))

        if len(self._stack) > 0:
            return self._stack[-1][1]
        else:
            return self._bottom_embed

    def get_solution(self):
        """ Get solution. If the problem has not been solved, return None.

        Return:
            list: If the problem has been solved, return result from
                sympy.solve. If not, return None.
        """
        if self._n_nuknown == 0:
            return None

        root = sympy.solve(self._equations)
        for i in range(self._n_nuknown):
            if self._operands[-i - 1] not in root:
                return None

        return root

    def get_top2(self):
        """ Get the top 2 embeddings of the stack.

        Return:
            tensor: Return tensor of shape (2, embed_dim)
        """
        if len(self._stack) >= 2:
            return torch.stack([self._stack[-1][1],
                                self._stack[-2][1]], dim=0)
        elif len(self._stack) == 1:
            return torch.stack([self._stack[-1][1],
                                self._bottom_embed], dim=0)
        else:
            return torch.stack([self._bottom_embed,
                                self._bottom_embed], dim=0)

    def get_height(self):
        """ Get the height of the stack.

        Return:
            int: height
        """
        return len(self._stack)

    def get_stack(self):
        return [self._bottom_embed] + [s[1] for s in self._stack]
