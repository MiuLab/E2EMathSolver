import torch
from define import OPERATIONS


class TreeIterator:
    def __init__(self, root):
        self.root = root
        self.queue = [self.root]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.queue) == 0:
            raise StopIteration
        else:
            node = self.queue.pop()
            op, left, right = node
            if type(left) is list:
                self.queue.insert(0, left)
            if type(right) is list:
                self.queue.insert(0, right)

            return node


def build_tree(stack_ops):
    stack = []
    for op in stack_ops:
        if op in [OPERATIONS.NOOP]:
            continue
        if op >= OPERATIONS.N_OPS:
            stack.append(op)
        else:
            right = stack.pop()
            left = stack.pop()
            stack.append([op, left, right])

    return stack[0]


def tree_to_op(node):
    if type(node) is not list:
        return [node]
    else:
        op, left, right = node
        return tree_to_op(left) + tree_to_op(right) + [op]


OP_INV_MAP = {
    OPERATIONS.ADD: OPERATIONS.SUB,
    OPERATIONS.SUB: OPERATIONS.ADD,
    OPERATIONS.MUL: OPERATIONS.DIV,
    OPERATIONS.DIV: OPERATIONS.MUL
}


def permute_stack_ops(stack_ops, revert_prob=0.25, transpose_prob=0.5):
    rands = torch.rand(len(stack_ops))
    tree = build_tree(stack_ops[1:])
    print(rands)

    # revert operands for ADD and MUL
    tree_it = TreeIterator(tree)
    for node, rand in zip(tree_it, rands):
        if rand < revert_prob:
            op, left, right = node
            if op in [OPERATIONS.ADD, OPERATIONS.MUL]:
                node[1], node[2] = right, left

    # transposition
    eq, left, right = tree
    print('right:', right)
    if rands[-1] < transpose_prob and type(right) is list:
        print('do trans')
        rop, rright, rleft = right
        left = [OP_INV_MAP[rop], left, rleft]
        right = rright
        tree = [eq, left, right]

    return [OPERATIONS.GEN_VAR] + tree_to_op(tree)


class PermuteStackOps(object):
    def __init__(self, revert_prob=0.25, transpose_prob=0.5):
        self.revert_prob = revert_prob
        self.transpose_prob = transpose_prob

    def __call__(self, sample):
        sample['operations'] = permute_stack_ops(sample['operations'],
                                                 self.revert_prob,
                                                 self.transpose_prob)
        return sample
