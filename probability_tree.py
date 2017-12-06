'''
Not machine learning, but is brute-force solution to a statistics problem
so I threw it in here. From HackerRank 10-days-of-statistics challenge.
Bernouli trial built as a tree, because 1. it is a tree and 2. we have
computers so who needs formulae?

Quoted from HackerRank:

The ratio of boys to girls for babies born in Russia is 1.09. If there is
1 child born per birth, what proportion of Russian families with exactly
6 children will have at least 3  boys?

Write a program to compute the answer using the above parameters. Then
print your result, rounded to a scale of 3 decimal places (e.g., 1.234).
'''
from collections import defaultdict
from timeit import default_timer
from math import isclose

prop_boys = 1.09
prop_girls = 1.0
p_boy = prop_boys/(prop_boys + prop_girls)
# p_girl = prop_girls/(prop_boys + prop_girls)


# first guess: heavyweight tree, stores all layers and parent/child status

class Node:
    def __init__(self, p, qty, parent):
        self.parent = parent
        self.p = p      # probability of being in this situation; 1 at head
        self.qty = qty  # number of successes including this node

    def add_children(self, probability):
        self.children = [Node(self.p*probability, self.qty + 1, self),
                         Node(self.p*(1-probability), self.qty, self)]

    def __repr__(self):
        return 'node: p={:.2f}, qty={}'.format(self.p, self.qty)


class ProbTree:
    def __init__(self, probability, depth, criterion):
        self.p = probability
        self.head = None
        self.criterion = criterion

        self.count = self.build(depth)

    def _build(self, node_list, depth_to_go):
        layer = []
        # print(depth_to_go, node_list)
        for node in node_list:
            node.add_children(self.p)
            layer += node.children
        if depth_to_go:
            return self._build(layer, depth_to_go - 1)
        else:
            return sum([node.p for node in layer if self.criterion(node.qty)])

    def build(self, depth):
        self.head = Node(1, 0, None)
        return self._build([self.head], depth-1)

    def get_count(self):
        return self.count


# refactor to minimal version; eliminates duplicate values (i.e. tree grows by 1 each level)
# and only stores current layer.

def build_tree(p, depth, criterion):
    layer = {0: 1}   # layer maps number of positives so far to probability of occurrence
    for i in range(depth):
        new = defaultdict(int)
        for k, v in layer.items():
            new[k] += v * (1-p)
            new[k+1] += v * p
        layer = new

    return sum([val for key, val in layer.items() if criterion(key)])


# additionally cuts off branches when they meet criterion. Should be marginally faster?

def build_pruned(p, depth, criterion):
    layer = {0: 1}   # layer maps number of positives so far to probability of occurrence
    running_sum = 0
    for i in range(depth):
        new = defaultdict(int)
        for key, val in layer.items():
            # we know key does not meet criterion
            new[key] += val * (1-p)

            # we have not tested key+1
            if criterion(key+1):
                running_sum += val * p
            else:
                new[key+1] += val * p
        layer = new

    return running_sum


if __name__ == '__main__':
    print('test functions are equivalent...')
    tree = ProbTree(p_boy, 6, lambda x: x >= 3).get_count()
    alt1 = build_tree(p_boy, 6, lambda x: x >= 3)
    alt2 = build_pruned(p_boy, 6, lambda x: x >= 3)

    assert isclose(tree, alt1)
    assert isclose(tree, alt2)

    print('OK')
    print()

    print('Tree class; d = 6:')
    start = default_timer()
    tree = ProbTree(p_boy, 6, lambda x: x >= 3).get_count()
    print(default_timer() - start)

    print('Tree class; d = 20:')
    start = default_timer()
    tree = ProbTree(p_boy, 20, lambda x: x >= 10).get_count()
    print(default_timer() - start)

    functions = [build_tree, build_pruned]
    depths = [20, 30, 40]
    criteria = [10, 20]

    for funct in functions:
        for d in depths:
            for crit in criteria:
                print()
                print('function={} d={} cutoff={}'.format(funct, d, crit))
                start = default_timer()
                tree = funct(p_boy, d, lambda x: x >= crit)
                print(default_timer() - start)
