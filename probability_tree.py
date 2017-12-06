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

prop_boys = 1.09
prop_girls = 1.0
p_boy = prop_boys/(prop_boys + prop_girls)
# p_girl = prop_girls/(prop_boys + prop_girls)


class Node:
    def __init__(self, p, qty, parent):
        self.parent = parent
        self.p = p      # probability of being in this situation; 1 at head
        self.qty = qty  # number of successes including this node

    def add_children(self, probability):
        self.children = [Node(self.p*probability, self.qty + 1, self),
                         Node(self.p*(1-probability), self.qty, self)]


class ProbTree:
    def __init__(self, probability, depth, criterion):
        self.p = probability
        self.head = None
        self.criterion = criterion
        count = self.build(depth)
        print('{:.3f}'.format(count))

    def _build(self, node_list, depth_to_go):
        layer = []
        for node in node_list:
            node.add_children(self.p)
            layer += node.children
        if depth_to_go:
            return self._build(layer, depth_to_go - 1)
        else:
            return sum([node.p for node in layer if node.qty >= self.criterion])

    def build(self, depth):
        self.head = Node(1, 0, None)
        return self._build([self.head], depth-1)


if __name__ == '__main__':
    tree = ProbTree(p_boy, 6, 3)
