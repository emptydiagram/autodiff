from functools import reduce
import math
import operator

class ADNode:
    def __init__(self, value, input_nodes=None, label=None, backprop_fn=None):
        self.value = value
        self.input_nodes = set([] if input_nodes is None else input_nodes)
        self.label = label
        self.backprop_fn = (lambda inp: None) if backprop_fn is None else backprop_fn
        self.deriv = 0.

    def __repr__(self):
        return f"ADNode({self.value}, \"{self.label}\", {[node.label for node in self.input_nodes]} | D = {self.deriv})"

    def __add__(self, other):
        bp_fn = lambda x: 1.
        return ADNode(self.value + other.value, (self, other), '+', bp_fn)

    def __mul__(self, other):
        def bp_fn(x):
            inputs = [self, other]
            for inp in inputs:
                if inp != x:
                    return inp.value
            raise Exception('unreachable code')
        return ADNode(self.value * other.value, (self, other), '*', bp_fn)

    def tanh(self):
        # tanh(x) = (e^{2x} - 1)/(e^{2x} + 1)
        tanh_x = (math.exp(2*self.value) - 1.)/(math.exp(2*self.value) + 1)
        # d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)
        bp_fn = lambda x: 1. - tanh_x**2
        return ADNode(tanh_x, (self,), 'tanh', bp_fn)

    def backward(self, deriv=1.):
        self.deriv = deriv
        for inp in self.input_nodes:
            child_deriv = self.deriv * self.backprop_fn(inp)
            inp.backward(child_deriv)

    @classmethod
    def sum(cls, others, label='Σ'):
        return ADNode(sum([o.value for o in others]), others, label)

    @classmethod
    def prod(cls, others, label='Π'):
        product = reduce(operator.mul, [o.value for o in others])
        return ADNode(product, others, label)

# Adapted from example #2 from Karpathy's micrograd video, starting ~52:50 into the video
# tanh neuron, with Autodiff implemented
def ak_example2():
    x1 = ADNode(2., label='x1')
    x2 = ADNode(0., label='x2')
    w1 = ADNode(-3., label='w1')
    w2 = ADNode(1., label='w2')
    b = ADNode(6.8813735870195432, label='b')

    w1x1 = w1 * x1
    w1x1.label = 'w1·x1'
    w2x2 = w2 * x2
    w2x2.label = 'w2·x2'
    w_dot_x = w1x1 + w2x2
    w_dot_x.label = 'w·x'
    w_dot_x_plus_b = w_dot_x + b
    w_dot_x_plus_b.label = 'w·x + b'
    out = w_dot_x_plus_b.tanh()

    print("============ backward ===========")
    out.backward()

    nodes = [w1, x1, w2, x2, b, w1x1, w2x2, w_dot_x, w_dot_x_plus_b, out]
    for node in nodes:
        print(node)


if __name__ == '__main__':
    ak_example2()