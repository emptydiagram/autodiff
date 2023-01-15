from functools import reduce
import math
import operator

class ADNode:
    def __init__(self, value, input_nodes=None, label=None, partial_deriv=None):
        self.value = value
        nodes = {}
        if input_nodes is not None:
            for in_node in input_nodes:
                if in_node in nodes:
                    nodes[in_node] += 1
                else:
                    nodes[in_node] = 1
        self.input_nodes = nodes
        self.label = label
        # Given a node computing some function f: (x_1, ..., x_n) |-> y
        # The partial_deriv function computes (x_i) |-> (the partial derivative df/dx_i)
        self.partial_deriv = (lambda inp: None) if partial_deriv is None else partial_deriv
        self.deriv = 0.

    def __repr__(self):
        return f"ADNode({self.value}, \"{self.label}\", {[node.label for node in self.input_nodes]} | D = {self.deriv})"

    def __neg__(self):
        return ADNode(-self.value, (self,), '-( )', lambda inp: -1)

    def __add__(self, other):
        if not(isinstance(other, ADNode)):
            other = ADNode(other, label=f"{other}")

        partial_deriv = lambda x: 1.
        return ADNode(self.value + other.value, (self, other), '+', partial_deriv)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not(isinstance(other, ADNode)):
            other = ADNode(other, label=f"{other}")

        partial_deriv = lambda x: 1. if x == self else -1.
        return ADNode(self.value + other.value, (self, other), '-', partial_deriv)

    def __mul__(self, other):
        if not(isinstance(other, ADNode)):
            other = ADNode(other, label=f"{other}")

        def partial_deriv(x):
            inputs = [self, other]
            for inp in inputs:
                if inp != x:
                    return inp.value
            raise Exception('unreachable code')
        return ADNode(self.value * other.value, (self, other), '*', partial_deriv)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not(isinstance(other, ADNode)):
            other = ADNode(other, label=f"{other}")

        if other.value == 0:
            raise Exception('ADNode division error')

        def partial_deriv(x):
            if x == self:
                return 1. / other.value
            d_dy = -self.value / (other.value**2)
            return d_dy
        return ADNode(self.value / other.value, (self, other), '/', partial_deriv)

    def __rtruediv__(self, other):
        return self / other

    def is_variable(self):
        return len(self.input_nodes) == 0

    def log(self):
        if self.value == 0:
            raise Exception('Cannot apply log to zero')

        partial_deriv = lambda x: 1. / x.value
        return ADNode(math.log(self.value), (self,), 'log', partial_deriv)

    def exp(self):
        partial_deriv = lambda x: math.exp(self.value)
        return ADNode(math.exp(self.value), (self,), 'exp', partial_deriv)

    def tanh(self):
        # tanh(x) = (e^{2x} - 1)/(e^{2x} + 1)
        tanh_x = (math.exp(2*self.value) - 1.)/(math.exp(2*self.value) + 1)
        # d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)
        partial_deriv = lambda x: 1. - tanh_x**2
        return ADNode(tanh_x, (self,), 'tanh', partial_deriv)

    def backward(self, deriv=1.):
        self.deriv += deriv
        for inp, mult in self.input_nodes.items():
            child_deriv = mult * self.deriv * self.partial_deriv(inp)
            inp.backward(child_deriv)

    @classmethod
    def sum(cls, others, label='Σ'):
        bp_fn = lambda x: 1.
        return ADNode(sum([o.value for o in others]), others, label, bp_fn)

    @classmethod
    def prod(cls, nodes, label='Π'):
        def bp_fn(x):
            res = 1.
            for inp in nodes:
                if inp != x:
                    res *= inp.value
            return res
        product = reduce(operator.mul, [n.value for n in nodes])
        return ADNode(product, nodes, label, bp_fn)

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

def prod_example():
    n = 5
    xs = []
    for i in range(n):
        xs.append(ADNode(i + 1., label=f'x{i+1}'))

    out = ADNode.prod(xs)
    out.backward()

    nodes = xs + [out]
    for node in nodes:
        print(node)

def div_example1():
    x = ADNode(4., label='x')
    y = ADNode(5., label='y')
    x_over_y = x / y

    print("============ backward ===========")
    x_over_y.backward()

    nodes = [x, y, x_over_y]
    for node in nodes:
        print(node)

def div_example2():
    x = ADNode(4., label='x')
    one_over_x = 1. / x

    print("============ backward ===========")
    one_over_x.backward()

    nodes = [x, one_over_x]
    for node in nodes:
        print(node)



def sigmoid_simple_example():
    x = ADNode(-1., label='x')
    neg_x = -x
    denom = 1. + neg_x.exp()
    out = 1. / denom

    print(f"σ(1 - σ) = {out.value * (1 - out.value)}")

    out.backward()

    nodes = [x, neg_x, denom, out]
    for node in nodes:
        print(node)

def sigmoid_advanced_example():
    n = 5
    x_values = [0.5, 5, 4, 3, 6]
    w_values = [2, -2, 3, -3, 1]
    xs = []
    for i in range(n):
        xs.append(ADNode(x_values[i], label=f'x{i+1}'))
    ws = []
    for i in range(n):
        ws.append(ADNode(w_values[i], label=f'w{i+1}'))

    dot = ADNode.sum([ws[i] * xs[i] for i in range(n)], label=f'w·x')
    out = 1. / (1. + (-dot).exp())

    print(f"σ(1 - σ) = {out.value * (1 - out.value)}")

    out.backward()

    nodes = ws + xs + [dot, out]
    for node in nodes:
        print(node)

def ak_bug_example1():
    a = ADNode(3.0, label='a')
    b = a + a
    b.label = 'b'

    print("============ backward ===========")
    b.backward()
    for node in [a, b]:
        print(node)

def ak_bug_example2():
    a = ADNode(-2.0, label='a')
    b = ADNode(3.0, label='b')
    d = a + b
    e = a * b
    f = d * e

    print("============ backward ===========")
    f.backward()
    for node in [a, b, d, e, f]:
        print(node)

def ak_tanh_decomposed_example():
    x = ADNode(1.25, label='x')
    num = (2*x).exp() - 1.
    denom = 1. + (2*x).exp()
    out = num / denom

    print("============ backward ===========")
    out.backward()
    for node in [x, num, denom, out]:
        print(node)


if __name__ == '__main__':
    # ak_example2()
    # prod_example()
    # div_example1()
    div_example2()
    # sigmoid_manual_example()
    # sigmoid_simple_example()
    # ak_bug_example1()
    # ak_bug_example2()
    # ak_tanh_decomposed_example()