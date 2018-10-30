from neuron import Neuron
import random
class NeuronLayer:
    # 神经层类
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias 一层中的所有神经元共享一个bias
        self.bias = bias if bias else random.random()
        random.random()
        # 生成0和1之间的随机浮点数float，它其实是一个隐藏的random.Random类的实例的random方法。
        # random.random()和random.Random().random()作用是一样的。
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
        # 在神经层的初始化函数中对每一层的bias赋值,利用神经元的init函数对神经元的bias赋值

    def inspect(self):
        # print该层神经元的信息
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        # 前向传播过程outputs中存储的是该层每个神经元的y/a的值(经过神经元激活函数的值有时被称为y有时被称为a)
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs