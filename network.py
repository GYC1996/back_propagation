import random
from layer import NeuronLayer
class NeuralNetwork:
    # 神经网络类
    LEARNING_RATE = 0.5

    # 设置学习率为0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None, hidden_layer_bias=None,
                 output_layer_weights=None, output_layer_bias=None):
        # 初始化一个三层神经网络结构
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):  # num_hidden,遍历隐藏层
            for i in range(self.num_inputs):  # 遍历输入层
                if not hidden_layer_weights:
                    # 如果hidden_layer_weights的值为空,则利用随机化函数对其进行赋值,否则利用hidden_layer_weights中的值对其进行更新
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):  # num_outputs,遍历输出层
            for h in range(len(self.hidden_layer.neurons)):  # 遍历输出层
                if not output_layer_weights:
                    # 如果output_layer_weights的值为空,则利用随机化函数对其进行赋值,否则利用output_layer_weights中的值对其进行更新
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):  # 输出神经网络信息
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):  # 返回输出层y值
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs) # 理论的前向传播输出

    # Uses online learning, ie updating the weights after each training case
    # 使用在线学习方式,训练每个实例之后对权值进行更新
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        # 反向传播
        # 1. Output neuron deltas输出层deltas
        pd_errors_wrt_output_neuron_total_net_input = [0]*len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # 对于输出层∂E/∂zⱼ=∂E/∂a*∂a/∂z=cost'(target_output)*sigma'(z)
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[
                o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas隐藏层deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0]*len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # 我们需要计算误差对每个隐藏层神经元的输出的导数,由于不是输出层所以dE/dyⱼ需要根据下一层反向进行计算,即根据输出层的函数进行计算
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o]* \
                                                    self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output*self.hidden_layer.neurons[
                h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights 更新输出层权重
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # 注意:输出层权重是隐藏层神经元与输出层神经元连接的权重
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o]*self.output_layer.neurons[
                    o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE*pd_error_wrt_weight

        # 4. Update hidden neuron weights 更新隐藏层权重
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # 注意:隐藏层权重是输入层神经元与隐藏层神经元连接的权重
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h]*self.hidden_layer.neurons[
                    h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE*pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        #  使用平方差计算训练集误差
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error
