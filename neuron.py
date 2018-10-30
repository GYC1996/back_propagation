import math
class Neuron:
    # 神经元类
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        # output即为输入即为y(a)意为从激活函数中的到的值
        return self.output

    def calculate_total_net_input(self):
        # 此处计算的为激活函数的输入值即z=W(n)x+b
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i]*self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # 使用sigmoid函数为激励函数,一下是sigmoid函数的定义
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1/(1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    # 确定神经元的总输入需要改变多少，以接近预期的输出
    # Now that we have the partial derivative of the error(Cost function) with respect to the output (∂E/∂yⱼ)
    # 我们可以根据cost function对y(a)神经元激活函数输出值的偏导数和激活函数输出值y(a)对激活函数输入值z=wx+b的偏导数计算delta(δ).
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ 关键key
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output)*self.calculate_pd_total_net_input_wrt_input()

    # The error for each neuron is calculated by the Mean Square Error method:
    # 每个神经元的误差由平均平方误差法计算
    def calculate_error(self, target_output):
        return 0.5*(target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # 对实际输出的误差的偏导是通过计算得到的--即self.output(y也常常用a表示经过激活函数后的值)
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)BP算法最后隐层求cost function 对a求导
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # 维基百科关于反向传播[1]的文章简化了以下内容，但大多数其他学习材料并没有简化这个过程[2]
    # = actual output - target output
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    # 注意我们一般将输出层神经元的输出为yⱼ,而目标标签(正确答案)表示为tⱼ.
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ)) 注意我们对于神经元使用的激活函数都是logistic函数
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    # 注意我们用j表示我们正在看的这层神经元的输出,我们用i表示这层的后一层的神经元.
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)这是sigmoid函数的导数表现形式.
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output*(1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # 激活函数的输入是所有输入的加权权重的总和
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # 总的净输入与给定的权重的偏导数(其他所有的项都保持不变)
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]