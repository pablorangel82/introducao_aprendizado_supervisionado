from random import uniform
import math

class Neuronio:
 
    def __init__(self, numero_atributos, pesos_aleatorios = False, bias_aleatorio = False):
        self.w = []
        self.y = 0
        for i in range (numero_atributos):
            peso = 0
            if pesos_aleatorios is True:
                peso = uniform(-1.0, 1.0)
            self.w.append (peso)
        self.b = 0
        if bias_aleatorio is True:
            self.b = uniform(-1.0, 1.0)

    def soma(self, x):
        z = 0
        for i in range (len(x)):
            z += (x[i] * self.w[i])
        z += self.b
        return z 

    def degrau(self, z):
        self.y = 1 if z >= 0 else 0
        return self.y

    def sigmoid(self, z):
        y = 1 / (1+ math.exp(-z))
        return y
    
    def tanh(self, z):
        y = ( math.exp (z) - math.exp(-z)) / (math.exp(z) + math.exp(-z)) 
        return y
    
    def relu(self,z):
        y = max(0,z)
        return y