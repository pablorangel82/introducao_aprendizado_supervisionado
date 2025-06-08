from neuronio import Neuronio

class Perceptron:
    
    def __init__(self, entradas, numero_iteracoes, taxa_aprendizado):
        self.numero_iteracoes = numero_iteracoes
        self.entradas = entradas
        self.taxa_aprendizado = taxa_aprendizado
        self.n = Neuronio (numero_atributos=len(entradas[0].x),pesos_aleatorios=False, bias_aleatorio=False)
        
    def treinar(self):
        for iteracao in range(self.numero_iteracoes):
            print(f'Iteração {iteracao+1}: Pesos = {self.n.w}, Bias = {self.n.b}') 
            for k in range(len (self.entradas)):
                x = self.entradas[k].x
                g = self.entradas[k].g
                z = self.n.soma(x)
                y = self.n.degrau(z)
                erro = g - y
                print(f'\t Entrada: {x}, Saída = {y}')
                for i in range(len(x)):
                    novo_peso = self.n.w[i] + (erro * x[i] * self.taxa_aprendizado)
                    print(f'\t\t w ({novo_peso}) =  w ({self.n.w[i]}) + Erro ({erro}) * x ({x[i]}) * eta ({self.taxa_aprendizado})')
                    self.n.w[i] = novo_peso
                    self.n.b += self.taxa_aprendizado * erro
            
