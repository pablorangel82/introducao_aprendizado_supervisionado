from neuronio import Neuronio
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',level=logging.INFO)

class Perceptron:
    
    def __init__(self, entradas, numero_iteracoes, taxa_aprendizado):
        self.numero_iteracoes = numero_iteracoes
        self.entradas = entradas
        self.taxa_aprendizado = taxa_aprendizado
        self.n = Neuronio (numero_atributos=len(entradas[0].x),pesos_aleatorios=False, bias_aleatorio=False)
        
    def treinar(self):
        for iteracao in range(self.numero_iteracoes):
            logging.info(f'Iteração {iteracao+1}: Pesos = {self.n.w}, Bias = {self.n.b}') 
            for k in range(len (self.entradas)):
                x = self.entradas[k].x
                g = self.entradas[k].g
                z = self.n.soma(x)
                y = self.n.degrau(z)
                erro = g - y
                logging.info(f'\t Entrada: {x}, Saída = {y}')
                for i in range(len(x)):
                    novo_peso = self.n.w[i] + (erro * x[i] * self.taxa_aprendizado)
                    logging.info(f'\t\t w ({novo_peso}) =  w ({self.n.w[i]}) + Erro ({erro}) * x ({x[i]}) * eta ({self.taxa_aprendizado})')
                    self.n.w[i] = novo_peso
                    self.n.b += self.taxa_aprendizado * erro
            
    #Plota as entradas junto com a fronteira de decisão da rede.
    def plotar (self, entradas):
        x_vals_entradas = []
        y_vals_entradas = []
        cores_entradas = []

        for entrada in entradas:
            x_vals_entradas.append(entrada.x[0])
            y_vals_entradas.append(entrada.x[1])
            cores_entradas.append(entrada.g)

        x_vals_entradas = np.array(x_vals_entradas)
        y_vals_entradas = np.array(y_vals_entradas)
        cores_entradas = np.array(cores_entradas)

        # Criando uma grade pra cobrir o espaço de entrada
        x_vals = np.linspace(x_vals_entradas.min()-0.5, x_vals_entradas.max()+0.5, 500)
        y_vals = np.linspace(y_vals_entradas.min()-0.5, y_vals_entradas.max()+0.5, 500)
        xx, yy = np.meshgrid(x_vals, y_vals)

        # Forward pra cada ponto da grade pra saber a saída da rede
        Z = np.ones_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                Z[i,j] = self.n.degrau(self.n.soma([xx[i,j], yy[i,j]]))

        Z = Z > 0  # limiar pra classificar como 0 ou 1
        
        # Plotando o contorno da decisão
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(x_vals_entradas[cores_entradas == 0], y_vals_entradas[cores_entradas == 0],
                    color='blue', label='Classe 0')
        plt.scatter(x_vals_entradas[cores_entradas == 1], y_vals_entradas[cores_entradas == 1],
                    color='red', label='Classe 1')

        plt.legend()
        plt.xlabel('Atributo 1')
        plt.ylabel('Atributo 2')
        plt.title('Classificação pelo Perceptron Simples')
        plt.show()
