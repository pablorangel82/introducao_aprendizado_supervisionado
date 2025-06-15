from neuronio import Neuronio
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',level=logging.INFO)

class MLP:
    
    def __init__(self, entradas, numero_neuronios_camada_oculta, numero_neuronios_camada_final, numero_iteracoes, taxa_aprendizado):
        self.numero_iteracoes = numero_iteracoes
        self.entradas = entradas
        self.taxa_aprendizado = taxa_aprendizado
        self.camada_oculta = []
        self.camada_saida = []

        for i in range(numero_neuronios_camada_oculta):
            self.camada_oculta.append(Neuronio (numero_atributos=len(entradas[0].x), pesos_aleatorios = True, bias_aleatorio = True))
        
        for i in range(numero_neuronios_camada_final):
            self.camada_saida.append(Neuronio (numero_atributos=len(self.camada_oculta),pesos_aleatorios = True, bias_aleatorio = True))
   
  
    def forward(self, entrada):
        logger.debug('\t Camada Oculta')
        for neuronio in self.camada_oculta:
            logger.debug(f' \t\t Pesos = {neuronio.w}, Bias = {neuronio.b}')
            z = neuronio.soma(entrada.x)
            neuronio.y = neuronio.sigmoid(z)

        logger.debug('\t Camada Final')
        entradas_ocultas = [n.y for n in self.camada_oculta]

        for neuronio_final in self.camada_saida:
            logger.debug(f' \t\t Pesos = {neuronio_final.w}, Bias = {neuronio.b}')
            z = neuronio_final.soma(entradas_ocultas)
            neuronio_final.y = neuronio_final.sigmoid(z)
            logger.debug(f'\t\t Saída = {neuronio_final.y }')
            saida = neuronio_final.y
        return saida
    
    def derivada_parcial_erro(self, neuronio, entrada):
        erro = neuronio.y - entrada.g 
        return erro
    
    def derivada_total_sigmoid(self, y):
        return y * (1 - y) 

    def backpropagation(self, entrada):
        erros_camada_saida = []
        diferencas_camada_saida = []

        # Retropropagar erros para trás (Camada Final para Camada Oculta)
        for neuronio in self.camada_saida:
            #Primeira Derivada: derivada de saída do neurônio pela entrada esperada
            erro = self.derivada_parcial_erro(neuronio, entrada)
            erros_camada_saida.append(erro)
            #Segunda Derivada: derivada de saída do neurônio pela entrada esperada
            ativacao = self.derivada_total_sigmoid(neuronio.y)
            #Produto da primeira derivada parcial (erro) pela segunda (ativação)
            diferenca = erro * ativacao
            diferencas_camada_saida.append(diferenca)

        erros_camada_oculta = [0 for _ in self.camada_oculta]
        diferencas_camada_oculta = [0 for _ in self.camada_oculta]

        # Retropropagar erros para trás (Camada Oculta para Camadas Ocultas Anteriores ou a Entrada)
        for i, neuronio_oculto in enumerate(self.camada_oculta):
            for j, neuronio_saida in enumerate(self.camada_saida):
                erros_camada_oculta[i] += neuronio_saida.w[i] * diferencas_camada_saida[j]
            #Produto das primeiras e segundas derivadas parciais pela terceira (peso)    
            diferencas_camada_oculta[i] = erros_camada_oculta[i] * self.derivada_total_sigmoid(neuronio_oculto.y)

        #Atualizar os pesos na camada de saída
        entradas_ocultas = [n.y for n in self.camada_oculta]
        for i, neuronio in enumerate(self.camada_saida):
            for j in range(len(neuronio.w)):
                neuronio.w[j] -= self.taxa_aprendizado * diferencas_camada_saida[i] * entradas_ocultas[j]
            neuronio.b -= self.taxa_aprendizado * diferencas_camada_saida[i]

        #Atualizar os pesos na camada oculta
        for i, neuronio in enumerate(self.camada_oculta):
            for j in range(len(neuronio.w)):
                neuronio.w[j] -= self.taxa_aprendizado * diferencas_camada_oculta[i] * entrada.x[j]
            neuronio.b -= self.taxa_aprendizado * diferencas_camada_oculta[i]

        erro_total = sum(e ** 2 for e in erros_camada_saida)
        return erro_total
    
    def treinar(self):
        for iteracao in range(self.numero_iteracoes):
            erro_total = 0
            for entrada in self.entradas:
                self.forward(entrada)
                erro_total += self.backpropagation(entrada)
            if (iteracao + 1) % 1000 == 0 or iteracao == 0:
                logging.info(f'Iteração {iteracao + 1}, Erro total: {erro_total}')

    def testar(self, entradas):
        for entrada in entradas:
            saida = self.forward(entrada)
            logging.info(f'Entrada: {entrada.x} -> Saída Recebida: {saida:.6f} | Esperada: {entrada.g}')
