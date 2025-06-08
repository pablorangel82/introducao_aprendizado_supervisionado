class Entrada:

    def __init__(self, valores_brutos, limites, saida_esperada):
        self.x = []
        self.normalizar(valores_brutos, limites)
        self.g = saida_esperada


    
    def normalizar(self,valores_brutos, limites):
        for i in range (len(valores_brutos)):
            valor_bruto = valores_brutos[i]
            limite = limites [i]
            _min = limite [0]
            _max = limite [1]
            valor = (valor_bruto - _min) / (_max - _min)
            self.x.append(valor)

