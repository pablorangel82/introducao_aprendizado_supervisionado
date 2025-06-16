from perceptron import Perceptron
from entrada import Entrada
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',level=logging.INFO)

e1 = Entrada (valores_brutos=[0,0],limites=[[0,1],[0,1]],saida_esperada=0)
e2 = Entrada (valores_brutos=[0,1],limites=[[0,1],[0,1]],saida_esperada=0)
e3 = Entrada (valores_brutos=[1,0],limites=[[0,1],[0,1]],saida_esperada=0)
e4 = Entrada (valores_brutos=[1,1],limites=[[0,1],[0,1]],saida_esperada=1)
entradas=[e1,e2,e3,e4]

perceptron = Perceptron (entradas, taxa_aprendizado=0.1, numero_iteracoes=10)
perceptron.treinar()
perceptron.plotar(entradas)