from mlp import MLP
from entrada import Entrada
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s',level=logging.INFO)

e1 = Entrada (valores_brutos=[0,0],limites=[[0,1],[0,1]],saida_esperada=0)
e2 = Entrada (valores_brutos=[0,1],limites=[[0,1],[0,1]],saida_esperada=1)
e3 = Entrada (valores_brutos=[1,0],limites=[[0,1],[0,1]],saida_esperada=1)
e4 = Entrada (valores_brutos=[1,1],limites=[[0,1],[0,1]],saida_esperada=0)


mlp = MLP (entradas=[e1,e2,e3,e4], numero_neuronios_camada_oculta=2, numero_neuronios_camada_final=1, taxa_aprendizado=0.5, numero_iteracoes=10000)
logging.info('\nTreinando...')
mlp.treinar()

logging.info('\nTestando...')
mlp.testar(entradas=[e1,e2,e3,e4])
