from entrada import Entrada
from neuronio import Neuronio

p1 = Entrada([49, 20000], [[18,120],[1200, 30000]])
p2 = Entrada([24, 5000], [[18,120],[1200, 30000]])
n1 = Neuronio(2)
n1.avaliacao(p1)
print(n1.w)


