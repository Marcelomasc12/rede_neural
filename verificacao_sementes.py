"""Verificação extra (fora da entrega principal): repete o MESMO pipeline do laboratório
variando a semente do treino da rede e, separadamente, a divisão treino/teste."""
import warnings, numpy as np, pandas as pd
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.simplefilter('ignore', ConvergenceWarning)
X, y = load_digits(return_X_y=True)
ARQ = {'16': (16,), '32': (32,), '64': (64,)}
def rodar(split_seed, net_seed, arq):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=split_seed, stratify=y)
    p = Pipeline([('e', StandardScaler()), ('r', MLPClassifier(hidden_layer_sizes=arq, activation='relu',
        solver='adam', max_iter=180, random_state=net_seed, early_stopping=True, n_iter_no_change=12))])
    p.fit(Xtr, ytr); return accuracy_score(yte, p.predict(Xte))
# A) divisão fixa (42), 20 sementes de rede
A = pd.DataFrame({k: [rodar(42, s, a) for s in range(20)] for k, a in ARQ.items()})
# B) 20 divisões diferentes, semente de rede = semente da divisão
B = pd.DataFrame({k: [rodar(s, s, a) for s in range(20)] for k, a in ARQ.items()})
for nome, d in [('A: divisão fixa, 20 sementes de rede', A), ('B: 20 divisões diferentes', B)]:
    print(nome); print(d.agg(['mean', 'std', 'min', 'max']).round(4).to_string())
    print('vezes que cada uma foi a melhor (empates contam p/ todas):', {k: int((d[k] == d.max(axis=1)).sum()) for k in d}); print()
print('A) 16 > 32 em', int((A['16'] > A['32']).sum()), 'de 20 | 64 > 32 em', int((A['64'] > A['32']).sum()), 'de 20')
print('B) 16 > 32 em', int((B['16'] > B['32']).sum()), 'de 20 | 64 > 32 em', int((B['64'] > B['32']).sum()), 'de 20')
