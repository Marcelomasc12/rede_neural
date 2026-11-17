# Parecer técnico — Reconhecimento de dígitos

**Equipe:** Marcelo Negrao, Maria Eduarda Aguiar, Maraian Martins 
**Data:** 16/11/2026  
**Versão do Python / scikit-learn:** Python 3.12.3 / scikit-learn 1.9.1  

## 1. Comparação dos modelos

Execução do script (divisão treino/teste e semente fixas em 42; 1.347 imagens de treino e 450 de teste).

| Configuração | Acurácia (%) | Tempo de treino (s) | Observações |
|---|---:|---:|---|
| MLP com 16 neurônios | 92,7 | 0,22 | F1 macro 0,925; 33 erros em 450; 1.210 parâmetros; parou na época 85; sem aviso de convergência |
| MLP com 32 neurônios | 90,7 | 0,13 | F1 macro 0,905; 42 erros; 2.410 parâmetros; parou na época 40; sem aviso de convergência |
| Variação experimental: `hidden_layer_sizes` de (32,) para (64,) (único parâmetro alterado) | 94,7 | 0,16 | F1 macro 0,947; 24 erros; 4.810 parâmetros; parou na época 40; sem aviso de convergência |

Os tempos são todos menores que 0,3 s e variaram um pouco entre execuções (a 1ª execução do 16 e do 32 deu 0,23 s e 0,14 s), então não servem para distinguir as configurações.

**Atenção ao resultado de uma execução só.** Nesta execução o modelo de 16 neurônios ficou à frente do de 32, o contrário do que se esperaria. Como a diferença é de 9 imagens em 450 (o erro-padrão da acurácia com 450 amostras é de cerca de 1,2 ponto percentual), repeti o mesmo pipeline (`verificacao_sementes.py`, fora da entrega principal) com 20 sementes de rede e divisão fixa, e com 20 divisões diferentes:

| Configuração | Média ± desvio (divisão fixa, 20 sementes) | Média ± desvio (20 divisões diferentes) |
|---|---:|---:|
| 16 neurônios | 91,5% ± 2,8 | 91,9% ± 2,3 |
| 32 neurônios | 93,4% ± 1,9 | 94,1% ± 2,2 |
| 64 neurônios | 94,3% ± 1,0 | 95,3% ± 1,3 |

Na média a ordem é 64 > 32 > 16, e o 16 só superou o 32 em 6 de 20 repetições (divisão fixa) e em 4 de 20 (divisões diferentes). O resultado em que o 16 ganhou do 32 foi, portanto, variação de treino e amostra, não efeito da arquitetura.

## 2. Dois erros observados

Os índices abaixo se referem ao conjunto de teste (`random_state=42`). As imagens têm só 8×8 pixels. Descrevo o que aparece nos valores dos pixels e nos gráficos; isso sugere por que o traçado confunde, mas não prova o que acontece dentro da rede.

- **Exemplo 1 (teste #25):** dígito verdadeiro = 8, previsão = 1. A tinta está concentrada numa faixa vertical estreita (colunas 2 a 4, com valor 16, o máximo, na coluna 3 das linhas 1 a 5; só o topo avança um pouco para a coluna 5), e as colunas 0, 1, 6 e 7 estão vazias. Um 8 típico do treino tem tinta também nas colunas 1–2 e 5–6 (as curvas laterais dos dois laços), e aqui elas praticamente não aparecem. O traçado parece um "1" mais grosso. O par 8→1 foi o erro mais frequente do modelo de 16 neurônios (10 dos 33 erros) e também o do de 32 (6 de 42).
- **Exemplo 2 (teste #62):** dígito verdadeiro = 0, previsão = 4. O zero é estreito (colunas 2 a 5, enquanto o zero médio ocupa as colunas 1 a 6) e o buraco central não está limpo na metade de baixo: há tinta dentro dele (valores 11, 7 e 16 nas linhas 4 e 5). Isso pode lembrar o cruzamento central de um 4.

Esses dois erros (#25 e #62) aparecem nos três modelos, junto com mais 10 imagens (12 erros em comum nos três). Ou seja, parte dos erros não depende do tamanho da rede e provavelmente vem de traçados ambíguos. Os gráficos da execução original (16 e 32 neurônios) estão em `resultados_antes_alteracao/`, e os da execução com 64 neurônios em `resultados/`.

## 3. Recomendação

Para uma aplicação inicial, recomendo a configuração de **64 neurônios ocultos**, com ressalvas:

- Teve a maior acurácia na execução do script (94,7%) e também a melhor média e a menor variação entre repetições (95,3% ± 1,3 contra 94,1% ± 2,2 do de 32).
- O custo extra é desprezível neste problema: o treino leva menos de 1 s e o modelo tem pouco mais de 4,8 mil parâmetros.
- A vantagem sobre o modelo de 32 é pequena (cerca de um ponto percentual na média) e do mesmo tamanho do desvio entre repetições. Se a simplicidade pesar mais, o de 32 é uma escolha defensável.
- O de 16 neurônios eu não recomendaria: foi o menos estável entre sementes, com variação de 2,3 a 2,8 pontos.

Nenhum dos modelos teve aviso de convergência. O treino foi interrompido pelo critério de parada antecipada (`early_stopping`), e não pelo limite de 180 iterações.

## 4. Antes de colocar em produção

- Testar com **dígitos reais da aplicação** (outra caneta, tamanho, inclinação, qualidade da digitalização). O `load_digits` tem só 1.797 imagens de 8×8 pixels, bem comportadas, e não representa escrita em geral.
- Medir com **mais dados e validação cruzada**, em vez de uma divisão única, e relatar a variação (como na tabela acima).
- Olhar **erros por classe** (o 8 concentrou boa parte dos erros nos modelos de 16 e de 32 neurônios: 16 de 33 e 14 de 42) e decidir o custo de cada tipo de erro, para escolher entre acurácia, precisão e recall por classe.
- Definir o que fazer com previsões incertas (por exemplo, limiar de confiança e envio para revisão humana).
- Testar outras alterações **uma de cada vez** (regularização, número de camadas, taxa de aprendizado) e comparar com um modelo de referência mais simples.
- Verificar desempenho, latência e monitoramento depois de implantado, pois os dados de campo mudam.

## 5. Conclusão

O experimento mostrou que um único número de acurácia, obtido numa só execução, pode enganar: na execução do script a rede menor superou a intermediária, mas repetindo o treino com outras sementes e outras divisões a ordem se inverteu, e as diferenças ficaram do tamanho da variação do próprio treino. Também mostrou que o erro não é só da rede: 12 imagens foram erradas pelos três modelos, o que aponta para traçados ambíguos e para a pouca resolução (8×8 pixels). Treinar mais ou ter mais neurônios não elimina esses casos, e generalização só pode ser avaliada com dados que o modelo não viu.

### Resposta à pergunta-chave

**Uma rede com mais neurônios sempre tem melhor resultado? Não.** No script, o modelo de 16 superou o de 32 numa execução, e as redes maiores mostraram, na média de 20 repetições, acurácia um pouco maior. Mesmo assim, o ganho é pequeno e a variação entre repetições é do mesmo tamanho.

- **O experimento permite concluir:** neste dataset e nesta configuração de treino, 64 neurônios foi, na média, ligeiramente melhor que 32 e claramente mais estável que 16; uma única execução não basta para ordenar modelos próximos; parte dos erros é compartilhada entre as arquiteturas.
- **O experimento não permite concluir:** que "mais neurônios é melhor" em geral; que 64 seja o ideal (só testamos 16, 32 e 64); que os resultados valham para escrita real, outras resoluções ou outros algoritmos; nem a causa interna de cada erro (as explicações da seção 2 são observações visuais).
