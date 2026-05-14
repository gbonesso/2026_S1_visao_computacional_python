"""
Laboratório 03 - Granulometria Morfológica

Disciplina: Visão Computacional

Objetivo
Investigar o uso de granulometria morfológica para caracterização de imagens em diferentes escalas.
Cada aluno deverá obter uma assinatura granulométrica para cada imagem, comparar os resultados
entre classes e analisar se esse tipo de descritor é útil para distinguir texturas ou materiais.
Base de imagens sugerida
Sugere-se usar um subconjunto da base KTH-TIPS em tons de cinza. Selecione 3 classes e use entre
10 e 20 imagens por classe, totalizando entre 30 e 60 imagens.
Página oficial da base:
https://www.csc.kth.se/cvap/databases/kth-tips/download.html
As imagens devem ser mantidas em tons de cinza e, se necessário, redimensionadas para o mesmo
tamanho.

Descrição geral da atividade
Cada aluno deverá representar cada imagem por uma assinatura granulométrica obtida por operações
morfológicas em múltiplas escalas. A escolha da estratégia de implementação fica a critério do aluno.
O laboratório deve incluir a geração das assinaturas para todas as imagens escolhidas, a comparação
entre as classes e uma análise dos resultados obtidos.

Configuração experimental mínima
Deve-se usar uma única base de imagens, com pelo menos 3 classes.
Cada classe deve conter entre 10 e 20 imagens.
Deve-se gerar uma assinatura granulométrica para cada imagem.
Deve-se apresentar pelo menos um gráfico médio por classe.
Deve-se realizar uma comparação entre as classes usando as assinaturas obtidas.

O que deve ser apresentado no relatório
Descrição da base utilizada e das classes selecionadas.
Descrição da abordagem adotada para obter as assinaturas granulométricas.
Exemplos de imagens da base.
Gráficos das assinaturas granulométricas individuais e médias por classe.
Análise de quais classes apresentaram curvas mais semelhantes e quais foram mais fáceis de separar.
Discussão sobre o efeito da faixa de escalas escolhida.
Relato das dificuldades encontradas ao longo do desenvolvimento.
Conclusão sobre a utilidade da granulometria morfológica para o problema estudado.

Entrega
O trabalho é individual.
Entregar o código-fonte da implementação.
Entregar um relatório em PDF contendo as figuras, tabelas e análises solicitadas.

Observações
O foco do laboratório é investigar se a granulometria morfológica produz descritores úteis para
caracterizar imagens em diferentes escalas.
É permitido usar funções prontas da biblioteca escolhida.
Pode implementar em C, C++ ou python.
"""