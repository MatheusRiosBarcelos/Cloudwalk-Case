# 🛡️ Cloudwalk Case — Detecção de Fraude e Chargeback em Transações

Case técnico de análise de risco transacional para a Cloudwalk: a partir de uma amostra de transações de pagamento, o projeto identifica padrões de fraude e chargeback, propõe uma política de decisão em camadas e treina um modelo de machine learning para pontuar o risco de cada transação em tempo real.

O projeto está estruturado em duas camadas de detecção, que se complementam:

1. **Camada 1 — Análise exploratória e regras de negócio**: investigação dos dados brutos para entender padrões de fraude (concentração por merchant, BIN, valor, horário) e identificação de possíveis anéis de fraude via grafo de entidades (usuário ↔ dispositivo ↔ cartão).
2. **Camada 2 — Modelo de ML**: um classificador treinado sobre uma feature store derivada, que gera um score contínuo de risco por transação e roteia cada uma para uma de três ações (auto-aprovar, autenticação adicional, ou bloqueio/revisão manual) — em vez de um único corte binário.

## 🎯 Problema de negócio

Dado o histórico de uma transação (valor, horário, usuário, cartão, dispositivo, merchant), qual a probabilidade dela resultar em chargeback? E, mais importante para o negócio: qual a ação certa a tomar — aprovar, pedir autenticação extra (3DS/OTP), ou bloquear — de forma que se capture o máximo de fraude possível com o mínimo de fricção para o cliente legítimo?

## 🏗️ Estrutura do repositório

```
Cloudwalk-Case/
├── exploratory_analyzes.ipynb   # Camada 1: EDA, análise de padrões e regras de negócio
├── fraud_dashboard.py           # Dashboard interativo em Streamlit (Camadas 1 e 2)
├── train_risk_model.py          # Camada 2: treino do modelo de risco + política de tiers
├── transactional-sample.csv     # Dados brutos de transações
├── fs_transactional_data.csv    # Feature store (features de velocidade, entidade, etc.)
├── risk_model.joblib            # Modelo treinado + thresholds + metadados (gerado)
├── risk_scores_test.csv         # Transações do conjunto de teste, já pontuadas (gerado)
├── requirements.txt             # Dependências do projeto
└── .devcontainer/               # Ambiente de desenvolvimento reprodutível (Docker)
```

## 📊 Camada 1 — Exploração e regras (`exploratory_analyzes.ipynb`, `fraud_dashboard.py`)

A investigação inicial cruza a taxa de chargeback com diferentes dimensões da transação:

- **Volume e tendência temporal**: evolução diária de volume de transações vs. taxa de chargeback.
- **Concentração por merchant e BIN**: quais merchants e faixas de cartão (BIN) concentram a maior taxa de chargeback, sinal comum de conta comprometida ou merchant de risco.
- **Distribuição de valores**: como o valor da transação se relaciona com a probabilidade de chargeback (em escala logarítmica, já que valores de transação costumam ser fortemente assimétricos).
- **Grafo de entidades (usuário ↔ dispositivo ↔ cartão)**: construído com `networkx`, conecta usuários, dispositivos e cartões que compartilham transações. Clusters com muitas transações concentradas e taxa de chargeback alta são o sinal mais forte de um anel de fraude organizado — não fraudadores isolados, mas grupos de entidades reaproveitando dispositivos/cartões entre si.
- **Worklist operacional**: tabelas de usuários, dispositivos e cartões mais arriscados, com ação recomendada (monitorar / autenticação extra / bloquear) baseada na taxa de chargeback de cada entidade.

## 🤖 Camada 2 — Modelo de risco (`train_risk_model.py`)

### Features utilizadas

- **Base**: valor da transação, hora do dia.
- **Velocidade**: contagem de transações em janelas de 1h/24h/7d, por usuário, cartão e dispositivo — captura comportamento de rajada, típico de fraude automatizada.
- **Diversidade de entidades**: quantidade de dispositivos distintos usados por um usuário em 24h, quantidade de merchants distintos por cartão em 24h.
- **Entidade nova**: flags indicando se é a primeira transação já vista daquele usuário, cartão ou dispositivo — contas/dispositivos novos concentram risco.
- **Razão de valor**: quanto o valor da transação atual foge do padrão histórico daquele usuário/cartão/dispositivo.
- **Taxa de risco histórica (out-of-time)**: taxa de chargeback de merchant e BIN, mas recalculada considerando **apenas transações anteriores** a cada transação — não a média do período inteiro.

### Decisões técnicas que valem destaque

- **Correção de vazamento de dados (data leakage) nas taxas de risco**: a feature store já trazia uma taxa de chargeback por merchant/BIN calculada sobre o período completo — o que inclui transações futuras (e a própria transação) no cálculo do risco daquele merchant. Isso é vazamento de rótulo. O projeto recalcula essas taxas de forma expansiva e defasada (`expanding().mean().shift()`), simulando exatamente a informação disponível no momento de cada transação, com fallback para a taxa global em casos de "cold start" (entidade nova).
- **Split temporal, não aleatório**: treino com o passado, teste com o futuro (`time_based_split`). Um split aleatório permitiria que o mesmo usuário/cartão/dispositivo aparecesse em treino e teste, deixando o modelo memorizar entidades em vez de aprender padrões generalizáveis — o split cronológico reproduz como o modelo seria de fato usado em produção.
- **Modelo**: `HistGradientBoostingClassifier` (scikit-learn), com `class_weight="balanced"` para lidar com o desbalanceamento natural entre transações legítimas e fraudulentas.
- **Política de decisão em três tiers, não um corte único**: em vez de um único threshold, o modelo define dois pontos de corte de forma orientada a dados — um `high_threshold` (acima do qual ainda se captura uma meta de recall de chargebacks, ex. 60%) e um `low_threshold` (abaixo do qual no máximo uma meta de falso-positivo é tolerada, ex. 5% dos legítimos). Entre os dois, a transação recebe fricção proporcional (autenticação adicional) em vez de aprovação ou bloqueio direto — um trade-off mais realista entre segurança e experiência do cliente.
- **Importância de features via permutation importance**, avaliada em PR-AUC (mais adequada que ROC-AUC para problemas com classes desbalanceadas, como fraude).

### Métricas de avaliação

O modelo é avaliado no conjunto de teste (mais recente, nunca visto em treino) com **ROC-AUC** e **PR-AUC** (com baseline = taxa real de chargeback no teste), além do breakdown de volume e taxa de chargeback por tier de decisão.

## 📺 Dashboard (`fraud_dashboard.py`)

Aplicação Streamlit que expõe interativamente as duas camadas:

- Filtros por período, merchant e BIN.
- KPIs de topo: volume, taxa de chargeback, valor em risco, volume total.
- Gráficos de tendência, concentração por merchant/BIN e distribuição de valores.
- Worklist operacional de entidades de risco.
- Visualização interativa dos clusters do grafo de fraude (com sliders para ajustar sensibilidade).
- Painel do modelo de ML: distribuição dos scores de risco, thresholds dos tiers, importância de features e tabela de transações pontuadas — só aparece se o modelo já tiver sido treinado.

## ▶️ Como executar

```bash
# 1. Instalar dependências (ou usar o Dev Container já configurado)
pip install -r requirements.txt

# 2. Treinar o modelo de risco (gera risk_model.joblib e risk_scores_test.csv)
python train_risk_model.py

# 3. Rodar o dashboard
streamlit run fraud_dashboard.py
```

## 🛠️ Stack de tecnologias

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3 |
| Análise de dados | Pandas, NumPy |
| Machine Learning | scikit-learn (HistGradientBoostingClassifier, permutation importance) |
| Análise de rede/grafo | NetworkX |
| Visualização | Plotly, Streamlit |
| Persistência de modelo | joblib |
| Ambiente | Dev Containers (Docker) |

## 💡 Principais aprendizados e destaques técnicos

- Identificação e correção de um vazamento de dados sutil (taxa de risco calculada sobre o período completo) — o tipo de erro que infla artificialmente a performance do modelo em desenvolvimento e quebra em produção.
- Validação temporal (out-of-time) como prática mais rigorosa que split aleatório para problemas com entidades recorrentes.
- Modelagem de decisão em múltiplos tiers, alinhando o output do modelo a uma política de negócio realista — em vez de tratar a saída do modelo como um fim em si.
- Combinação de detecção baseada em regras/grafo (explicável, rápida de implantar) com um modelo estatístico (mais preciso, mas menos transparente) — abordagem de defesa em camadas comum em sistemas de risco de pagamento.
