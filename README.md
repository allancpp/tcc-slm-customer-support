# Especialização de Small Language Models (SLMs) via QLoRA para Classificação de Intenções e Resposta Estruturada em Suporte ao Cliente

#### Aluno: [Allan Ferreira](https://github.com/link_do_github)

#### Orientador: [Leonardo Forero Mendoza](https://github.com/leofome8)

---

Trabalho apresentado ao curso [GenAI e LLM MASTER](https://ica.ele.puc-rio.br/cursos/ia-generativa-large-language-models/) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".


 Notebooks do experimento:
* [01 — Análise Exploratória e Particionamento Estratificado](https://www.google.com/search?q=notebooks/01_eda_and_split.ipynb)
* [02 — Avaliação dos Baselines (Zero-Shot, Few-Shot e API)](https://www.google.com/search?q=notebooks/02_baselines_evaluation.ipynb)
* [03 — Fine-Tuning Eficiente com QLoRA](https://www.google.com/search?q=notebooks/03_lora_finetuning.ipynb)
* [04 — Avaliação Final, Otimização de Inferência e Gráficos](https://www.google.com/search?q=notebooks/04_evaluation_and_charts.ipynb)
* [05 — Demonstração Interativa com Gradio](https://www.google.com/search?q=notebooks/05_gradio_demo.ipynb)




---

### Resumo

A automação de centrais de atendimento ao cliente com Modelos de Linguagem de Grande Porte (LLMs) via APIs externas impõe desafios relevantes de privacidade de dados, dependência de terceiros e aderência a esquemas estruturados de negócio. Por outro lado, Modelos de Linguagem Pequenos (*Small Language Models* — SLMs) executados em infraestrutura própria oferecem soberania de dados e previsibilidade operacional, mas apresentam limitações significativas em classificação multiclasse complexa quando utilizados apenas com engenharia de *prompts*.

Este trabalho investiga empiricamente a eficácia do ajuste fino eficiente em parâmetros (*Quantized Low-Rank Adaptation* — QLoRA) aplicado ao modelo compacto `Qwen/Qwen2.5-1.5B-Instruct` (1,54 bilhão de parâmetros) para executar, em uma única passagem (*single-pass*), a classificação de mensagens de clientes em 27 intenções distintas e a geração simultânea de respostas humanizadas em formato JSON estrito. Utilizando uma partição estratificada rigorosa do *Bitext Customer Support Dataset* (4.000 exemplos de treino, 500 de validação e 300 de teste inédito), o modelo ajustado foi comparado a estratégias *Zero-Shot*, *Few-Shot* e ao modelo de fronteira `Gemini 3.5 Flash-Lite` via API.

Os resultados demonstram que o *fine-tuning* com QLoRA elevou a acurácia de classificação de **57,33%** (*Zero-Shot*) para **99,67%** (e **99,33%** após fusão de pesos), superando em **+7,67 pontos percentuais** o modelo via API (**92,00%**) e atingindo **99,32%** de *F1-Macro* e **0,3716** de *ROUGE-L*. Adicionalmente, a fusão permanente dos adaptadores LoRA em precisão FP16 (`merge_and_unload()`) reduziu a latência média de inferência de **13,620s** para **4,266s** (aceleração de **3,2x**), consolidando uma solução precisa, auditável e pronta para implantação em ambientes privados.

### Abstract

Automating customer support operations with Large Language Models (LLMs) via external APIs introduces critical challenges regarding data privacy, vendor dependency, and strict adherence to structured business schemas. Conversely, Small Language Models (SLMs) deployed on private infrastructure ensure data sovereignty and predictable operational costs, yet suffer from substantial performance degradation in complex multiclass classification when relying solely on prompt engineering.

This work empirically investigates the impact of Parameter-Efficient Fine-Tuning via Quantized Low-Rank Adaptation (QLoRA) applied to `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters) to simultaneously perform intent classification across 27 customer support categories and structured JSON response generation in a single pass. Using a stratified split of the Bitext Customer Support Dataset (4,000 training, 500 validation, and 300 unseen gold test samples), the fine-tuned SLM was benchmarked against Zero-Shot, Few-Shot, and the frontier `Gemini 3.5 Flash-Lite` API baseline.

Experimental results show that QLoRA fine-tuning increased classification accuracy from **57.33%** (Zero-Shot) to **99.67%** (**99.33%** after weight merging), outperforming the frontier API baseline (**92.00%**) by **+7.67 percentage points**, while achieving **99.32%** Macro-F1 and **0.3716** ROUGE-L. Furthermore, merging the LoRA adapters into FP16 base weights (`merge_and_unload()`) reduced average inference latency from **13.620s** to **4.266s** (**3.2x speedup**), demonstrating a highly accurate, privacy-preserving solution suitable for production deployment.

### 1. Introdução

O atendimento digital ao cliente exige sistemas capazes de interpretar solicitações em linguagem natural com alta precisão semântica, rotear chamados entre dezenas de fluxos operacionais e formular respostas empáticas, padronizadas e integráveis a sistemas corporativos via formatos estruturados como JSON. Embora LLMs comerciais de grande porte apresentem forte capacidade de generalização, o envio contínuo de mensagens de clientes para APIs externas acarreta riscos de conformidade (como a LGPD), custos variáveis atrelados ao volume de tokens e vulnerabilidade a indisponibilidades ou alterações de modelos de terceiros.

Nesse contexto, os *Small Language Models* (SLMs) — modelos abertos com menos de 2 bilhões de parâmetros — emergem como alternativa estratégica para implantação *on-premise* ou em nuvem privada (*VPC*), exigindo recursos computacionais modestos. Contudo, em tarefas de domínio específico com alto número de classes e restrições rígidas de formatação de saída, técnicas puramente baseadas em contexto (*Zero-Shot* e *Few-Shot*) frequentemente falham em manter consistência estrutural e discriminativa.

O objetivo deste trabalho é avaliar técnica e economicamente se a especialização de um SLM compacto (`Qwen2.5-1.5B-Instruct`) por meio de **QLoRA** permite superar modelos de fronteira via API na tarefa conjunta de **classificação de 27 intenções de suporte** e **geração de resposta em JSON**, analisando também o impacto da fusão de pesos (`merge_and_unload()`) na latência e no custo operacional de inferência.

### 2. Modelagem

#### 2.1. Base de Dados e Particionamento Estratificado

Utilizou-se o *Bitext Customer Support LLM Chatbot Training Dataset*, composto por 26.872 interações anotadas em **27 intenções distintas** de suporte ao cliente (tais como `cancel_order`, `track_order`, `get_refund`, `change_shipping_address`, `payment_issue`, entre outras). Para assegurar rigor experimental, ausência de vazamento de dados (*data leakage*) e reprodutibilidade (`seed = 42`), aplicou-se uma amostragem estratificada em três partições disjuntas:

* **Conjunto de Treinamento (`data/train_set_4000.json`):** 4.000 exemplos (~148 amostras por intenção).
* **Conjunto de Validação (`data/val_set_500.json`):** 500 exemplos (~18 a 19 amostras por intenção), utilizado para monitoramento da perda de validação e seleção do melhor *checkpoint*.
* **Conjunto de Teste Ouro (`data/test_set_300.json`):** 300 exemplos inéditos (~11 amostras por intenção), mantidos estritamente isolados para avaliação comparativa final.

#### 2.2. Formulação da Tarefa e Engenharia de Prompt

Todos os modelos foram avaliados sob o mesmo *System Prompt*, instruindo o assistente a retornar exclusivamente um objeto JSON válido contendo dois campos obrigatórios: `"intent"` (restrito à lista oficial das 27 intenções) e `"response"` (resposta educada e objetiva ao cliente). Foram avaliadas cinco configurações experimentais:

1. **Qwen2.5-1.5B (Zero-Shot):** Modelo base em precisão FP16 sem exemplos no contexto.
2. **Qwen2.5-1.5B (Few-Shot):** Modelo base em FP16 com exemplos de demonstração inseridos no *prompt* (*in-context learning*).
3. **Qwen2.5-1.5B + LoRA (4-bit Dinâmico):** Modelo base quantizado em 4-bit com adaptadores LoRA acoplados dinamicamente em tempo de inferência.
4. **Qwen2.5-1.5B + LoRA (FP16 Fundido):** Modelo ajustado após fusão permanente das matrizes LoRA aos pesos base em precisão FP16 (`merge_and_unload()`).
5. **Gemini 3.5 Flash-Lite (API):** *Baseline* comercial de fronteira avaliado via API externa sob o mesmo esquema de instrução.

#### 2.3. Arquitetura QLoRA e Hiperparâmetros de Treinamento

O ajuste fino foi conduzido sobre o `Qwen/Qwen2.5-1.5B-Instruct` utilizando quantização **4-bit NormalFloat (`nf4`)** com dupla quantização (*double quantization*) e adaptadores de baixo posto (**LoRA**). 

* **Rank e Escala:** `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`.
* **Módulos Alvo:** Todas as projeções lineares de atenção e MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Otimização e Agendamento:** Otimizador `paged_adamw_8bit`, taxa de aprendizado inicial de `2e-4` com decaimento cosseno (*cosine scheduler*), *warmup ratio* de `0.05`, tamanho de lote efetivo de `16` (*batch size* `4` com `4` passos de acumulação de gradiente), comprimento máximo de sequência de `512` tokens e `2 épocas` completas (`500` passos) em GPU NVIDIA T4 (15 GB VRAM).

### 3. Resultados

A avaliação quantitativa no *Gold Test Set* (300 mensagens inéditas) considerou cinco métricas: **Acurácia** e **F1-Macro** (para classificação multiclasse balanceada), **ROUGE-L** (para aderência lexical e estrutural da resposta gerada), **Latência Média por Mensagem (s)** e **Custo Estimado por 1.000 Mensagens (USD)**.

| Modelo | Acurácia (%) | F1-Macro (%) | ROUGE-L | Latência Média (s) | Custo / 1k msgs (USD)* |
| --- | --- | --- | --- | --- | --- |
| **Qwen2.5-1.5B (Zero-Shot)** | 57,33% | 43,21% | 0,2025 | 1,980 | $0,1925 |
| **Qwen2.5-1.5B (Few-Shot)** | 53,67% | 36,20% | 0,1914 | 1,927 | $0,1874 |
| **Qwen2.5-1.5B + LoRA (4-bit Dinâmico)** | **99,67%** | 96,27% | **0,3716** | 13,620 | $1,3242 |
| **Qwen2.5-1.5B + LoRA (FP16 Fundido)** | **99,33%** | **99,32%** | 0,3536 | 4,266 | $0,4147 |
| **Gemini 3.5 Flash-Lite (API)** | 92,00% | 88,73% | 0,2162 | **0,973** | **$0,0004** |

**Nota metodológica: Para os modelos locais executados em GPU NVIDIA T4 dedicada, o custo operacional foi estimado proporcionalmente ao tempo ocupado de computação considerando uma tarifa de referência de US$ 0,35/hora. Para o modelo via API, utilizou-se o custo real faturado por tokens de entrada e saída.*

#### 3.1. Análise Comparativa e Engenharia de Inferência

* **Salto de Desempenho com Especialização:** O *fine-tuning* QLoRA elevou a acurácia do `Qwen2.5-1.5B-Instruct` de **57,33%** para **99,67%** (299 acertos em 300 mensagens inéditas), superando o *baseline* comercial `Gemini 3.5 Flash-Lite` (**92,00%**) em **+7,67 pontos percentuais** e elevando o **F1-Macro para 99,32%** (ganho de **+10,59 p.p.** sobre a API).
* **Degradação no Few-Shot em Modelos Compactos:** A adição de exemplos no *prompt* (*Few-Shot*) reduziu a acurácia do modelo base de 57,33% para 53,67%, evidenciando que janelas de contexto extensas sem atualização de pesos dispersam a atenção em arquiteturas de 1,5B parâmetros.
* **Otimização de Latência via Fusão de Pesos (`merge_and_unload()`):** A execução inicial com o adaptador LoRA acoplado dinamicamente sobre o modelo quantizado em 4-bit apresentou latência média de **13,620s** devido ao *overhead* de dequantização e multiplicação separada das matrizes $A$ e $B$ a cada passo autoregressivo. Ao fundir permanentemente os pesos LoRA ao modelo base em precisão **FP16** (`merge_and_unload()`), a **latência média caiu para 4,266s — uma redução de 68,7% (aceleração de 3,2x)** — reduzindo o custo estimado de infraestrutura em GPU de **$1,3242** para **$0,4147** por mil mensagens, preservando **99,33% de acurácia** e alcançando o maior **F1-Macro (99,32%)** do experimento.

#### 3.2. Protótipo Interativo (Gradio)

Para validação qualitativa e demonstração em tempo real, foi implementada uma interface web em **Gradio** (`src/app.py` e `notebooks/05_gradio_demo.ipynb`) utilizando o modelo fundido em FP16, capaz de receber mensagens livres de clientes e retornar a intenção classificada, a resposta sugerida, o *payload* JSON estruturado e a latência medida por requisição.

### 4. Conclusões

Os experimentos comprovam que *Small Language Models* abertos especializados via QLoRA representam uma solução altamente competitiva e tecnicamente superior a modelos genéricos de fronteira via API para tarefas estruturadas de domínio específico. O modelo `Qwen2.5-1.5B-Instruct` ajustado atingiu **99,33% a 99,67% de acurácia** e **99,32% de F1-Macro** na classificação simultânea de 27 intenções e geração de respostas em JSON, eliminando riscos de exposição de dados sensíveis a terceiros e garantindo controle total sobre o ciclo de vida do modelo.

Adicionalmente, demonstrou-se que a etapa de engenharia de pós-treinamento — especificamente a fusão dos adaptadores LoRA em precisão nativa FP16 (`merge_and_unload()`) — é indispensável para viabilizar o *deploy* em produção, reduzindo em **68,7%** o tempo de resposta sem perda de capacidade preditiva. Como trabalhos futuros, sugere-se a avaliação de motores de inferência otimizados (como *vLLM* ou *llama.cpp* com quantização GGUF/AWQ) para reduzir a latência abaixo de 1 segundo em ambientes *on-premise*, bem como a integração do classificador a um pipeline de *Retrieval-Augmented Generation* (RAG) conectado à base de políticas internas da organização.


### 5. Instalação

Para reproduzir o ambiente localmente, clone o repositório, crie um ambiente virtual e instale as dependências listadas em `requirements.txt`:

```bash
git clone [https://github.com/link_do_repositorio.git](https://github.com/link_do_repositorio.git)
cd link_do_repositorio
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

Para execução em nuvem via **Google Colab**, recomenda-se selecionar o ambiente de execução com **GPU NVIDIA T4** (15 GB VRAM) e manter a estrutura de diretórios `data/`, `results/` e `lora_adapters/` na raiz `/content/`.

### 6. Como executar

#### 6.1. Reprodução dos Experimentos via Notebooks (Google Colab)

Execute os notebooks na ordem sequencial abaixo, respeitando os arquivos de entrada exigidos em cada etapa:

1. **`notebooks/01_eda_and_split.ipynb` (Análise Exploratória e Splits):**
* *Entradas:* Nenhuma (baixa o dataset diretamente do Hugging Face Hub).
* *Saídas geradas:* `data/train_set_4000.json`, `data/val_set_500.json`, `data/test_set_300.json`, `data/valid_intents.json` e `results/intent_distribution.png`.


2. **`notebooks/02_baselines_evaluation.ipynb` (Avaliação dos Baselines):**
* *Uploads necessários em `/content/`:* suba `test_set_300.json` e `valid_intents.json` para a pasta `data/` e configure a variável `GEMINI_API_KEY` nos *Secrets* do Colab.
* *Saídas geradas:* `results/preds_qwen_zeroshot.csv`, `results/preds_qwen_fewshot.csv` e `results/preds_gemini_flash.csv`.


3. **`notebooks/03_lora_finetuning.ipynb` (Treinamento QLoRA):**
* *Uploads necessários em `/content/`:* suba `train_set_4000.json`, `val_set_500.json` e `valid_intents.json` para a pasta `data/`.
* *Saídas geradas:* adaptadores treinados em `lora_adapters/final_model/` (compactados em `final_model.zip` na raiz `/content/`) e `results/training_history.json`.


4. **`notebooks/04_evaluation_and_charts.ipynb` (Avaliação Final, Fusão de Pesos e Gráficos):**
* *Uploads necessários em `/content/`:* suba `final_model.zip` na raiz `/content/`; `test_set_300.json` e `valid_intents.json` em `data/`; e `preds_qwen_zeroshot.csv`, `preds_qwen_fewshot.csv`, `preds_gemini_flash.csv` e `training_history.json` em `results/`.
* *Saídas geradas:* `results/preds_qwen_lora.csv`, `results/preds_qwen_lora_merged.csv`, `results/final_benchmark_summary.csv`, `results/lora_loss_curve.png`, `results/models_comparison.png` e `results/confusion_matrix_lora.png`.


5. **`notebooks/05_gradio_demo.ipynb` (Interface Web Interativa no Colab):**
* *Uploads necessários em `/content/`:* suba `final_model.zip` na raiz `/content/` e `valid_intents.json` na pasta `data/`.
* *Saída:* inicializa a aplicação Gradio e gera uma URL pública temporária (`share=True`) para inferência interativa.



#### 6.2. Execução Local da Interface Interativa (`src/app.py`)

Após extrair o arquivo `final_model.zip` para criar a pasta `lora_adapters/final_model/` na raiz do projeto, inicie a aplicação web via terminal:

```
unzip final_model.zip
python src/app.py
```

A interface estará acessível localmente em `[http://127.0.0.1:7860](http://127.0.0.1:7860)`.

---

Matrícula: 252.100.240

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Inteligência Artificial Generativa & Large Language Models*