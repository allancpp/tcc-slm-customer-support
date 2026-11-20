
# Assistente de Atendimento ao Cliente com SLM: Classificação de Intenções e Geração de Respostas via LoRA

#### Aluno: [Allan Ferreira](https://github.com/SEU_USUARIO_GITHUB)
#### Orientador: [Nome do Orientador](https://github.com/LINK_ORIENTADOR)

---

Trabalho apresentado ao curso [GenAI e LLM MASTER](https://ica.ele.puc-rio.br/cursos/ia-generativa-large-language-models/) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [Link para o código](https://github.com/SEU_USUARIO_GITHUB/slm-customer-support-lora)
- Trabalhos relacionados:
  - [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  - [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
  - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

### Resumo

Este trabalho investiga a viabilidade técnica e econômica da substituição de Modelos de Linguagem Grandes (LLMs) acessados via API por Modelos de Linguagem Pequenos (*Small Language Models* — SLMs) adaptados localmente para a automação de atendimento ao cliente. Utilizando o conjunto de dados público *Bitext Customer Support*, avalia-se o desempenho do modelo `Qwen2.5-1.5B-Instruct` na tarefa dupla de classificação de intenções e geração de respostas de suporte. A metodologia compara o SLM em cenários *zero-shot* e *few-shot*, um modelo de fronteira via API (`GPT-4o-mini`) como referência de qualidade e custo, e o SLM submetido ao ajuste fino eficiente por parâmetros (*Parameter-Efficient Fine-Tuning* — PEFT) utilizando a técnica LoRA (*Low-Rank Adaptation*) em ambiente gratuito com GPU T4. *(Os resultados quantitativos finais serão consolidados após a conclusão dos experimentos).*

### Abstract

This work investigates the technical and economic feasibility of replacing API-based Large Language Models (LLMs) with locally adapted Small Language Models (SLMs) for customer support automation. Using the public *Bitext Customer Support* dataset, we evaluate the `Qwen2.5-1.5B-Instruct` model on the dual task of intent classification and support response generation. The methodology benchmarks the base SLM under zero-shot and few-shot settings, a frontier API model (`GPT-4o-mini`) as a quality and cost baseline, and the SLM fine-tuned via Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) on a free T4 GPU environment. *(Final quantitative results will be consolidated upon experiment completion).*

### 1. Introdução

O atendimento ao cliente em escala envolve o processamento diário de milhares de solicitações recorrentes, como cancelamento de pedidos, rastreamento de entregas, reembolsos e alterações cadastrais. Embora Modelos de Linguagem Grandes (LLMs) comerciais ofereçam alta capacidade de compreensão e geração textual, seu uso contínuo via API acarreta custos operacionais elevados, latência de rede e dependência de terceiros.

Neste contexto, este projeto busca responder à seguinte questão de pesquisa: **um Small Language Model (SLM) de aproximadamente 1,5 bilhão de parâmetros, especializado por meio da técnica LoRA, consegue atingir desempenho próximo ao de um LLM comercial na classificação de intenções e resposta ao cliente, com custo computacional significativamente inferior?**

### 2. Modelagem

A arquitetura experimental está estruturada em quatro etapas reprodutíveis:

1. **Curadoria e Divisão de Dados:** Utilização do dataset *Bitext Customer Support* (~27.000 mensagens rotuladas por intenção e resposta de referência), com isolamento de um conjunto de teste fixo e balanceado (*Gold Test Set*) de 300 mensagens para garantir comparação justa entre todos os modelos.
2. **Linha de Base (*Baseline* SLM):** Avaliação do modelo `Qwen2.5-1.5B-Instruct` sem ajuste fino nos regimes *zero-shot* e *few-shot*.
3. **Benchmark com LLM Comercial:** Avaliação do mesmo conjunto de teste utilizando o modelo `GPT-4o-mini` via API para estabelecimento da referência superior de qualidade, latência e custo por requisição.
4. **Adaptação Eficiente via LoRA:** Ajuste fino supervisionado (*Supervised Fine-Tuning* — SFT) do `Qwen2.5-1.5B-Instruct` utilizando as bibliotecas `transformers` e `peft` em GPU NVIDIA T4, seguido pela disponibilização de uma interface interativa em `Gradio`.

*(A formulação matemática detalhada dos adaptadores LoRA e os hiperparâmetros finais serão documentados na etapa de treinamento).*

### 3. Resultados

*(Seção em construção — será preenchida no Passo 5 com as tabelas comparativas de Acurácia, F1-Macro, ROUGE-L, Latência média e Custo estimado por 1.000 requisições, além da matriz de confusão).*

### 4. Conclusões

*(Seção em construção — será preenchida no Passo 5 após a análise dos trade-offs entre qualidade e custo).*

### 5. Referências

- HU, Edward J. et al. **LoRA: Low-Rank Adaptation of Large Language Models**. International Conference on Learning Representations (ICLR), 2022. Disponível em: https://arxiv.org/abs/2106.09685.
- QWEN TEAM. **Qwen2.5 Technical Report**. arXiv preprint arXiv:2412.15115, 2024. Disponível em: https://arxiv.org/abs/2412.15115.
- BITEXT. **Bitext Customer Support LLM Chatbot Training Dataset**. Hugging Face Datasets, 2024. Disponível em: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset.

### 6. Instalação

Clone o repositório e instale as dependências com versões fixadas em um ambiente virtual Python:

```bash
git clone [https://github.com/SEU_USUARIO_GITHUB/slm-customer-support-lora.git](https://github.com/SEU_USUARIO_GITHUB/slm-customer-support-lora.git)
cd slm-customer-support-lora
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
---

Matrícula: 252.100.240

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Inteligência Artificial Generativa & Large Language Models*
