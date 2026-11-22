import os
import re
import json
import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "lora_adapters/final_model")
INTENTS_PATH = os.getenv("INTENTS_PATH", "data/valid_intents.json")

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    valid_intents = json.load(f)

INTENTS_STR = ", ".join(valid_intents)

SYSTEM_PROMPT = f"""You are a customer support AI assistant.
Your task is to:
1. Classify the customer's message into EXACTLY ONE of the following 27 valid intents:
[{INTENTS_STR}]
2. Write a helpful, polite, and concise customer support response.

Respond ONLY with a valid JSON object using this exact format:
{{"intent": "<exact_intent_name>", "response": "<your_reply>"}}"""


def load_merged_model():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


tokenizer, model = load_merged_model()


def parse_model_output(raw_text: str):
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    try:
        data = json.loads(cleaned)
        intent = str(data.get("intent", "unknown")).strip().lower()
        response = str(data.get("response", "")).strip()
        return intent, response, json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        found_intent = "unknown"
        for v_intent in valid_intents:
            if v_intent in cleaned.lower():
                found_intent = v_intent
                break
        return found_intent, cleaned, cleaned


def predict_support(customer_message: str):
    if not customer_message or not customer_message.strip():
        return "⚠️ Digite uma mensagem", "", "0.000 s", "{}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": customer_message.strip()}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.perf_counter() - t0

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_out = tokenizer.decode(generated_ids, skip_special_tokens=True)
    intent, reply, raw_json = parse_model_output(raw_out)

    return f"🎯 {intent}", reply, f"{latency:.3f} s", raw_json


def build_demo():
    with gr.Blocks(theme=gr.themes.Soft(), title="SLM Customer Support Assistant (LoRA)") as demo:
        gr.Markdown(
            """
            # 🤖 Assistente de Suporte ao Cliente — Qwen2.5-1.5B + LoRA
            **TCC Pós-Graduação IA Generativa & LLMs (PUC-Rio)** | Classificação de 27 Intenções + Resposta Estruturada (JSON)
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                msg_input = gr.Textbox(
                    label="Mensagem do Cliente (Customer Message)",
                    placeholder="Ex: I want to cancel my order #4829 and get a refund...",
                    lines=4
                )
                submit_btn = gr.Button("Classificar e Gerar Resposta", variant="primary")
                gr.Examples(
                    examples=[
                        ["I need to cancel my order right away, I bought the wrong item."],
                        ["Where is my package? It hasn't arrived yet and the tracking isn't updating."],
                        ["How do I change the shipping address on my account?"],
                        ["I was charged twice on my credit card for the same purchase!"],
                        ["Can I speak to a human customer service representative please?"]
                    ],
                    inputs=msg_input,
                    label="Exemplos Rápidos (Clique para preencher)"
                )
            with gr.Column(scale=1):
                intent_output = gr.Textbox(label="Intenção Classificada (1 de 27 classes)")
                reply_output = gr.Textbox(label="Resposta Sugerida ao Cliente", lines=5)
                latency_output = gr.Textbox(label="Latência de Inferência")
                json_output = gr.Code(label="Saída Estruturada (JSON Bruto)", language="json")

        submit_btn.click(
            fn=predict_support,
            inputs=msg_input,
            outputs=[intent_output, reply_output, latency_output, json_output]
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()