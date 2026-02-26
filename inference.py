import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =====================================================
# 1. CONFIGURAÇÕES
# =====================================================
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_DIR = "./outputs"  

# =====================================================
# 2. CARREGAR TOKENIZADOR E MODELO BASE
# =====================================================
print("Carregando o Tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

print("Carregando o Mistral original em 4-bits...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# =====================================================
# 3. VESTIR O MODELO 
# =====================================================
print("Injetando o adaptador MARC21 no cerebro do Mistral...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# =====================================================
# 4. FUNÇÃO DE FORMATAÇÃO VISUAL
# =====================================================
def formatar_marc_personalizado(texto_marc):
    linhas = texto_marc.strip().split('\n')
    resultado = []
    
    for linha in linhas:
        if not linha.strip():
            continue
            
        if linha.startswith('='):
            linha = linha[1:]
            
        linha = linha.replace('$', ' |')
        
        if len(linha) >= 6 and linha[0:3].isdigit():
            tag = linha[0:3]
            if not tag.startswith('00'):
                indicadores = linha[4:6].replace('\\', '_').replace('  ', '__').replace(' ', '_')
                linha = linha[:4] + indicadores + linha[6:]
                
        resultado.append(linha)
        
    return '\n'.join(resultado)

# =====================================================
# 5. PREPARAR A PERGUNTA (PROMPT CORRIGIDO)
# =====================================================
prompt = """<|im_start|>user
Gere para mim o MARC21 completo da seguinte obra:

Título completo: O segredo de Luísa
Autor: Fernando Dolabela
Ano: 2008 
Imprenta: São Paulo : Sextante, 2008

<|im_end|>
<|im_start|>assistant
"""

# =====================================================
# 6. GERAR E FORMATAR O REGISTRO
# =====================================================
print("\nGerando registro MARC21...\n")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=800,  
    temperature=0.1, 
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Corta os tokens do prompt e decodifica estritamente a nova resposta
tamanho_do_prompt = inputs.input_ids.shape[-1]
tokens_gerados = outputs[0][tamanho_do_prompt:]
texto_bruto_gerado = tokenizer.decode(tokens_gerados, skip_special_tokens=True).strip()

# Aplica a nossa formatação customizada
registro_final = formatar_marc_personalizado(texto_bruto_gerado)

# Exibe na tela e salva em arquivo para evitar problemas com acentos no terminal
print(registro_final)

with open("resultado.txt", "w", encoding="utf-8") as f:
    f.write(registro_final)
    
print("\n=== Registro salvo com sucesso no arquivo 'resultado.txt' ===")