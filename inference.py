import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =====================================================
# 1. CONFIGURAÇÕES
# =====================================================
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_DIR = "./outputs"  # Pasta onde estão os arquivos salvos pelo treinamento

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
# 3. VESTIR O MODELO (FUSÃO)
# =====================================================
print("Injetando o adaptador MARC21 no cérebro do Mistral...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# =====================================================
# 4. PREPARAR A PERGUNTA (PROMPT)
# =====================================================
# IMPORTANTE: A formatação e as tags <|im_start|> devem ser 
# rigorosamente iguais ao que você usou no preparation.py
prompt = """<|im_start|>user
Você é um catalogador profissional do SiBi/UFPR e deve seguir **rigorosamente** o Manual de Catalogação do SiBi/UFPR versão 2025.

Regras obrigatórias:
- Campo 040: sempre 040 ## |a BR-CuUPA |b por |c BR-CuUPA
- 245: nunca usar subcampo |h (DGM) para livros impressos
- 250 (edição): ignorar reimpressões; só registrar 1ª edição se aparecer explicitamente
- 260/264 (imprenta): usar [S.l.] quando não houver local, [s.n.] quando não houver editora; datas aproximadas entre colchetes [19--], [201-], etc.
- Autoridades (100/110/111): usar subcampos a, c, q, d conforme forma autorizada
- Título (245): transcrever exatamente a/b/c, sem pontuação extra dentro dos subcampos
- Notas (5XX): seguir exemplos do manual (502 para teses, 520 para resumos, 591 para notas locais, etc.)
- Subjects (650): usar LCSH da LC

Deixe os assuntos em português.
Gere o registro MARC21 completo para este livro aplicando todas as regras acima:

Título completo: O Senhor dos Anéis : A Sociedade do Anel
Autor: Tolkien, J. R. R.
Ano: 2020
Edição: 1
Imprenta: São Paulo : Martins Fontes, 2020
Assuntos: Literatura Fantástica; Ficção
<|im_end|>
<|im_start|>assistant
"""

# Converter o texto para tensores e enviar para a GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# =====================================================
# 5. GERAR O REGISTRO
# =====================================================
print("\nGerando registro MARC21...\n")
# Temperature baixa (0.1) faz o modelo ser mais robótico, preciso e menos "criativo", ideal para catalogação
outputs = model.generate(
    **inputs, 
    max_new_tokens=400, 
    temperature=0.1, 
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decodificar e cortar o prompt da resposta final
resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
registro_marc = resposta_completa.split("<|im_start|>assistant")[-1].strip()

print(registro_marc)