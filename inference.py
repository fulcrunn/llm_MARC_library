import torch
import re
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
print("Injetando o adaptador MARC21 no cérebro do Mistral...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# =====================================================
# 4. FUNÇÃO DE FORMATAÇÃO VISUAL
# =====================================================
def formatar_marc_personalizado(texto_marc):
    """
    Converte o formato padrão gerado pelo modelo (baseado no str do pymarc)
    para o formato visual solicitado pelo usuário com pipes (|) e underlines (_).
    """
    linhas = texto_marc.strip().split('\n')
    resultado = []
    
    for linha in linhas:
        if not linha.strip():
            continue
            
        # 1. Remove o '=' inicial do pymarc, se existir
        if linha.startswith('='):
            linha = linha[1:]
            
        # 2. Substitui o separador de subcampo ($) pelo pipe (|)
        linha = linha.replace('$', ' |')
        
        # 3. Trata os indicadores (posição 4 e 5 da string)
        if len(linha) >= 6 and linha[0:3].isdigit():
            tag = linha[0:3]
            # Campos de controle (00X) não têm indicadores
            if not tag.startswith('00'):
                # Extrai os indicadores e substitui contra-barras ou espaços vazios por '_'
                indicadores = linha[4:6].replace('\\', '_').replace('  ', '__').replace(' ', '_')
                # Reconstrói a linha com os novos indicadores
                linha = linha[:4] + indicadores + linha[6:]
                
        resultado.append(linha)
        
    return '\n'.join(resultado)

# =====================================================
# 5. PREPARAR A PERGUNTA (PROMPT)
# =====================================================
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

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# =====================================================
# 6. GERAR E FORMATAR O REGISTRO
# =====================================================
print("\nGerando registro MARC21...\n")

outputs = model.generate(
    **inputs, 
    max_new_tokens=400, 
    temperature=0.1, 
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
texto_bruto_gerado = resposta_completa.split("<|im_start|>assistant")[-1].strip()

# Aplica a nossa formatação customizada
registro_final = formatar_marc_personalizado(texto_bruto_gerado)

print("=== REGISTRO MARC GERADO E FORMATADO ===")
print(registro_final)