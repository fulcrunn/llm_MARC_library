import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =====================================================
# CONFIGURAÇÕES DE CAMINHO
# =====================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "./outputs"  # A pasta onde o seu modelo treinado foi salvo

print("Iniciando o carregamento do modelo (Isso pode levar um minuto)...")

# =====================================================
# 1. CARREGAR O MODELO BASE EM 4-BIT (Obrigatório no QLoRA)
# =====================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# =====================================================
# 2. ACOPLAR O ADAPTADOR TREINADO (O Conhecimento MARC)
# =====================================================
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() # Coloca o modelo em modo de inferência (desliga o cálculo de gradientes)

print("✅ Modelo carregado com sucesso!\n")

# =====================================================
# 3. DADOS DO LIVRO PARA TESTE
# =====================================================
# Altere estes dados para testar obras diferentes para o seu artigo
titulo_teste = "Adaptive filter theory"
autor_teste = "Simon Haykin"
ano_teste = "1996"
edicao_teste = "3rd ed."
imprenta_teste = "Prentice Hall"

# =====================================================
# 4. CONSTRUÇÃO DO PROMPT EXATO DO TREINAMENTO
# =====================================================
prompt = f"""<|im_start|>user
Você é um catalogador profissional do SiBi/UFPR e deve seguir **rigorosamente** o Manual de Catalogação do SiBi/UFPR versão 2025.

Regras obrigatórias:
  tag indicador 1 indicador 2: descrição da tag
- 090 0 ?: código de classificação, usar Classificação Decimal de Dewey (CDD) ou Classificação Decimal Universal (CDU) conforme disponível; se ambos, priorizar CDD  
- 100 ? ?: usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar 110 ou 111
- 240 ? ?: usar para títulos uniformes, com subcampo a para título e subcampo d para data de criação (ex: "Brasil. Ministério da Educação. Secretaria de Educação Superior. Universidade Federal do Paraná. Setor de Ciências Humanas, Letras e Artes. Departamento de História. Curso de História.")
- 245 ? ?: título principal no subcampo a, subtítulo no subcampo b, responsabilidade no subcampo c; transcrever exatamente sem pontuação extra
- 250 ? ?: edição ignorar reimpressões; só registrar 1ª edição se aparecer explicitamente
- 260/264 ? ?: impreta, usar [S.l.] quando não houver local, [s.n.] quando não houver editora; datas aproximadas entre colchetes [19--], [201-], etc.
- 300 ? ?: descrição física, usar subcampos a para extensão (ex: "300 p."), b para ilustrações (ex: "il.") e c para dimensões (ex: "21 cm")
- 490 0 ?: série, usar subcampo a para título da série e subcampo v para número da série (ex: "490 0 $a Coleção UFPR. $v 10")
- 500 ? ?: notas gerais, usar subcampo a para texto da nota (ex: "500 $a Inclui bibliografia.")
- 504 ? ?: bibliografia, usar subcampo a para texto da nota (ex: "504 $a Bibliografia: p. 290-300.")
- 505 ? ?: sumário, usar subcampo a para texto do sumário (ex: "505 $a Capítulo 1: Introdução -- Capítulo 2: Metodologia.")
- 590 ? ?: notas locais, usar subcampo a para texto da nota (ex: "590 $a Exemplar disponível apenas para consulta local.")
- 600 ? ?: assuntos, usar subcampos a para assunto principal, x para subdivisão de assunto, z para localidade e y para forma de assunto
- 650 ? ?: assuntos, usar LCSH da LC ou DeCS da BIREME, Autoridades da Fundação Biblioteca Nacional, com subcampos a, x, z, y conforme aplicável, em português brasileiro
- 700 ? ?: autores secundários, usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar apenas o 710
- 710 ? ?: autores corporativos, usar subcampos a para nome da entidade, c para data de criação, q para qualificação e d para data de extinção
- 740 ? ?: títulos relacionados, usar subcampo a para título e subcampo d para data de criação. Não usar o 730.

Gere o registro MARC21 completo para este livro aplicando todas as regras acima. 
ATENÇÃO ÀS SEGUINTES TAREFAS INTELECTUAIS:
1. Você deve deduzir e classificar os assuntos (tags 650) em português brasileiro com base no título e autor da obra.
2. Se não houver informação disponível para um campo específico (ex: sem indicação de edição), apenas omita a tag do registro final. Não gere campos vazios.

Título completo: {titulo_teste}
Autor: {autor_teste}
Ano: {ano_teste}
Edição: {edicao_teste}
Imprenta: {imprenta_teste}

Responda **APENAS** com o registro MARC completo (todos os campos necessários, com indicadores e subcampos exatos).
<|im_end|>
<|im_start|>assistant
"""

# =====================================================
# 5. GERAÇÃO DO REGISTRO
# =====================================================
print(f"📚 Catalogando a obra: {titulo_teste}\n")
print("-" * 60)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512, # Limite de tamanho da resposta
        temperature=0.1,    # Baixa temperatura = respostas mais determinísticas e precisas (ideal para catalogação)
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

# Decodifica apenas a resposta gerada pelo modelo, ignorando o prompt gigante de entrada
resposta = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(resposta.strip())
print("-" * 60)