import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =====================================================
# CONFIGURAÇÕES DE CAMINHO
# =====================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

print("Iniciando o carregamento do Mistral PURO (Baseline)...")

# =====================================================
# 1. CARREGAR O MODELO BASE (Sem o adaptador LoRA)
# =====================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# ATENÇÃO: O tokenizer agora vem do modelo original, não da sua pasta outputs
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval() 

print("✅ Modelo puro carregado com sucesso!\n")

# =====================================================
# 2. DADOS DO LIVRO PARA TESTE
# =====================================================
titulo_teste = "A Inteligência Artificial na Ciência da Informação : impactos na catalogação"
autor_teste = "Silva, Maria Clara"
ano_teste = "2026"
edicao_teste = "1. ed."
imprenta_teste = "Curitiba : Editora UFPR, 2026"

# =====================================================
# 3. CONSTRUÇÃO DO PROMPT
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
# 4. GERAÇÃO DO REGISTRO
# =====================================================
print(f"📚 Baseline tentando catalogar a obra: {titulo_teste}\n")
print("-" * 60)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512, 
        temperature=0.1,    
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

resposta = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(resposta.strip())
print("-" * 60)