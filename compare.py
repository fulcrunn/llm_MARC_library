import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =====================================================
# CONFIGURAÇÕES DE CAMINHO
# =====================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "./outputs"  # A pasta onde o seu modelo treinado foi salvo

print("Carregando a rede neural na GPU (Isso pode levar um minuto)...\n")

# =====================================================
# 1. CARREGAR O MODELO BASE (O cérebro original)
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
# 2. ACOPLAR O ADAPTADOR (O seu treinamento de 17h)
# =====================================================
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# =====================================================
# 3. DADOS DE TESTE INÉDITOS
# =====================================================
titulo_teste = "A Inteligência Artificial na Ciência da Informação : impactos na catalogação"
autor_teste = "Silva, Maria Clara"
ano_teste = "2026"
edicao_teste = "1. ed."
imprenta_teste = "Curitiba : Editora UFPR, 2026"

prompt = f"""<|im_start|>user
Você é um catalogador profissional do SiBi/UFPR e deve seguir **rigorosamente** o Manual de Catalogação do SiBi/UFPR versão 2025.

Regras obrigatórias:
  tag indicador 1 indicador 2: descrição da tag
- 090 0 ?: código de classificação, usar Classificação Decimal de Dewey (CDD) ou Classificação Decimal Universal (CDU) conforme disponível; se ambos, priorizar CDD  
- 100 ? ?: usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar 110 ou 111
- 240 ? ?: usar para títulos uniformes, com subcampo a para título e subcampo d para data de criação
- 245 ? ?: título principal no subcampo a, subtítulo no subcampo b, responsabilidade no subcampo c
- 250 ? ?: edição ignorar reimpressões; só registrar 1ª edição se aparecer explicitamente
- 260/264 ? ?: impreta, usar [S.l.] quando não houver local, [s.n.] quando não houver editora
- 300 ? ?: descrição física, usar subcampos a para extensão (ex: "300 p.")
- 490 0 ?: série, usar subcampo a para título da série e subcampo v para número da série
- 500 ? ?: notas gerais, usar subcampo a para texto da nota
- 504 ? ?: bibliografia, usar subcampo a para texto da nota
- 505 ? ?: sumário, usar subcampo a para texto do sumário
- 590 ? ?: notas locais, usar subcampo a para texto da nota
- 600 ? ?: assuntos, usar subcampos a para assunto principal, x para subdivisão de assunto, z para localidade
- 650 ? ?: assuntos, usar LCSH da LC ou DeCS da BIREME, Autoridades da Fundação Biblioteca Nacional em português brasileiro
- 700 ? ?: autores secundários, usar subcampos a, c, q, d conforme forma autorizada
- 710 ? ?: autores corporativos, usar subcampos a para nome da entidade, c para data de criação
- 740 ? ?: títulos relacionados, usar subcampo a para título e subcampo d para data de criação. Não usar o 730.

Gere o registro MARC21 completo para este livro aplicando todas as regras acima. 
ATENÇÃO ÀS SEGUINTES TAREFAS INTELECTUAIS:
1. Você deve deduzir e classificar os assuntos (tags 650) em português brasileiro com base no título e autor da obra.
2. Se não houver informação disponível para um campo específico, apenas omita a tag do registro final. Não gere campos vazios.

Título completo: {titulo_teste}
Autor: {autor_teste}
Ano: {ano_teste}
Edição: {edicao_teste}
Imprenta: {imprenta_teste}

Responda **APENAS** com o registro MARC completo (todos os campos necessários, com indicadores e subcampos exatos).
<|im_end|>
<|im_start|>assistant
"""

# Função auxiliar para gerar o texto e não repetir código
def gerar_catalogacao(modelo_ativo):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = modelo_ativo.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print("-" * 70)
print(f"📚 Obra em Análise: {titulo_teste}\n")

# =====================================================
# INFERÊNCIA 1: O MODELO TREINADO (COM ADAPTADOR)
# =====================================================
print("🤖 1. RESULTADO DO MISTRAL COM FINE-TUNING (O seu modelo):")
resposta_treinada = gerar_catalogacao(model)
print(resposta_treinada)
print("-" * 70)

# =====================================================
# INFERÊNCIA 2: O MODELO PURO (SEM ADAPTADOR)
# =====================================================
print("🧠 2. RESULTADO DO MISTRAL PURO (Desativando o treinamento):")
# O comando genialmente simples que "desliga" temporariamente o LoRA
with model.disable_adapter():
    resposta_pura = gerar_catalogacao(model)
print(resposta_pura)
print("-" * 70)