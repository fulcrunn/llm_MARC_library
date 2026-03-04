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
titulo_teste = "Making poor man's guitars: cigar box guitars, the frying pan banjo, and other DIY instruments"
autor_teste = "Shane Speal"
ano_teste = "2018"
edicao_teste = ""
imprenta_teste = " Mount Joy:Fox Chapel Publishing,"

prompt = f"""<|im_start|>user
Voce e um catalogador profissional do SiBi/UFPR e deve seguir **rigorosamente** o Manual de Catalogacao do SiBi/UFPR versao 2025.

Regras obrigatorias:
  tag indicador 1 indicador 2: descricao da tag
- 090 0 ?: codigo de classificacao, usar Classificacao Decimal de Dewey (CDD) ou Classificacao Decimal Universal (CDU) conforme disponivel; se ambos, priorizar CDD  
- 100 ? ?: usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar 110 ou 111
- 240 ? ?: usar para titulos uniformes, com subcampo a para titulo e subcampo d para data de criacao (ex: "Brasil. Ministerio da Educacao. Secretaria de Educacao Superior. Universidade Federal do Parana. Setor de Ciencias Humanas, Letras e Artes. Departamento de Historia. Curso de Historia.")
- 245 ? ?: titulo principal no subcampo a, subtitulo (geralmente vem depois do sinal de :) no subcampo b, responsabilidade no subcampo c ; transcrever exatamente sem pontuacao extra
- 250 ? ?: edicao ignorar reimpressoes; so registrar 1ª edicao se aparecer explicitamente
- 260/264 ? ?: impreta, usar [S.l.] quando nao houver local, [s.n.] quando nao houver editora; datas aproximadas entre colchetes [19--], [201-], etc.
- 300 ? ?: descricao fisica, usar subcampos a para extensao (ex: "300 p."), b para ilustracoes (ex: "il.") e c para dimensoes (ex: "21 cm")
- 490 0 ?: serie, usar subcampo a para titulo da serie e subcampo v para numero da serie (ex: "490 0 $a Colecao UFPR. $v 10")
- 500 ? ?: notas gerais, usar subcampo a para texto da nota (ex: "500 $a Inclui bibliografia.")
- 504 ? ?: bibliografia, usar subcampo a para texto da nota (ex: "504 $a Bibliografia: p. 290-300.")
- 505 ? ?: sumario, usar subcampo a para texto do sumario (ex: "505 $a Capitulo 1: Introducao -- Capitulo 2: Metodologia.")
- 590 ? ?: notas locais, usar subcampo a para texto da nota (ex: "590 $a Exemplar disponivel apenas para consulta local.")
- 600 ? ?: assuntos, usar subcampos a para assunto principal, x para subdivisao de assunto, z para localidade e y para forma de assunto
- 650 ? ?: assuntos, usar LCSH da LC ou DeCS da BIREME, Autoridades da Fundacao Biblioteca Nacional, com subcampos a, x, z, y conforme aplicavel, em portugues brasileiro
- 700 ? ?: autores secundarios, usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar apenas o 710
- 710 ? ?: autores corporativos, usar subcampos a para nome da entidade, c para data de criacao, q para qualificacao e d para data de extincao
- 740 ? ?: titulos relacionados, usar subcampo a para titulo e subcampo d para data de criacao. Nao usar o 730.

Gere o registro MARC21 completo para este livro aplicando todas as regras acima. 
Se o primeiro indicador estiver ausente, adicione um - no lugar.
Se o segundo indicador estiver ausente, adicione um - no lugar.
ATENCAO AS SEGUINTES TAREFAS INTELECTUAIS:
1. Voce deve deduzir os assuntos (tags 650) em com base na assossiacao entre titulo e autor da obra, apresente a traducao para o portugues.
2. Se nao houver informacao disponivel para um campo especifico (ex: sem indicacao de edicao), apenas omita a tag do registro final. Nao gere campos vazios.
3. O tag 082 deve apresentar o numero da Dewey Decimal Classification correspondente ao assunto.

Titulo completo: {titulo_teste}
Autor: {autor_teste}
Ano: {ano_teste}
Edicao: {edicao_teste}
Imprenta: {imprenta_teste}

Responda **APENAS** com o registro MARC completo (todos os campos necessarios, com indicadores e subcampos exatos).
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