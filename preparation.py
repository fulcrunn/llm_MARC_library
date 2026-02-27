import os
import json
from pymarc import map_xml
from pymarc import Record
import fitz  # PyMuPDF para PDF
from docx import Document
from tqdm import tqdm

# ==================== CONFIG ====================
MARC_FOLDER = '/dataset/chunk/'
PDF_FOLDER  = '/dataset/pdfs_catalogacao/'
#DOC_FOLDER  = '/dataset/docs_catalogacao/'
OUTPUT_JSONL = '/dataset/train_dataset.jsonl'
# ================================================

# Limite para teste (ex: processe só 50k registros primeiro)
MAX_RECORDS = 10000   # aumente depois que testar
# ===============================================
data = []

# ==================== 1. PROCESSAMENTO DOS REGISTROS MARC ====================
def format_marc_record(record):
    # === Extração do Autor e Data Augmentation (Truque da Inversão) ===
    author_prompt = ''
    author_field = record.get('100')
    # === Extração do Autor e Data Augmentation (50/50) ===
    author_prompt = ''
    author_field = record.get('100')
    if author_field:
        a = author_field.get('a', '')
        c = author_field.get('c', '')
        q = author_field.get('q', '')
        d = author_field.get('d', '')
        
        # O autor completo na ordem que está no MARC (ex: Silva, Mario)
        author_original = f"{a} {c} {q} {d}".strip().replace('  ', ' ')
        author_prompt = author_original
        
        # Joga uma moeda: 50% de chance de "desinverter" o nome no prompt
        if ',' in a and random.choice([True, False]):
            partes = a.split(',', 1)
            sobrenome = partes[0].strip()
            nome = partes[1].strip()
            author_prompt = f"{nome} {sobrenome} {c} {q} {d}".strip().replace('  ', ' ')

    # === Título 245 a/b/c ===
    title_field = record.get('245')
    title = title_field.get('a', 'Sem título') if title_field else 'Sem título'
    subtitle = title_field.get('b', '') if title_field else ''
    responsibility = title_field.get('c', '') if title_field else ''

    full_title = title
    if subtitle:
        full_title += f" : {subtitle}"
    if responsibility:
        full_title += f" / {responsibility}"

    # === Ano (prioriza 260, fallback 264) ===
    year = ''
    if record.get('260') and 'c' in record['260']:
        year = record['260']['c']
    elif record.get('264') and 'c' in record['264']:
        year = record['264']['c']

    # === Extras úteis (edição e imprenta) ===
    edition = record['250'].get('a', '') if record.get('250') else ''
    imprint = ''
    if record.get('260'):
        loc = record['260'].get('a', '')
        pub = record['260'].get('b', '')
        imprint = f"{loc} : {pub}, {year}".strip(' :,')
    elif record.get('264'):
        loc = record['264'].get('a', '')
        pub = record['264'].get('b', '')
        imprint = f"{loc} : {pub}, {year}".strip(' :,')

    # === PROMPT COM TODAS AS REGRAS UFPR 2025 ===
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
- 600 ? ?: assuntos, usar subcampos a para assunto principal, x para subdivisão de assunto, z para localidade e y para forma de assunto (ex: "650 ? $a História $x Brasil $z Paraná")
- 650 ? ?: assuntos, usar LCSH da LC ou DeCS da BIREME, Autoridades da Fundação Biblioteca Nacional, com subcampos a, x, z, y conforme aplicável, em português brasileiro (ex: "650 ? $a História $x Brasil $z Paraná")
- 700 ? ?: autores secundários, usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar apenas o 710
- 710 ? ?: autores corporativos, usar subcampos a para nome da entidade, c para data de criação, q para qualificação e d para data de extinção (ex: "710 2 $a Universidade Federal do Paraná. $c 1912-")
- 740 ? ?: títulos relacionados, usar subcampo a para título e subcampo d para data de criação (ex: "740 0 $a História do Paraná."). Não usar o 730.

Gere o registro MARC21 completo para este livro aplicando todas as regras acima. 
ATENÇÃO ÀS SEGUINTES TAREFAS INTELECTUAIS:
1. Você deve deduzir e classificar os assuntos (tags 650) em português brasileiro com base no título e autor da obra.
2. Se não houver informação disponível para um campo específico (ex: sem indicação de edição), apenas omita a tag do registro final. Não gere campos vazios.

Título completo: {full_title}
Autor: {author_prompt}
Ano: {year}
Edição: {edition}
Imprenta: {imprint}

Responda **APENAS** com o registro MARC completo (todos os campos necessários, com indicadores e subcampos exatos).
<|im_end|>
<|im_start|>assistant
{str(record)}
<|im_end|>"""

    return {"text": prompt}


# ====================== 2. XMLs ======================
print("Processando MARC XML (streaming, baixo uso de memória)...")
count = 0
for filename in os.listdir(MARC_FOLDER):
    if filename.endswith('.xml') or filename.endswith('.xml.gz'):
        path = os.path.join(MARC_FOLDER, filename)
        print(f"Processando {filename}...")

        def process_record(record):
            global count
            if count >= MAX_RECORDS:
                return
            data.append(format_marc_record(record))
            count += 1
            if count % 10000 == 0:
                print(f"  → {count} registros processados")

        map_xml(process_record, path)  # streaming perfeito para arquivos grandes!

# ====================== 2. PDFs ======================
print("Processando PDFs...")
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith('.pdf'):
        path = os.path.join(PDF_FOLDER, filename)
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        # Divide em chunks de ~800 tokens
        chunks = [text[i:i+6000] for i in range(0, len(text), 6000)]
        for chunk in chunks:
            data.append({
                "text": f"<|im_start|>user\nExplique as regras de catalogação MARC a partir deste trecho do documento:\n{chunk}\n<|im_end|>\n<|im_start|>assistant\n{chunk}\n<|im_end|>"
            })

'''
# ====================== 3. DOCs ======================
print("Processando DOC/DOCX...")
for filename in os.listdir(DOC_FOLDER):
    if filename.endswith(('.doc', '.docx')):
        path = os.path.join(DOC_FOLDER, filename)
        doc = Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        chunks = [text[i:i+6000] for i in range(0, len(text), 6000)]
        for chunk in chunks:
            data.append({
                "text": f"<|im_start|>user\nResuma as melhores práticas de catalogação MARC deste documento:\n{chunk}\n<|im_end|>\n<|im_start|>assistant\n{chunk}\n<|im_end|>"
            })
'''
# ====================== 4. SALVAR JSONL ======================
# Salva o dataset
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Pronto! Dataset criado com {len(data)} exemplos → {OUTPUT_JSONL}")