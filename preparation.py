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
    # === EXTRAÇÃO ENRIQUECIDA (conforme manual UFPR) ===
    # Título 245 a/b/c
    title_field = record.get('245')
    title = title_field.get('a', 'Sem título') if title_field else 'Sem título'
    subtitle = title_field.get('b', '') if title_field else ''
    responsibility = title_field.get('c', '') if title_field else ''

    full_title = title
    if subtitle:
        full_title += f" : {subtitle}"
    if responsibility:
        full_title += f" / {responsibility}"

    # Autor 100 (subcampos que você mencionou: a, c, q, d)
    author = ''
    author_field = record.get('100')
    if author_field:
        a = author_field.get('a', '')
        c = author_field.get('c', '')
        q = author_field.get('q', '')
        d = author_field.get('d', '')
        author = f"{a} {c} {q} {d}".strip().replace('  ', ' ')

    # Ano (prioriza 260, fallback 264)
    year = ''
    if record.get('260') and 'c' in record['260']:
        year = record['260']['c']
    elif record.get('264') and 'c' in record['264']:
        year = record['264']['c']

    # Subjects 650
    subjects = ''
    subject_field = record.get('650')
    if subject_field: #'a', 'x', 'z', 'y']
      a = subject_field.get('a','')
      x = subject_field.get('x','')
      z = subject_field.get('z','')
      y = subject_field.get('y','')
      subject = f"{a}; {x}; {z}; {y}".strip().replace('  ',' ')

    # Extras úteis (edição e imprenta)
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

    # === PROMPT COM REGRAS DO MANUAL 2025 ===
    prompt = f"""<|im_start|>user
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

Título completo: {full_title}
Autor: {author}
Ano: {year}
Edição: {edition}
Imprenta: {imprint}
Assuntos: {subjects}

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