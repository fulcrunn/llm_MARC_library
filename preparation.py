import os
import json
import gzip
import random
from io import BytesIO

import pymupdf
from pymarc import parse_xml_to_array
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = os.path.expanduser("~/programas/llm_MARC_library")
MARC_FOLDER = os.path.join(BASE_DIR, "datasetsUnzip")
PDF_FOLDER = MARC_FOLDER
OUTPUT_JSONL = os.path.join(MARC_FOLDER, "train_dataset.jsonl")

# Limite máximo de registros MARC.
MAX_RECORDS = 5_000_000

# Tamanho dos chunks dos PDFs.
PDF_CHUNK_SIZE = 6000

# Extensões aceitas.
XML_EXTENSIONS = (".xml", ".xml.gz")


# ============================================================
# VALIDAÇÃO DOS CAMINHOS
# ============================================================

def validate_paths():
    """Valida os diretórios necessários antes de iniciar o processamento."""

    print("\n=== Configuração de caminhos ===")
    print(f"BASE_DIR:     {BASE_DIR}")
    print(f"MARC_FOLDER:  {MARC_FOLDER}")
    print(f"PDF_FOLDER:   {PDF_FOLDER}")
    print(f"OUTPUT_JSONL: {OUTPUT_JSONL}")

    if not os.path.isdir(MARC_FOLDER):
        raise FileNotFoundError(
            f"\nPasta de MARC não encontrada:\n{MARC_FOLDER}\n\n"
            "Verifique se o diretório existe e se o caminho está correto."
        )

    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(
            f"\nPasta de PDFs não encontrada:\n{PDF_FOLDER}"
        )

    # Garante que a pasta de saída exista.
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)


# ============================================================
# FORMATAÇÃO DO REGISTRO MARC
# ============================================================

def format_marc_record(record):
    """
    Converte um registro pymarc em um exemplo de treinamento.

    A estrutura do prompt original foi preservada.
    """

    # --------------------------------------------------------
    # Autor
    # --------------------------------------------------------

    author_prompt = ""
    author_field = record.get("100")

    if author_field:
        a = author_field.get("a", "") or ""
        c = author_field.get("c", "") or ""
        q = author_field.get("q", "") or ""
        d = author_field.get("d", "") or ""

        author_original = f"{a} {c} {q} {d}".strip()
        author_original = " ".join(author_original.split())

        author_prompt = author_original

        # 50% de chance de inverter "Sobrenome, Nome"
        if "," in a and random.choice([True, False]):
            partes = a.split(",", 1)
            sobrenome = partes[0].strip()
            nome = partes[1].strip()

            author_prompt = (
                f"{nome} {sobrenome} {c} {q} {d}"
            ).strip()

            author_prompt = " ".join(author_prompt.split())

    # --------------------------------------------------------
    # Título 245
    # --------------------------------------------------------

    title_field = record.get("245")

    title = (
        title_field.get("a", "Sem título")
        if title_field
        else "Sem título"
    )

    subtitle = (
        title_field.get("b", "")
        if title_field
        else ""
    )

    responsibility = (
        title_field.get("c", "")
        if title_field
        else ""
    )

    full_title = title

    if subtitle:
        full_title += f" : {subtitle}"

    if responsibility:
        full_title += f" / {responsibility}"

    # --------------------------------------------------------
    # Ano: prioriza 260$c e usa 264$c como fallback
    # --------------------------------------------------------

    year = ""

    if record.get("260") and "c" in record["260"]:
        year = record["260"].get("c", "")
    elif record.get("264") and "c" in record["264"]:
        year = record["264"].get("c", "")

    # --------------------------------------------------------
    # Edição
    # --------------------------------------------------------

    edition = (
        record["250"].get("a", "")
        if record.get("250")
        else ""
    )

    # --------------------------------------------------------
    # Imprenta
    # --------------------------------------------------------

    imprint = ""

    if record.get("260"):
        loc = record["260"].get("a", "")
        pub = record["260"].get("b", "")
        imprint = f"{loc} : {pub}, {year}".strip(" :,")

    elif record.get("264"):
        loc = record["264"].get("a", "")
        pub = record["264"].get("b", "")
        imprint = f"{loc} : {pub}, {year}".strip(" :,")

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""<|im_start|>user
Você é um catalogador profissional do SiBi/UFPR e deve seguir **rigorosamente** o Manual de Catalogação do SiBi/UFPR versão 2025.

Regras obrigatórias:
  tag indicador 1 indicador 2: descrição da tag
- 090 0 ?: código de classificação, usar Classificação Decimal de Dewey (CDD) ou Classificação Decimal Universal (CDU) conforme disponível; se ambos, priorizar CDD
- 100 ? ?: usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar 110 ou 111
- 240 ? ?: usar para títulos uniformes, com subcampo a para título e subcampo d para data de criação (ex: "Brasil. Ministério da Educação. Secretaria de Educação Superior. Universidade Federal do Paraná. Setor de Ciências Humanas, Letras e Artes. Departamento de História. Curso de História.")
- 245 ? ?: título principal no subcampo a, subtítulo no subcampo b, responsabilidade no subcampo c; transcrever exatamente sem pontuação extra
- 250 ? ?: edição ignorar reimpressões; só registrar 1ª edição se aparecer explicitamente
- 260/264 ? ?: imprenta, usar [S.l.] quando não houver local, [s.n.] quando não houver editora; datas aproximadas entre colchetes [19--], [201-], etc.
- 300 ? ?: descrição física, usar subcampos a para extensão (ex: "300 p."), b para ilustrações (ex: "il.") e c para dimensões (ex: "21 cm")
- 490 0 ?: série, usar subcampo a para título da série e subcampo v para número da série (ex: "490 0 $a Coleção UFPR. $v 10")
- 500 ? ?: notas gerais, usar subcampo a para texto da nota (ex: "500 $a Inclui bibliografia.")
- 504 ? ?: bibliografia, usar subcampo a para texto da nota (ex: "504 $a Bibliografia: p. 290-300.")
- 505 ? ?: sumário, usar subcampo a para texto do sumário (ex: "505 $a Capítulo 1: Introdução -- Capítulo 2: Metodologia.")
- 590 ? ?: notas locais, usar subcampo a para texto da nota (ex: "590 $a Exemplar disponível apenas para consulta local.")
- 600 ? ?: assuntos, usar subcampos a para assunto principal, x para subdivisão de assunto, z para localidade e y para forma de assunto
- 650 ? ?: usar LCSH da LC ou DeCS da BIREME, Autoridades da Fundação Biblioteca Nacional, com subcampos a, x, z, y conforme aplicável, em português brasileiro
- 700 ? ?: autores secundários, usar subcampos a, c, q, d conforme forma autorizada; se autor corporativo, usar apenas o 710
- 710 ? ?: autores corporativos, usar subcampos a para nome da entidade, c para data de criação, q para qualificação e d para data de extinção
- 740 ? ?: títulos relacionados, usar subcampo a para título e subcampo d para data de criação. Não usar o 730.

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


# ============================================================
# LEITURA DOS ARQUIVOS XML / XML.GZ
# ============================================================

def open_text_file(filepath):
    """
    Abre XML normal ou XML.GZ em modo texto.

    Retorna um context manager.
    """

    if filepath.lower().endswith(".gz"):
        return gzip.open(
            filepath,
            mode="rt",
            encoding="utf-8",
            errors="ignore",
        )

    return open(
        filepath,
        mode="r",
        encoding="utf-8",
        errors="ignore",
    )


def parse_single_marc_record(xml_chunk):
    """
    Faz o parsing de um único <record> MARCXML.

    O registro é colocado dentro de uma collection artificial,
    permitindo que o pymarc faça o parsing do fragmento.
    """

    cabecalho_fake = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<collection xmlns="http://www.loc.gov/MARC21/slim">'
    )

    rodape_fake = "</collection>"

    xml_to_parse = (
        cabecalho_fake
        + xml_chunk
        + rodape_fake
    )

    records = parse_xml_to_array(
        BytesIO(xml_to_parse.encode("utf-8"))
    )

    if records:
        return records[0]

    return None


def map_xml_robusto(process_record_func, filepath, pbar_xml, state):
    """
    Lê XML/XML.GZ registro por registro.

    Registros XML corrompidos são descartados individualmente,
    permitindo que o processamento continue.

    state é um dicionário compartilhado contendo:
        state["count"]
        state["errors"]
    """

    record_buffer = []
    in_record = False

    try:
        with open_text_file(filepath) as f:

            for line in f:

                if state["count"] >= MAX_RECORDS:
                    break

                # Início de um registro.
                if "<record" in line:

                    in_record = True
                    record_buffer = [line]

                    # Caso <record> e </record> estejam na mesma linha.
                    if "</record>" in line:
                        in_record = False

                elif in_record:

                    record_buffer.append(line)

                    if "</record>" in line:
                        in_record = False

                # Só processa quando encontrou o fechamento.
                if not in_record and record_buffer:

                    xml_chunk = "".join(record_buffer)
                    record_buffer = []

                    try:
                        record = parse_single_marc_record(xml_chunk)

                        if record is not None:
                            process_record_func(record)

                    except Exception:
                        state["errors"] += 1

                        # Mantém o processamento dos demais registros.
                        continue

                if state["count"] >= MAX_RECORDS:
                    break

    except Exception as exc:
        raise RuntimeError(
            f"Erro ao ler o arquivo XML '{filepath}': {exc}"
        ) from exc


# ============================================================
# DESCOBERTA E ORDENAÇÃO DOS XMLs
# ============================================================

def find_xml_files(folder):
    """
    Localiza arquivos .xml e .xml.gz.

    Os arquivos são ordenados pelo tamanho, do menor
    para o maior, mantendo a estratégia original.
    """

    arquivos_xml = []

    for filename in os.listdir(folder):

        lower = filename.lower()

        if lower.endswith(XML_EXTENSIONS):

            path = os.path.join(folder, filename)

            if os.path.isfile(path):
                arquivos_xml.append(path)

    arquivos_xml.sort(key=os.path.getsize)

    return arquivos_xml


# ============================================================
# PROCESSAMENTO DOS PDFs
# ============================================================

def process_pdfs(output_file, state):
    """
    Processa PDFs e adiciona seus chunks diretamente ao JSONL.

    Não acumula todos os exemplos em memória.
    """

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("\nNenhum PDF encontrado.")
        return

    print(f"\nProcessando {len(pdf_files)} PDF(s)...")

    for filename in tqdm(
        pdf_files,
        desc="Lendo PDFs",
        unit=" arq",
        colour="blue",
    ):

        path = os.path.join(PDF_FOLDER, filename)

        try:
            doc = pymupdf.open(path)

            try:
                text_parts = []

                for page in doc:
                    page_text = page.get_text("text")

                    if page_text:
                        text_parts.append(page_text)

                text = "\n".join(text_parts)

            finally:
                doc.close()

            for i in range(0, len(text), PDF_CHUNK_SIZE):

                chunk = text[i:i + PDF_CHUNK_SIZE]

                if not chunk.strip():
                    continue

                item = {
                    "text": (
                        "<|im_start|>user\n"
                        "Explique as regras de catalogação MARC "
                        "a partir deste trecho do documento:\n"
                        f"{chunk}\n"
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                        f"{chunk}\n"
                        "<|im_end|>"
                    )
                }

                output_file.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                state["output_count"] += 1

        except Exception as exc:

            state["pdf_errors"] += 1

            print(
                f"\n[AVISO] Não foi possível processar "
                f"o PDF '{filename}': {exc}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    validate_paths()

    print(
        "\nProcessando MARC XML "
        "(priorizando arquivos menores/UFPR)..."
    )

    arquivos_xml = find_xml_files(MARC_FOLDER)

    if not arquivos_xml:
        raise FileNotFoundError(
            f"\nNenhum arquivo .xml ou .xml.gz encontrado em:\n"
            f"{MARC_FOLDER}"
        )

    print(
        f"\nEncontrados {len(arquivos_xml)} arquivo(s) XML."
    )

    for path in arquivos_xml:
        size_mb = os.path.getsize(path) / (1024 ** 2)

        print(
            f"  - {os.path.basename(path)} "
            f"({size_mb:.2f} MB)"
        )

    # --------------------------------------------------------
    # Estado global controlado explicitamente.
    # --------------------------------------------------------

    state = {
        "count": 0,
        "errors": 0,
        "output_count": 0,
        "pdf_errors": 0,
    }

    # --------------------------------------------------------
    # Barra de progresso
    # --------------------------------------------------------

    pbar_xml = tqdm(
        total=MAX_RECORDS,
        desc="Extraindo XML",
        unit=" reg",
        colour="green",
    )

    # --------------------------------------------------------
    # Abre o JSONL uma única vez.
    #
    # Isso evita armazenar milhões de registros em `data`.
    # --------------------------------------------------------

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as output_file:

        def process_record(record):
            """
            Callback chamado para cada registro MARC válido.
            """

            if state["count"] >= MAX_RECORDS:
                return

            item = format_marc_record(record)

            output_file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

            state["count"] += 1
            state["output_count"] += 1

            pbar_xml.update(1)

        # ----------------------------------------------------
        # Processamento dos XMLs
        # ----------------------------------------------------

        for path in arquivos_xml:

            if state["count"] >= MAX_RECORDS:
                break

            filename = os.path.basename(path)

            print(f"\nLendo: {filename}...")

            map_xml_robusto(
                process_record,
                path,
                pbar_xml,
                state,
            )

            print(
                f"Registros válidos acumulados: "
                f"{state['count']:,}"
            )

    pbar_xml.close()

    # --------------------------------------------------------
    # PDFs
    # --------------------------------------------------------

    # PDFs são processados apenas depois dos XMLs.
    #
    # Como o arquivo JSONL foi fechado acima, usamos append.
    if os.path.isdir(PDF_FOLDER):

        with open(
            OUTPUT_JSONL,
            "a",
            encoding="utf-8",
        ) as output_file:

            process_pdfs(
                output_file,
                state,
            )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 60)

    print(
        f"Registros MARC processados: "
        f"{state['count']:,}"
    )

    print(
        f"Erros em registros XML: "
        f"{state['errors']:,}"
    )

    print(
        f"Exemplos escritos no JSONL: "
        f"{state['output_count']:,}"
    )

    print(
        f"Erros em PDFs: "
        f"{state['pdf_errors']:,}"
    )

    print(f"\nDataset: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()