# Código para separar os MARC do arquivo gigante em pequenos arquivos individuias.

from lxml import etree
import pandas as pd
import os
import sys

input_file = "/workspace/inputs/final_output_file.xml"
output_prefix = 'marc_chunk_'
records_size = 0
records_limits = 100000
chunk_num = 5000
current_records = []

# Cabeçalho XML básico (ajuste se o seu arquivo tiver namespace específico)
header = '''<?xml version="1.0" encoding="UTF-8"?>
<collection xmlns="http://www.loc.gov/MARC21/slim">
'''
footer = '</collection>'
parser = etree.XMLParser(recover=True)
context = etree.iterparse(input_file, events=('end',), tag='{http://www.loc.gov/MARC21/slim}record', recover=True)  # ou sem namespace se não tiver

for event, element in context:
  # Converte o <record> pra string
  record_str = etree.tostring(element, encoding='unicode', pretty_print=True)
  # Adiciona ao chunk atual
  current_records.append(record_str)
  # Limpa o elemento pra economizar memória
  element.clear()

  if records_size <= records_limits:
    # Escreve o chunk
        output_file = f"/workspace/outputs/{output_prefix}{chunk_num:03d}.xml"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.writelines(current_records)
            f.write(footer)
        print(f"Chunk {chunk_num} salvo: {output_file} (com um total de {records_size} registros)")

        chunk_num += 1
        current_records = []
        records_size +=1

# Último chunk se sobrar
if current_records:
    output_file = f"/workspace/outputs/{output_prefix}{chunk_num:03d}.xml"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(current_records)
        f.write(footer)
    print(f"Último chunk salvo: {output_file}")