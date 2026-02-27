# Código para separar os MARC do arquivo gigante em pequenos arquivos individuias.

from lxml import etree
import pandas as pd
import os
import sys

input_file = "./input/final_output_file"
output_prefix = 'marc_chunk_'
target_size = 5000 # 5 GB
current_size = 0
chunk_num = 1
current_records = []

# Cabeçalho XML básico (ajuste se o seu arquivo tiver namespace específico)
header = '''<?xml version="1.0" encoding="UTF-8"?>
<collection xmlns="http://www.loc.gov/MARC21/slim">
'''
footer = '</collection>'

context = etree.iterparse(input_file, events=('end',), tag='{http://www.loc.gov/MARC21/slim}record')  # ou sem namespace se não tiver

for event, element in context:
  # Converte o <record> pra string
  record_str = etree.tostring(element, encoding='unicode', pretty_print=True)
  # Adiciona ao chunk atual
  current_records.append(record_str)
  # Estima tamanho (aprox, em bytes)
  current_size += len(record_str.encode('utf-8'))
  # Limpa o elemento pra economizar memória
  element.clear()

  if current_size >= target_size / (1024**2):
    # Escreve o chunk
        output_file = f"{output_prefix}{chunk_num:03d}.xml"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.writelines(current_records)
            f.write(footer)
        print(f"Chunk {chunk_num} salvo: {output_file} (~{current_size / (1024**3):.2f} GB)")

        chunk_num += 1
        current_records = []
        current_size = 0

# Último chunk se sobrar
if current_records:
    output_file = f"{output_prefix}{chunk_num:03d}.xml"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(current_records)
        f.write(footer)
    print(f"Último chunk salvo: {output_file}")