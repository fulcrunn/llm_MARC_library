# Code to separate a large XML file into smaller chunks of approximately 5 GB each, based on the size of the records. 
# Each chunk will be saved as a separate XML file with a proper header and footer. 
# The script uses the lxml library to parse the XML efficiently and manage memory usage.

from lxml import etree
import pandas as pd
import os
import sys
from pathlib import Path

# Configurations

input_folder = "/workspace/inputs"  # Folder containing the large XML files
#input_file = r"P:\Artigos\Nova pasta\codigo\marc.xml"  # Path to the large XML file
output_prefix = 'marc_chunk_'
target_size = 2 * (1024**3) # 2 GB
current_size = 0
chunk_num = 1
current_records = []


# Create the directory
try:
    out_put_folder_path = Path("/workspace/marc_chunks") # Folder to save the chunked XML files
    print(f"Directory '{out_put_folder_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{out_put_folder_path}' already exists.")    
try:
    out_put_folder_path.mkdir()
    print(f"Directory '{out_put_folder_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{out_put_folder_path}' already exists.")


directory_path = Path(input_folder) # Use the Path class to handle the directory path
# Use a list comprehension to filter only files and get their names
file_names = [file.name for file in directory_path.iterdir() if file.is_file()]

for file_name in file_names:
    input_file = directory_path / file_name

    # Cabeçalho XML básico (ajuste se o seu arquivo tiver namespace específico)
    header = '<?xml version="1.0" encoding="UTF-8"?> \n<collection>\n'
    footer = '</collection>'
    parser = etree.XMLParser(recover=True) # Create an XML parser that can recover from errors
    context = etree.iterparse(input_file, events=('end',), tag='record', parser=parser)  # ou sem namespace se não tiver

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
                output_file = out_put_folder_path / f"{output_prefix}{chunk_num:03d}.xml"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(header)
                    f.writelines(current_records)
                    f.write(footer)
                print(f"Chunk {chunk_num} salvo: {output_file} (~{current_size / (1024**3):.2f} GB)")

                # Reseta os contadores para o próximo chunk
                chunk_num += 1
                current_records = []
                current_size = 0

        # Último chunk se sobrar
        if current_records:
            output_file = out_put_folder_path / f"{output_prefix}{chunk_num:03d}.xml"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.writelines(current_records)
                f.write(footer)
            print(f"Último chunk salvo: {output_file}")