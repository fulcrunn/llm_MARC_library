import os
import re

# =====================================================
# CONFIGURAÇÃO
# =====================================================
ARQUIVO_SUJO = '/workspace/inputs/lc_data.xml'
ARQUIVO_LIMPO = '/workspace/inputs/lc_data_clean.xml'

# Essa Regex mágica captura QUALQUER caractere de controle ASCII invisível
# (do \x00 ao \x08, do \x0B ao \x0C, e do \x0E ao \x1F) 
# que são estritamente proibidos em documentos XML válidos.
filtro_invisiveis = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')

def sanitizar_marcxml(input_file, output_file):
    print(f"Iniciando varredura a laser no arquivo: {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        primeiro_xml_decl = False
        primeira_collection = False
        
        for num_linha, linha in enumerate(fin, 1):
            
            # 1. ELIMINA CARACTERES INVISÍVEIS ILEGAIS LOGO DE CARA
            linha_limpa = filtro_invisiveis.sub('', linha)
            linha_strip = linha_limpa.strip()
            
            # 2. Trata a declaração <?xml ... ?> duplicada
            if linha_strip.startswith('<?xml'):
                if not primeiro_xml_decl:
                    fout.write(linha_limpa)
                    primeiro_xml_decl = True
                continue 
                
            # 3. Trata a tag de abertura <collection ...> duplicada
            if linha_strip.startswith('<collection'):
                if not primeira_collection:
                    fout.write(linha_limpa)
                    primeira_collection = True
                continue 
                
            # 4. Ignora tags de fechamento </collection> perdidas no meio do arquivo
            if linha_strip.startswith('</collection>'):
                continue
                
            # Escreve a linha purificada
            fout.write(linha_limpa)
            
            if num_linha % 5000000 == 0:
                print(f"Já sanitizamos {num_linha} linhas...")
                
        # Garante o fechamento perfeito do arquivo
        fout.write('\n</collection>\n')
        
    print(f"\n✅ Sanitização extrema concluída! Arquivo perfeito salvo em: {output_file}")

if __name__ == "__main__":
    sanitizar_marcxml(ARQUIVO_SUJO, ARQUIVO_LIMPO)