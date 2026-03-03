import os
import re
from tqdm import tqdm

# =====================================================
# CONFIGURAÇÃO
# =====================================================
ARQUIVO_SUJO = '/workspace/inputs/lc_data.xml'
ARQUIVO_LIMPO = '/workspace/inputs/lc_data_clean.xml'

filtro_invisiveis = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')

def sanitizar_marcxml(input_file, output_file):
    print(f"Iniciando varredura a laser no arquivo: {input_file}...")
    
    # Pega o tamanho exato do arquivo no disco (instantâneo)
    tamanho_total = os.path.getsize(input_file)
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        primeiro_xml_decl = False
        primeira_collection = False
        
        # Configura o tqdm para medir Bytes (B), formatando automaticamente para KB, MB, GB
        with tqdm(total=tamanho_total, unit='B', unit_scale=True, unit_divisor=1024, desc="Sanitizando", colour="yellow") as pbar:
            
            for linha in fin:
                # Atualiza a barra de progresso com o peso (em bytes) da linha atual
                pbar.update(len(linha.encode('utf-8')))
                
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
                
            # Garante o fechamento perfeito do arquivo ao final do loop
            fout.write('\n</collection>\n')
        
    print(f"\n✅ Sanitização extrema concluída! Arquivo perfeito salvo em: {output_file}")

if __name__ == "__main__":
    sanitizar_marcxml(ARQUIVO_SUJO, ARQUIVO_LIMPO)