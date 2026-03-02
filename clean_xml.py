import os

# =====================================================
# CONFIGURAÇÃO
# =====================================================
# Substitua pelos caminhos corretos do seu RunPod
ARQUIVO_SUJO = '/workspace/inputs/lc_data.xml'
ARQUIVO_LIMPO = '/workspace/inputs/lc_data_clean.xml'

def limpar_marcxml(input_file, output_file):
    print(f"Limpando o arquivo: {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        primeiro_xml_decl = False
        primeira_collection = False
        
        for num_linha, linha in enumerate(fin, 1):
            linha_limpa = linha.strip()
            
            # 1. Trata a declaração <?xml ... ?>
            if linha_limpa.startswith('<?xml'):
                if not primeiro_xml_decl:
                    fout.write(linha)
                    primeiro_xml_decl = True
                continue # Se já escreveu a primeira, ignora as próximas
                
            # 2. Trata a tag de abertura <collection ...>
            if linha_limpa.startswith('<collection'):
                if not primeira_collection:
                    fout.write(linha)
                    primeira_collection = True
                continue # Se já escreveu a primeira, ignora as repetidas no meio
                
            # 3. Ignora tags de fechamento </collection> no meio do arquivo
            if linha_limpa.startswith('</collection>'):
                continue
                
            # Escreve os registros reais (<record>, <datafield>, etc)
            fout.write(linha)
            
            if num_linha % 5000000 == 0:
                print(f"Já passamos pela linha {num_linha}...")
                
        # Garante que o arquivo inteiro termina com APENAS UM fechamento
        fout.write('\n</collection>\n')
        
    print(f"\n✅ Limpeza concluída! Arquivo salvo em: {output_file}")
    print("Você já pode apagar o arquivo sujo para liberar espaço.")

if __name__ == "__main__":
    limpar_marcxml(ARQUIVO_SUJO, ARQUIVO_LIMPO)