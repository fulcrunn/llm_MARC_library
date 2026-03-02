import os

def particionar_marcxml(arquivo_entrada, prefixo_saida, tamanho_gb=10):
    # Converte GB para Bytes
    limite_bytes = tamanho_gb * 1024 * 1024 * 1024
    
    cabecalho_linhas = []
    parte_atual = 1
    tamanho_atual = 0
    arquivo_saida = None
    
    print(f"Lendo o arquivo original: {arquivo_entrada}...")
    
    with open(arquivo_entrada, 'r', encoding='utf-8') as f_in:
        # 1. Identificar e guardar o cabeçalho original (antes do primeiro <record>)
        for linha in f_in:
            if '<record>' in linha or '<record ' in linha:
                primeira_linha_record = linha
                break
            cabecalho_linhas.append(linha)
            
        cabecalho = "".join(cabecalho_linhas)
        rodape = "</collection>\n"
        
        # Função auxiliar para criar um novo arquivo
        def abrir_nova_parte(num_parte):
            nome_arquivo = f"{prefixo_saida}_parte{num_parte}.xml"
            f_out = open(nome_arquivo, 'w', encoding='utf-8')
            f_out.write(cabecalho)
            print(f"Criando {nome_arquivo}...")
            return f_out, len(cabecalho.encode('utf-8'))

        # Abre o primeiro arquivo de saída
        arquivo_saida, tamanho_atual = abrir_nova_parte(parte_atual)
        
        # Escreve aquela primeira linha de <record> que usamos para parar o loop do cabeçalho
        arquivo_saida.write(primeira_linha_record)
        tamanho_atual += len(primeira_linha_record.encode('utf-8'))
        
        # 2. Processar o resto do arquivo linha por linha
        registros_processados = 1
        
        for linha in f_in:
            # Ignora o rodapé original, nós adicionaremos o nosso em cada arquivo
            if '</collection>' in linha:
                continue
                
            arquivo_saida.write(linha)
            tamanho_atual += len(linha.encode('utf-8'))
            
            # Se terminou de escrever um registro, checa se atingiu o limite de 10GB
            if '</record>' in linha:
                registros_processados += 1
                
                if registros_processados % 100000 == 0:
                    print(f"Processados {registros_processados} registros. Tamanho da parte {parte_atual}: {tamanho_atual / (1024**3):.2f} GB")
                
                if tamanho_atual >= limite_bytes:
                    # Fecha perfeitamente o arquivo atual
                    arquivo_saida.write(rodape)
                    arquivo_saida.close()
                    print(f"✅ Parte {parte_atual} finalizada com {tamanho_atual / (1024**3):.2f} GB.\n")
                    
                    # Abre o próximo arquivo
                    parte_atual += 1
                    arquivo_saida, tamanho_atual = abrir_nova_parte(parte_atual)
                    
        # 3. Fecha o último arquivo quando terminar a leitura de tudo
        if arquivo_saida:
            arquivo_saida.write(rodape)
            arquivo_saida.close()
            print(f"✅ Parte {parte_atual} finalizada com {tamanho_atual / (1024**3):.2f} GB.")

    print("\n🎉 Particionamento concluído com sucesso!")

# ==========================================
# CONFIGURAÇÃO DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # Substitua pelo nome do seu arquivo gigantesco da Library of Congress
    ARQUIVO_ORIGINAL = "final_output_file.xml" 
    
    # O prefixo dos arquivos que serão gerados
    PREFIXO_DESTINO = "dataset_lc"
    
    # Tamanho desejado em Gigabytes
    TAMANHO_POR_PARTE = 10 
    
    particionar_marcxml(ARQUIVO_ORIGINAL, PREFIXO_DESTINO, TAMANHO_POR_PARTE)