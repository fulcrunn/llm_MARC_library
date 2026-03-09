def contar_registros_sujo(caminho_arquivo):
    contador = 0
    # Abrimos como texto puro para evitar o parser de XML
    with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as f:
        for linha in f:
            # Procuramos pela tag de abertura do registro MARCXML
            # O strip() ajuda caso haja espaços, e o lower() previne variação de caixa
            if '<record' in linha.lower():
                contador += 1
    return contador

# Execução
arquivo = 'lc_data.xml'
print("Iniciando contagem de registros no arquivo sujo...")
total = contar_registros_sujo(arquivo)
print(f"Total de registros encontrados: {total}")