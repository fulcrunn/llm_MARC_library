#!/bin/bash
set -e

echo "🔄 Atualizando sistema..."
apt update

echo "📦 Instalando dependências básicas..."
apt install -y git python3 python3-venv ninja-build build-essential

echo "📥 Clonando repositório..."
if [ ! -d "llm_MARC_library" ]; then
    git clone https://github.com/fulcrunn/llm_MARC_library.git
fi

echo "📂 Acessando diretório do projeto..."
cd llm_MARC_library/

echo "🐍 Criando ambiente virtual..."
python3 -m venv venv

echo "⚡ Ativando ambiente virtual..."
source venv/bin/activate

echo "⬆ Atualizando pip..."
pip install --upgrade pip

echo "📚 Instalando requirements..."
pip install -r requirements.txt

echo "⚡ Instalando flash-attn..."
MAX_JOBS=4 pip install flash-attn==2.5.7 --no-build-isolation

echo "🔎 Testando flash-attn..."
python -c "import flash_attn; print('flash_attn OK')"

echo "🔎 Testando CUDA..."
python -c "import torch; print(torch.cuda.is_available())"

echo "📦 Instalando gdown..."
pip install gdown

echo "⬇ Baixando dataset..."
gdown --fuzzy "https://drive.google.com/file/d/10VCcLPWjJP4fc0B05H0Ki0xMqSSEqMv0/view?usp=sharing"

echo "✅ Setup concluído!"