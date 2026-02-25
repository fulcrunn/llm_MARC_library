#!/bin/bash
set -e

echo "🔄 Atualizando sistema..."
apt update

echo "📦 Instalando dependências do sistema..."
apt install -y git python3 python3-venv ninja-build build-essential

echo "📥 Clonando repositório..."
if [ ! -d "llm_MARC_library" ]; then
    git clone https://github.com/fulcrunn/llm_MARC_library.git
fi

cd llm_MARC_library/

echo "🐍 Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

echo "⬆ Atualizando pip..."
pip install --upgrade pip

# -------------------------------------------------
# 🔥 ORDEM CRÍTICA DE INSTALAÇÃO
# -------------------------------------------------

echo "🔥 Instalando PyTorch 2.2.0 (CUDA 12.1)..."
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

echo "📦 Instalando NumPy compatível (evita crash ABI)..."
pip install numpy==1.26.4

echo "📚 Instalando dependências do projeto..."
pip install -r requirements.txt

echo "⚡ Reinstalando bitsandbytes para garantir GPU..."
pip install --force-reinstall bitsandbytes==0.43.1

echo "⚡ Instalando flash-attn (pode levar ~70 min)..."
MAX_JOBS=4 pip install flash-attn==2.5.7 --no-build-isolation --verbose

# -------------------------------------------------
# 🧪 TESTES
# -------------------------------------------------

echo "🔎 Testando Torch..."
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "🔎 Testando NumPy..."
python -c "import numpy; print('NumPy:', numpy.__version__)"

echo "🔎 Testando bitsandbytes..."
python -c "import bitsandbytes as bnb; print('bitsandbytes OK')"

echo "🔎 Testando flash-attn..."
python -c "import flash_attn; print('flash_attn OK')"

echo "📦 Instalando gdown..."
pip install gdown

echo "⬇ Baixando dataset..."
gdown --fuzzy "https://drive.google.com/file/d/10VCcLPWjJP4fc0B05H0Ki0xMqSSEqMv0/view?usp=sharing"

echo "✅ Ambiente configurado com sucesso!"