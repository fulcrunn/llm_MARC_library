#!/bin/bash
set -e

echo "🔄 Atualizando sistema e instalando dependências base..."
apt-get update
apt-get install -y git python3-pip python3-dev ninja-build build-essential wget

echo "📥 Clonando repositório..."
if [ ! -d "llm_MARC_library" ]; then
    git clone https://github.com/fulcrunn/llm_MARC_library.git
fi

# Entrando na pasta do projeto recém-clonado
cd llm_MARC_library/

echo "⬆ Atualizando pip..."
pip3 install --upgrade pip

echo "🔥 Instalando PyTorch 2.2.0 (CUDA 12.1)..."
pip3 install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "⚡ Instalando flash-attn..."
pip3 install packaging ninja
MAX_JOBS=4 pip3 install flash-attn==2.5.7 --no-build-isolation --verbose

echo "📚 Instalando dependências do projeto..."
# Agora ele vai encontrar o requirements.txt corretamente
pip3 install -r requirements.txt

echo "🔎 Testando dependências críticas..."
python3 -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import triton; print('Triton:', triton.__version__)"
python3 -c "import bitsandbytes as bnb; print('bitsandbytes OK')"
python3 -c "import flash_attn; print('flash_attn OK')"

echo "⬇ Baixando dataset..."
gdown --fuzzy "https://drive.google.com/file/d/10VCcLPWjJP4fc0B05H0Ki0xMqSSEqMv0/view?usp=sharing"

echo "✅ Pod configurado com sucesso!"