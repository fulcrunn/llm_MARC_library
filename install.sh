#!/bin/bash
set -e

echo "🔄 Atualizando sistema e instalando dependências base..."
apt-get update
apt-get install -y git python3-pip python3-dev ninja-build build-essential wget
apt-get install gdown -y
gdown --fuzzy "https://drive.google.com/file/d/1p_N-exEkaDAmFNODdFKKqLER8shw2_nG/view?usp=drive_link"
gdown --fuzzy "https://drive.google.com/file/d/1uFQcfFqaPzXtuYObg719yQ-czRUXhXDM/view?usp=drive_link"

echo "📥 Clonando repositório..."
if [ ! -d "llm_MARC_library" ]; then
    git clone https://github.com/fulcrunn/llm_MARC_library.git
fi

cd llm_MARC_library/

echo "⬆ Atualizando pip..."
pip3 install --upgrade pip

echo "🔥 Instalando PyTorch 2.4.1 (CUDA 12.1)..."
pip3 install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "⚡ Instalando flash-attn..."
pip3 install packaging ninja
MAX_JOBS=4 pip3 install flash-attn==2.6.3 --no-build-isolation --verbose

echo "📚 Instalando dependências do projeto..."
pip3 install -r requirements.txt

echo "⬇ Baixando dataset..."
gdown --fuzzy "https://drive.google.com/file/d/10VCcLPWjJP4fc0B05H0Ki0xMqSSEqMv0/view?usp=sharing"

echo "🔎 Testando dependências críticas..."

echo "🔎 Testando Transformers..."
python3 -c "import transformers; print('Transformers:', transformers.__version__)"

echo "🔎 Testando Torch..."
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "🔎 Testando Triton..."
python -c "import triton; print('Triton:', triton.__version__)"

echo "🔎 Testando bitsandbytes..."
python -c "import bitsandbytes as bnb; print('bitsandbytes OK')"

echo "🔎 Testando flash-attn..."
python -c "import flash_attn; print('flash_attn OK')"

echo "📚 Install tmux"
apt-get install -y tmux

echo "✅ Pod configurado com sucesso!"