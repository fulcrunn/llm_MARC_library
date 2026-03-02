#!/bin/bash
set -e

echo "Install nano"
apt-get update
apt-get install -y nano

echo "Install tmux"
apt-get install -y tmux

echo "✅ Pre install concluded successfully!"

'''
#!/bin/bash
set -e

echo "Instalando unzip"
apt install unzip

echo "Criando pastas"
mkdir inputs
#mkdir inputs_ufpr
#mkdir inputs_lc
mkdir outputs
mkdir lc_dataset
mkdir pdfs
echo "Descompactando arquivos"
unzip master-gdc-gdcdatasets-2020445551_2019-2020445551_2019.zip -d ./lc_dataset
unzip inputs_ufpr.zip -d ./inputs
cd ./lc_dataset/2020445551_2019
apt-get install -y pv
zcat ./Books.All.2019.*.xml.gz | pv > /workspace/inputs/final_output_file.xml
#mv final_output_file.xml /workspace/inputs
#cd /
#cd ./inputs

'''