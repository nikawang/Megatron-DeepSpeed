apt-get install wget
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://huggingface.co/gpt2/resolve/main/vocab.json?download=true -O gpt2-vocab.json
wget https://huggingface.co/gpt2/resolve/main/merges.txt?download=true -O gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz

MAX_JOBS=8
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
python setup.py install --cuda_ext --cpp_ext

cd ..
git clone https://github.com/nikawang/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
python ./setup.py install
cp ./megatron/data/helpers{,.so}

python tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix meg-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8

mkdir -p dataset/BookCorpusDataset_text_document
cp meg-gpt2* dataset/BookCorpusDataset_text_document/

bash run-benchmark-model.sh