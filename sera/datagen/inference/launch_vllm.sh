echo "MODEL_NAME=$1"
echo "TP_SIZE=$2"
echo "PORT=$3"
echo "RANDOM_SEED=$4"
vllm serve $1 --port $3 \
            --tensor-parallel-size $2 \
            --max-model-len 32768 \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --enforce-eager \
            --seed $4 \
            --disable-cascade-attn