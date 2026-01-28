echo "MODEL_NAME=$1"
echo "TP_SIZE=$2"
echo "PORT=$3"
echo "RANDOM_SEED=$4"
python3 -m sglang.launch_server \
        --model-path $1 \
        --tp-size $2 \
        --tool-call-parser glm45  \
        --mem-fraction-static 0.87 \
        --disable-shared-experts-fusion \
        --host 0.0.0.0 \
        --port $3 \
        --random-seed $4 \
        --context-length 128000 \
        --allow-auto-truncate