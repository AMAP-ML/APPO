bash /home/wangxucong.wxc/AERPO/evaluation/vllm_scripts/vllm_launch_reasoning_model_cuda4-7.sh
bash /home/wangxucong.wxc/AERPO/evaluation/vllm_scripts/vllm_launch_summarize_model_cuda0-3_qwen3_14b.sh

# 上面开启两个心跳，完成之后

bash /home/wangxucong.wxc/AERPO/evaluation/infer_local_sds.sh v1
 

# Optional: upload logs if OSS credentials and destination are supplied.
if [ -n "${OSS_ENDPOINT:-}" ] && [ -n "${OSS_ACCESS_KEY_ID:-}" ] && [ -n "${OSS_ACCESS_KEY_SECRET:-}" ] && [ -n "${OSS_LOG_DEST:-}" ]; then
    ossutil64 cp -r -u -j 120 \
        --endpoint="${OSS_ENDPOINT}" \
        --access-key-id="${OSS_ACCESS_KEY_ID}" \
        --access-key-secret="${OSS_ACCESS_KEY_SECRET}" \
        /home/wangxucong.wxc/log "${OSS_LOG_DEST}"
fi


