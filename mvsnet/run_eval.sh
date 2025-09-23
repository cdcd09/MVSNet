#!/bin/bash


'''
chmod +x run_eval.sh
./run_eval.sh
'''
#SCANS=(1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)
SCANS=(4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)

#1개당 6분 가량 걸림 4시 20분 쯤 끝날예정


# 각 스캔 번호에 대해 반복문을 실행합니다.
for SCAN_ID in ${SCANS[@]}
do
  # 현재 어떤 스캔을 처리 중인지 터미널에 출력합니다.
  echo "================================================="
  echo "Processing Scan ID: $SCAN_ID"
  echo "================================================="

  # python test.py 명령어를 실행합니다.
  # --dense_folder 부분의 경로가 각 스캔 ID에 맞게 동적으로 변경됩니다.
  python test.py \
    --dense_folder /data/dtu/dtu_eval/scan$SCAN_ID \
    --pretrained_model_ckpt_path /data/ckpt/model.ckpt \
    --ckpt_step 150000 \
    --view_num 5 \
    --max_d 256 \
    --max_w 1600 \
    --max_h 1152 \
    --regularization 3DCNNs \
    --inverse_depth False \
    --adaptive_scaling True
done

echo "All scans have been processed."
