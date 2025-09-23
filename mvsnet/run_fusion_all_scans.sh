#!/bin/bash

# MVSNet DTU 전체 스캔에 대한 point cloud fusion 스크립트
# 모든 DTU evaluation 스캔 폴더에 대해 depthfusion.py 실행





# 기본 설정
DTU_EVAL_DIR="/data/dtu/dtu_eval"
FUSIBILE_EXE="/workspace/fusibile/build/fusibile"
PROB_THRESHOLD=0.8
DISP_THRESHOLD=0.25
NUM_CONSISTENT=3

# 색상 코드 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}========================================${NC}"
echo -e "${PURPLE}  MVSNet DTU Point Cloud Fusion Script ${NC}"
echo -e "${PURPLE}========================================${NC}"
echo ""

# DTU 평가 디렉터리 확인
if [ ! -d "$DTU_EVAL_DIR" ]; then
    echo -e "${RED}Error: DTU evaluation directory not found: $DTU_EVAL_DIR${NC}"
    exit 1
fi

# fusible 실행 파일 확인
if [ ! -f "$FUSIBILE_EXE" ]; then
    echo -e "${RED}Error: fusibile executable not found: $FUSIBILE_EXE${NC}"
    exit 1
fi

# 스캔 폴더 목록 생성 (scan으로 시작하는 폴더들)
SCAN_FOLDERS=($(find "$DTU_EVAL_DIR" -maxdepth 1 -type d -name "scan*" | sort))

if [ ${#SCAN_FOLDERS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No scan folders found in $DTU_EVAL_DIR${NC}"
    exit 1
fi

echo -e "${CYAN}Found ${#SCAN_FOLDERS[@]} scan folders:${NC}"
for folder in "${SCAN_FOLDERS[@]}"; do
    echo -e "  ${BLUE}$(basename $folder)${NC}"
done
echo ""

# 각 스캔 폴더에 대해 fusion 실행
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=${#SCAN_FOLDERS[@]}

echo -e "${GREEN}Starting point cloud fusion for all scans...${NC}"
echo ""

for scan_folder in "${SCAN_FOLDERS[@]}"; do
    SCAN_NAME=$(basename "$scan_folder")
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Processing: $SCAN_NAME${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # depths_mvsnet 폴더 확인
    DEPTHS_FOLDER="$scan_folder/depths_mvsnet"
    if [ ! -d "$DEPTHS_FOLDER" ]; then
        echo -e "${RED}Warning: depths_mvsnet folder not found in $SCAN_NAME${NC}"
        echo -e "${RED}Skipping $SCAN_NAME...${NC}"
        ((FAIL_COUNT++))
        echo ""
        continue
    fi
    
    # PFM 파일 개수 확인
    PFM_COUNT=$(find "$DEPTHS_FOLDER" -name "*_init.pfm" | wc -l)
    echo -e "${CYAN}Found $PFM_COUNT depth maps in $SCAN_NAME${NC}"
    
    if [ $PFM_COUNT -eq 0 ]; then
        echo -e "${RED}Warning: No depth maps found in $SCAN_NAME${NC}"
        echo -e "${RED}Skipping $SCAN_NAME...${NC}"
        ((FAIL_COUNT++))
        echo ""
        continue
    fi
    
    # 시작 시간 기록
    START_TIME=$(date +%s)
    echo -e "${GREEN}Starting fusion for $SCAN_NAME at $(date)${NC}"
    
    # depthfusion.py 실행
    python depthfusion.py \
        --dense_folder "$scan_folder" \
        --fusibile_exe_path "$FUSIBILE_EXE" \
        --prob_threshold $PROB_THRESHOLD \
        --disp_threshold $DISP_THRESHOLD \
        --num_consistent $NUM_CONSISTENT
    
    # 결과 확인
    FUSION_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $FUSION_EXIT_CODE -eq 0 ]; then
        # PLY 파일 확인
        PLY_FILES=$(find "$scan_folder/points_mvsnet" -name "final3d_model.ply" 2>/dev/null)
        if [ -n "$PLY_FILES" ]; then
            PLY_SIZE=$(du -h $(echo "$PLY_FILES" | head -n1) | cut -f1)
            echo -e "${GREEN}✓ Successfully completed $SCAN_NAME (${DURATION}s, PLY size: $PLY_SIZE)${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${YELLOW}⚠ $SCAN_NAME completed but no PLY file found (${DURATION}s)${NC}"
            ((FAIL_COUNT++))
        fi
    else
        echo -e "${RED}✗ Failed to process $SCAN_NAME (${DURATION}s)${NC}"
        ((FAIL_COUNT++))
    fi
    
    echo ""
done

# 최종 결과 요약
echo -e "${PURPLE}========================================${NC}"
echo -e "${PURPLE}          FUSION SUMMARY               ${NC}"
echo -e "${PURPLE}========================================${NC}"
echo -e "${GREEN}Total scans processed: $TOTAL_COUNT${NC}"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${CYAN}Point cloud files generated:${NC}"
    find "$DTU_EVAL_DIR" -name "final3d_model.ply" -exec ls -lh {} \; | while read line; do
        echo -e "  ${BLUE}$line${NC}"
    done
fi

echo ""
echo -e "${PURPLE}Fusion script completed at $(date)${NC}"

# 종료 코드 설정
if [ $FAIL_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi