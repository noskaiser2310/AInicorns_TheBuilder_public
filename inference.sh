#!/bin/bash
# VNPT AI Hackathon - Inference Pipeline
# Entry-point theo yêu cầu BTC:
# - Đọc /code/private_test.json (được BTC mount vào)
# - Xuất submission.csv

echo "============================================================"
echo "VNPT AI Age of AInicorns - Track 2 The Builder"
echo "Team: Just2Try"
echo "============================================================"

# Run main prediction
python predict.py --input /code/private_test.json --output /code/submission.csv

echo "============================================================"
echo "Inference complete!"
echo "Output: /code/submission.csv"
echo "============================================================"
