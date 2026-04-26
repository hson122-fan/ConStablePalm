import json
import os

OUTPUT_FILE = "prompt.json"
PROMPT_TEXT = "a realistic high-quality palmprint"
TOTAL_SAMPLES = 10
def generate_large_jsonl():

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Vòng lặp chạy từ 1 đến 12000
        for i in range(1, TOTAL_SAMPLES + 1):
            

            entry = {
                "label": str(i) + ".png",
                "palm": str(i) + ".png",
                "prompt": PROMPT_TEXT
            }
            
            f.write(json.dumps(entry) + ',\n')

    print(f"Đã tạo thành công file: {OUTPUT_FILE} tại thư mục {os.getcwd()}")
    print(f"Tổng số dòng dữ liệu: {TOTAL_SAMPLES}")

generate_large_jsonl()