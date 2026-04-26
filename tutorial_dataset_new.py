import json
import cv2
import numpy as np
import os
import argparse

from torch.utils.data import Dataset

class MyDataset(Dataset):
    # Thêm tham số data_dir vào hàm khởi tạo
    def __init__(self, data_dir):
        self.data = []
        self.data_dir = data_dir # Lưu lại đường dẫn gốc
        
        # Tự động tìm file prompt.json trong thư mục được truyền vào
        json_path = os.path.join(self.data_dir, 'prompt.json')
        
        with open(json_path, 'rt', encoding='utf-8') as f:
                self.data = json.load(f)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['label']
        target_filename = item['palm']
        prompt = item['prompt']

        source_path = os.path.join(self.data_dir, 'label', source_filename)
        target_path = os.path.join(self.data_dir, 'palm', target_filename)

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

if __name__ == '__main__':
    # 1. Khởi tạo đối tượng đọc tham số
    parser = argparse.ArgumentParser(description="Dataset Loader cho ControlNet")
    
    # 2. Định nghĩa tham số --data_dir
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, # Bắt buộc người dùng phải nhập tham số này
        help='Đường dẫn tới thư mục chứa dataset (VD: ./training/fill50k)'
    )
    
    # 3. Phân tích các tham số được nhập từ terminal
    args = parser.parse_args()
    
    # 4. Khởi tạo Dataset với đường dẫn vừa nhận
    print(f"Đang tiến hành đọc dữ liệu từ: {args.data_dir}")
    dataset = MyDataset(data_dir=args.data_dir)
    print(f"Đã tải thành công {len(dataset)} mẫu dữ liệu.")
