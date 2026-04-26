from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_new import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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
print(f"Đang tiến hành đọc dữ liệu từ: {args.data_dir}")

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(data_dir=args.data_dir)
print(f"Đã tải thành công {len(dataset)} mẫu dữ liệu.")

dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=2, 
    strategy="ddp", 
    precision="16-mixed", 
    callbacks=[logger],
    accumulate_grad_batches=4,
    max_epochs=15,
    limit_train_batches=0.5
)


# Train!
trainer.fit(model, dataloader)
trainer.save_checkpoint("./models/controlnet_final_v1.ckpt")
