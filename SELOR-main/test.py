import torch

# 指定第 7 张 GPU（编号从 0 开始）




# 打印当前设备信息
print("Using device:", device)
print("Device name:", torch.cuda.get_device_name(device))
# 查看当前设备的显存使用情况
allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB

print(f"GPU {device.index} - Allocated Memory: {allocated:.2f} MB")
print(f"GPU {device.index} - Reserved Memory: {reserved:.2f} MB")
