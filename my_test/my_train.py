from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 训练模型
train_results = model.train(
    data="./1848371414130941954.yaml",  # 数据集 YAML 路径
    epochs=2,  # 训练轮次
    imgsz=640,  # 训练图像尺寸
    device="cpu",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
)

# 评估模型在验证集上的性能
metrics = model.val()

