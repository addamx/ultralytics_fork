from ultralytics import YOLO



if __name__ == "__main__":
    # 加载模型
    '''
    yolo/model.py(engine/model.py#Model)
    0. __init__
    self.callbacks = callbacks.get_default_callbacks() # 参考 ultralytics/utils/callbacks/base.py#default_callbacks, 包含trainer, validator, predictor, exporter的各阶段的hook

    1. _load
    > attempt_load_one_weight(yolo11n.pt)
    >> ckpt = torch_safe_load(yolo11n.pt)
    >> model = ckpt.model # DetectionModel 序列化的nn.Module
    >> 超参数 model.args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))} 
    
    model.yaml 对应 yolo11n.yaml，记录了backbone、head、nc(80)、scale(n)、ch(3)
    '''
    model = YOLO("yolo11n.pt")

    # 训练模型
    '''
    engine/model.py
    1. train
    > self.trainer = _smart_load("trainer")
    > yolo.detect.DetectionTrainer(engine/trainer.py#BaseTrainer) __init__
    >> trainer.model = get_model(self.model)
    '''
    train_results = model.train(
        data="./1848371414130941954.yaml",  # 数据集 YAML 路径
        epochs=2,  # 训练轮次
        imgsz=640,  # 训练图像尺寸
        device="cpu",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
    )

    # 评估模型在验证集上的性能
    metrics = model.val()
