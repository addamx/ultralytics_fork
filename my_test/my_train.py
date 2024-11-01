from ultralytics import YOLO



if __name__ == "__main__":
    # 加载模型
    '''
    yolo/model.py(engine/model.py#Model)
    0. __init__
    self.callbacks = callbacks.get_default_callbacks() # 参考 ultralytics/utils/callbacks/base.py default_callbacks, 包含trainer, validator, predictor, exporter的各阶段的hook

    1. _load
    attempt_load_one_weight(yolo11n.pt) # 获取 model(DetectionModel) 和 ckpt
        ckpt = torch_safe_load(yolo11n.pt)
        model = ckpt.model # DetectionModel
    model.args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  超参数
    model.yaml 对应 yolo11n.yaml，记录了backbone、head、nc(80)、scale(n)、ch(3)
    model = YOLO("yolo11n.pt")
    
    
    1.1 DetectionModel __init__
    parse_model(deepcopy(self.yaml)) # 根据模型 yaml文件（eg. cfg/models/11/yolo11.yaml）构建模型
        
    '''
    model = YOLO("yolo11n.pt")

    # 训练模型
    '''
    engine/model.py train
    1. self.trainer = _smart_load("trainer") # DetectionTrainer(BaseTrainer)
    trainer.model = self.trainer.get_model(self.model) #nn/task.py DetectionModel
    DetectionModel.load(self.model) 从序列化模型中提取weights
    
    
    DetectionTrainer(BaseTrainer)
    2. self.trainer.train() # BaseTrainer.train()
    world_size > 0: (多卡)
        generate_ddp_command -> torch.distributed.launch
    else: 单卡
        self._do_train(）
        
    3. self._do_train(）
    self._setup_train(world_size) # 3.1
    
        3.1 self._setup_train
        set_model_attributes
            model.nc    # 类数目
            model.names # 类别名
            model.args # 超参数
        Freeze layers
        amp模式（模型权重和激活使用 16 位精度存储，但某些计算保持 32 位精度，以保持数值稳定性）
            self.amp, self.scaler
        self.stride = 32
        check_train_batch_size
        self.train_loader = self.get_dataloader # 3.1.1
        self.optimizer = self.build_optimizer
        self._setup_scheduler()
    
            3.1.1 get_dataloader
            build_dataset 
                build_yolo_dataset
                    YOLODataset(BaseDataset)
                        __init__
                            cache_images
                            transforms = build_transforms()
            build_dataloader
            
    self.optimizer.zero_grad()
    while True:
        self.model.train()
        
        pbar = enumerate(self.train_loader)
        for i, batch in pbar:
            # Forward
            with autocast(self.amp):
                batch = self.preprocess_batch(batch)
                # !!!
                self.loss, self.loss_items = self.model(batch)
                
            # Backward
            self.scaler.scale(self.loss).backward()
        
            # Optimize 
            self.optimizer_step()
            
        # Validation
        self.metrics, self.fitness = self.validate()
        self.save_model()
    
    self.plot_metrics()
        
    
    
    
        
    '''
    train_results = model.train(
        data="./1848371414130941954.yaml",  # 数据集 YAML 路径
        epochs=2,  # 训练轮次
        imgsz=640,  # 训练图像尺寸
        device="cpu",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
    )

    # 评估模型在验证集上的性能
    metrics = model.val()
