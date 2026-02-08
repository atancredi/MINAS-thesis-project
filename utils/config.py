import os

class TrainingConfig:

    dataset_path: str
    learning_rate: float
    epochs: int
    batch_size: int
    patience: int

    w_amp: float
    w_fd: float
    w_wass: float
    w_sam: float
    
    forward_model_name: str
    
    model_dir: str
    arch_version: str
    model_name: str
    model_path: str
    config_path: str

    def __init__(self, prefix: str, config_path: str, forward_model_name: str = None):
        from dotenv import load_dotenv
        load_dotenv(config_path, override=True)

        self.config_path = config_path
        
        self.dataset_path = os.getenv("DATASET_PATH")
        self.arch_version = os.getenv("ARCH_VERSION")
        self.model_dir = os.getenv("MODEL_DIR")

        self.learning_rate = float(os.getenv("LR"))
        self.epochs = int(os.getenv("EPOCHS"))
        self.batch_size = int(os.getenv("BATCH_SIZE"))
        self.patience = int(os.getenv("PATIENCE"))
    
        self.w_amp = float(os.getenv("W_AMP"))
        self.w_fd = float(os.getenv("W_FD"))
        self.w_wass = float(os.getenv("W_WASS"))
        self.w_sam = float(os.getenv("W_SAM"))

        if forward_model_name != None:
            self.forward_model_name = forward_model_name
        else:
            self.forward_model_name = os.getenv("FORWARD_MODEL_NAME")

        self.model_name = f"{prefix}_{self.arch_version}_{self.epochs}"
        self.model_path = f"{self.model_dir}/{self.model_name}.pth"
    
    def print(self):
        print("dataset:", self.dataset_path)
        print("model_dir:", self.model_dir)
        print("arch version", self.arch_version)
        print("lr:", self.learning_rate, "epochs:", self.epochs, "batch size:", self.batch_size, "patience:", self.patience)
        print("loss weights", self.w_amp, self.w_fd, self.w_wass, self.w_sam)
