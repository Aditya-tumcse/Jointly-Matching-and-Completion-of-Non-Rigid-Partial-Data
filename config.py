import torch

class Config():
    training_set_dir = "/home/aditya/PycharmProjects/OpenCV-python/Project_2/patch_files/"
    validation_set_dir = "/cluster/sorona/hyu/ptdcv/goalkeeper"
    train_batch_size = 64
    train_number_epochs = 100
    learning_rate = 0.0001
    contrastive_margin = 1.0
    resnet_depth = 18
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_workers = 4
    plot_frequency = 300
    
class Training_Data_Config():
    number_keypoints = 400
    stride = 4
    training_data_size = 2