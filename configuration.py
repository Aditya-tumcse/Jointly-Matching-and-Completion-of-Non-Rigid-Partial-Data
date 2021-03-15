import torch

class training_configuration():
    #training_set_dir = "/cluster/sorona/hyu/ptdcv/ballet/"
    training_set_dir = "/home/aditya/PycharmProjects/OpenCV-python/Project_2/patch_files/"
    train_batch_size = 64
    train_number_epochs = 100
    learning_rate = 0.0001
    contrastive_margin = 1.0
    resnet_depth = 18
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_workers = 4
    plot_frequency = 100
    
class Training_Data_Config():
    number_keypoints = 400
    stride = 4
    training_data_size = 50000

class Validation_Data_Config():
    number_keypoints = 416
    stride = 4
    validation_data_size = 20000
    validation_set_dir = "/cluster/sorona/hyu/ptdcv/goalkeeper/"
