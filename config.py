class Config():
    training_set_dir = "/cluster/sorona/hyu/ptdcv/ballet/"
    validation_set_dir = "/cluster/sorona/hyu/ptdcv/goalkeeper"
    train_batch_size = 64
    train_number_epochs = 100
    learning_rate = 0.0001
    contrastive_margin = 1.0
    resnet_depth = 18

class Training_Data_Config():
    number_keypoints = 400
    stride = 4
    