import train
import wandb
import config

params = dict(config.Config.train_number_epochs,config.Config.train_batch_size,config.Config.learning_rate,config.Training_Data_Config.training_data_size,config.Config.contrastive_margin,config.Config.resnet_depth)

if __name__ == '__main__':
    wandb.login()
    model = train.wandb_initiliazer(params)
