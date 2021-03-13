import train
import wandb
import config

params = dict(epochs=config.Config.train_number_epochs,batch_size=config.Config.train_batch_size,lr=config.Config.learning_rate,dataset_size=config.Training_Data_Config.training_data_size,loss_margin=config.Config.contrastive_margin,resnet_depth=config.Config.resnet_depth)

if __name__ == '__main__':
    wandb.login()
    model = train.wandb_initiliazer(params)
