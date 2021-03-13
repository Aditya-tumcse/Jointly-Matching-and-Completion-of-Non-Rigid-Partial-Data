import train
import wandb
import configuration

params = dict(epochs=configuration.Config.train_number_epochs,batch_size=configuration.Config.train_batch_size,lr=configuration.Config.learning_rate,dataset_size=configuration.Training_Data_Config.training_data_size,loss_margin=configuration.Config.contrastive_margin,resnet_depth=configuration.Config.resnet_depth)

if __name__ == '__main__':
    wandb.login()
    model = train.wandb_initiliazer(params)
