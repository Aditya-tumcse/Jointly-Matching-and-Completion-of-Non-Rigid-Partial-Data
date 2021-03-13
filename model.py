import train
import wandb
import configuration

params = dict(epochs=configuration.training_configuration.train_number_epochs,batch_size=configuration.training_configuration.train_batch_size,lr=configuration.training_configuration.learning_rate,dataset_size=configuration.Training_Data_Config.training_data_size,loss_margin=configuration.training_configuration.contrastive_margin,resnet_depth=configuration.training_configuration.resnet_depth)

if __name__ == '__main__':
    wandb.login()
    model = train.wandb_initiliazer(params)
