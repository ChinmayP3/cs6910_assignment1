<strong>Dataset</strong>. : FASHION MNIST <br />

<strong>About Dataset</strong> : we have 60000 datapoints as training images and 10000 datapoints as test images with each image is a standardized 28*28 size in gray scale (i.e., 784 pixels) <br/>

<strong>Schema</strong> : We started with loading the FMINST dataset. For preprocessing we have flattened the data and done one hot encoding for the output labels. Then we split the train data into train and valid data in a random fashion and started training the model.

<strong> Optimizers for Back_Propagation used </strong> : stochastic gradient descent, momemtum based gradient descent,nestrov gradient descent,rmsprop, adam, nadam <br/>

<strong> Training and Evaluating the model :</strong> <br/>
<strong>step 1 :</strong> Modify the sweep configuration as required or as you intrested. <br/>
<strong>step 2:</strong> Setup the sweep as follows :
```
sweep_id = wandb.sweep(sweep_config_temp, entity="cs21m027_cs21m011", project="DL_ASG_1_final")
wandb.agent(sweep_id, train)
```
The training will be done automatically for training data and validate the model on validation data and also for the testing data. wandb automatically log plots and charts in the mentioned entity under given project name. wandb will generate train , valid and test accuracy and their losses along with confusion matrix on test data and also sample image of each class along with its label.

<strong>Hyperparameter Tuning </strong> : <br/>

We have split the training data into 90:10 ratio in a random fashion using stratify that is provided by sklearn library as follows :</br>
```
train_x, valid_x, cat_train_y, cat_val_y = train_test_split(flatted_train_images, train_labels, test_size=0.1, stratify = train_labels ,random_state=42)
```

i.e., 54000 data points belong to training data and 6000 data points belong to validation data </br>

Optimizers mentioned above are used to learn parameters for our neural network which are Weights(W) and Biases(b) using back propagation and sweep configuration offered by wandb is used to tune hyperparameters.

The following hyperparameters: <br/>
epochs , hl (number of hidden layers) , hs (size of each hidden layer) , learning_rate , optimizer , batch_size , initialization_strategy , activation_function are tuned using wandb as follows : <br/>

If you want to change any parameter such as number of hidden layers , size of every hidden layer, learning rate , weight decay , epochs, batch size , activation function and weight initilization. change the sweep configuration in below code i.e., add parameters in the below sample configuration for the parameter you require to modify: <br/>
```
sweep_config_temp={
  "parameters": {
        "hidden_layers":{
            "values":[3,4,5]
        },
        "batch size":{
          "values":[32,64]
        }
    }
}
```

After setting up the sweep config and invoking it using the command ```wandb.agent(sweep_id,train)``` ,
train function will be invoked for every combination possible for the hyperparameter choices given in the sweep configuration and corresponding optimizer that matched with any in ```"optimizer":{"values":['nadam']}``` will be called further. <br/>

<strong>Evaluating a model </strong> : <br/>

We are evaluating the model using mainly ```cross_entropy_loss``` but if you want to evaluate the model using squared loss then this choice should be added in the train_function as follows : <br/>
```
def train():
  run=wandb.init()
  config = wandb.config
 
  loss_fun=['squared_loss']
```
If you want to compare the loss between cross entropy and squared loss then write as follows in train() function:
```
def train():
  run=wandb.init()
  config = wandb.config
 
  loss_fun=['cross entropy' , 'squared_loss']
```
<strong> Flexibility of the code </strong> : <br/>

If you want to add new Optimizer for example 'eve' do the following steps : <br/>

<strong> step1</strong> : <br/> Write the eve optimizer code in a function and name it as eve and then Add eve as optimizer value in the sweep configuration as follows : <br/>
```
"optimizer":{ "values" : ['eve'] }
```
<strong>step2</strong> : <br/> Add the following two lines of code in train function : <br/>
```
if config.optimizer == 'eve':
    W , b = eve(train_x,train_y,valid_x,valid_y,d,hl,ol,config.act_fun,loss_fun,config.epochs,config.eta,config.strat,alpha,config.batch_size)
```
