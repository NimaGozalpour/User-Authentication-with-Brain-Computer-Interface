
import numpy as np
import os
import pickle
import math

from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf

import argparse
from preprocess import preprocess
from pathlib import Path





# Build  Keras model
def build_model(dropout, penalty_rate, neoron_num, num_classes, X_shape, num_layers = 9):
    layer_list = []

    layer_list += [Dense(neoron_num, input_shape=(X_shape[1],), kernel_regularizer=regularizers.l2(penalty_rate*3)),\
                    BatchNormalization(),\
                    Activation('relu'),\
                    Dropout(dropout),]
    hidden_layer = num_layers - 2
    for i in range(hidden_layer):
        layer_list += [Dense(neoron_num, kernel_regularizer=regularizers.l2(penalty_rate*3)),\
                            BatchNormalization(),\
                            Activation('relu'),\
                            Dropout(dropout),]
    
    layer_list += [Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(penalty_rate*3))]
        
    model = Sequential(layer_list)
    return model

### learning rate schaduler
def IrCal(epoch, T0, IrMax = 0.0005, IrMin = 0):
    i= 1
    Ti = 10
    TotalP = 10
    while epoch > TotalP:
      Ti *= 2
      TotalP += Ti
      i += 1
    TotalP -= Ti
    Tcur = epoch - TotalP
    return IrMin +0.5*(IrMax - IrMin)*(1+math.cos((Tcur/Ti)*math.pi))

def IrCal_how_to_prune(epoch, T0, warmUp = 10, IrMax = 0.0001, IrMin = 0):
    i= 1
    
    if epoch <= warmUp:
       return (1  * epoch + 1) * IrMax * 1/(warmUp + 1)
    Ti = args.maxEpoch
    TotalP = args.maxEpoch
    while epoch > TotalP:
      Ti *= 2
      TotalP += Ti
      i += 1
    TotalP -= Ti
    Tcur = epoch - TotalP - warmUp
    return IrMin +(IrMax - IrMin)*(1+(math.cos(((Tcur/Ti) + 1) * math.pi * 0.5)))


def scheduler(epoch, lr):
    return IrCal_how_to_prune(epoch, 10)


def get_save_dir(args):
    curr_dir = Path(__file__).parent.absolute()
    model_save_dir = os.path.join(curr_dir, '..', "models", args.dataset)
    path = os.path.join(model_save_dir, str(int(args.window*1000))+'ms')
    
    try:
        os.makedirs(path, exist_ok= True)
    except:
        print("already existed")
    
    if args.TaskNum == 0:
        if args.TaskDependent == False:
            path = os.path.join(path, 'random:{}'.format(str(args.TaskDependent)), args.band + '_Run:'\
                                + str(args.RN) + "_UnicornPlacement:" + str(args.UnicornPlacement) + \
                                "_Action:" + str(args.Action)+ "_task_dependent:" + str(args.TaskDependent) +\
                                '_cv_'+ str(args.TCV).replace(' ','') + '_test_' + str(args.TT).replace(' ',''))
        else:
            path = os.path.join(path, 'random:{}'.format(str(args.TaskDependent)), args.band + '_Run:'\
                                + str(args.RN) + "_UnicornPlacement:" + str(args.UnicornPlacement) +\
                                "_Action:" + str(args.Action)+ "_task_dependent:" + str(args.TaskDependent))
    else:
        path = os.path.join(path, args.band + "_" + str(int(args.window*1000))+'ms' + "_UnicornPlacement:"\
                            + str(args.UnicornPlacement) + "_Action:" + str(args.Action)+ "_task_dependent:"\
                            + str(args.TaskDependent)+ "_TaskNum:"+str(args.TaskNum))
    
    try:
        os.makedirs(path, exist_ok= True)
    except:
        print("already existed")

    return path
    
    

def main(args):
    #get preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess.main(args)
    
    
    maxEpoch = args.maxEpoch
    
    dataset = args.dataset
    
    train_batch_size = args.batchSize
    
    verbose_ = int(args.verbose)
    # Create training and validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    
    # prefetch to accelerate training
    train_buffer_size = len(X_train)  # Set to the number of samples in your training dataset
    
    train_dataset = train_dataset.shuffle(buffer_size=train_buffer_size)
    train_dataset = train_dataset.batch(train_batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_batch_size = 256
    
    val_dataset = val_dataset.batch(val_batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    # set Hyperparameters of model
    dropout = 0.1
    penalty_rate = 0.01
    neoron_num = 800

    #get model and set optimizers and callbacks
    model = build_model(dropout, penalty_rate, neoron_num, num_classes, X_train.shape)
    
    adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon= 1e-07, weight_decay=None, amsgrad=False)
    model.compile(optimizer= adam, loss='CategoricalCrossentropy', metrics=['accuracy'])
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),\
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights= True)]
    
    
    
    #print(model.summary())
    # Train the model using the training dataset and validate using the validation dataset
    results = model.fit(train_dataset, epochs= maxEpoch, validation_data=val_dataset, callbacks=[callback], verbose= verbose_)



    #evaluate model via test set 
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    
    
    
    # save model and results of model.fit 
    path =  get_save_dir(args)
    
    Results_path = os.path.join(path, "results")
    with open(Results_path, "wb") as fp:
        pickle.dump(results, fp)
    
    model_path = os.path.join(path, 'model.keras')
    model.save(model_path)


if __name__ == '__main__':
    # Configure CLI parser early. This way we don't need to load TF if there's a missing arg.
    parser = argparse.ArgumentParser(description='different scenario',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Define CLI options.
    parser.add_argument('--band', default= 'allbands', help='Band Name: allbands ,delta, theta, alpha, beta, gamma')
    
    parser.add_argument('--window', default = 4, type = float, help='Window size of sampling')
    
    parser.add_argument('--UnicornPlacement', default = 'False', help='True or False')
    
    parser.add_argument('--Action', default = 'True', help='Including task excuation EMMI dataset, True or Flase')
    
    parser.add_argument('--TaskDependent', default = 'True', help='True or Flase, True means random, otherwise specifiy TCV and TT')
    
    parser.add_argument('--TaskNum', default = 0, type = int, help='(For EMMI dataet)0 means training using all task,\
                        for training on individual task enter number of that task example: 1, 2, 3,...,14')
    
    parser.add_argument('--TCV', type=int, nargs='+', default=[11, 12], help='Tasks for cross validation')
    parser.add_argument('--TT', type=int, nargs='+', default=[13, 14], help='Tasks for test')
    parser.add_argument('--RN', type=int, default = 1, help='Run Number')
    parser.add_argument('--maxEpoch', type=int, default = 220, help='Max Epoch must be integer default is 220')
    parser.add_argument('--dataset', default = 'EMMI', help='EMMI, DEAP, MAHNOB')
    parser.add_argument('--batchSize', default = 256, type = int ,help='EMMI, DEAP,(256 good) MAHNOB(64 good)')
    parser.add_argument('--verbose', default = 0 , type = int , help='0, 1')
    
    args = parser.parse_args()

    main(args)
