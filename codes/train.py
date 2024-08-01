
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







'''

if task_dependent == True:
    print("Dataset: {}, band: {}, window: {}s, UnicornPlacement: {}, Action: {}, random: {}, Run:  {}".format(dataset, band, window, UnicornPlacement, Action, task_dependent, rn))
else:
    print("Dataset: {}, band: {}, window: {}s, UnicornPlacement: {}, Action: {}, random: {}, CV_trials: {}, Test_trails: {}, Run: {}".format(dataset, band, window, UnicornPlacement, Action, task_dependent, str(tcv), str(tt), rn))



mapping_MAHNOB_USERid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
                  13: 11, 14: 12, 16: 13, 17: 14, 18: 15, 19: 16, 20: 17, 21: 18, 22: 19,
                  23: 20, 24: 21, 25: 22, 27: 23, 28: 24, 29: 25, 30: 26
                }

def get_train_val_test(df_deleted, samples_mat, num_classes, dataset_ = 'EMMI',task_dependent = True, tcv = [0], tt = [0]):
    
    if dataset == 'EMMI' or dataset == 'DEAP':
        if task_dependent == True:
            index = df_deleted.index.tolist()
            index = np.array(index, dtype = int)
            X = samples_mat[index, :]
    
    
    
            y = np.zeros((len(df_deleted.index),1))
            samples_per_persons = int(len(df_deleted.index)/num_classes)
            for i in range(num_classes):
                y[i * samples_per_persons: (i+1) * samples_per_persons] = df_
    
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size= 0.3,
                random_state= 42,
                stratify=y
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_test,
                y_test,
                test_size= 0.5,
                random_state= 42,
                stratify= y_test
            )
    
    
        
        else:
            val_lables = []
            test_lables = []
    
            for i in tcv:
                val_lables += df_deleted.index[df_deleted['task'] ==  int(i)].tolist()
            df_val = df_deleted[df_deleted.index.isin(val_lables)]
    
    
            for i in tt:
                test_lables += df_deleted.index[df_deleted['task'] ==  int(i)].tolist()
            df_test = df_deleted[df_deleted.index.isin(test_lables)]
    
            df_train = df_deleted[~df_deleted.index.isin(val_lables + test_lables)]
            X_train = samples_mat[ df_train.index.tolist(), :]
            X_val = samples_mat[ df_val.index.tolist(), :]
            X_test = samples_mat[ df_test.index.tolist(), :]
            
    
            y_train = np.zeros((len(df_train.index),1))
            samples_per_persons = int(len(df_train.index)/num_classes)
            for i in range(num_classes):
                y_train[i * samples_per_persons: (i+1) * samples_per_persons] = i
    
            y_val = np.zeros((len(df_val.index),1))
            samples_per_persons = int(len(df_val.index)/num_classes)
            for i in range(num_classes):
                y_val[i * samples_per_persons: (i+1) * samples_per_persons] = i
    
            y_test = np.zeros((len(df_test.index),1))
            samples_per_persons = int(len(df_test.index)/num_classes)
            for i in range(num_classes):
                y_test[i * samples_per_persons: (i+1) * samples_per_persons] = i


    elif dataset == 'MAHNOB':
        if task_dependent == True:
            index = df_deleted.index.tolist()
            index = np.array(index, dtype = int)
            X = samples_mat[index, :]
    
    
    
            y = np.array(df_deleted.label.tolist()).reshape((len(df_deleted.index),1)) 
            
    
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size= 0.3,
                random_state= 42,
                stratify=y
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_test,
                y_test,
                test_size= 0.5,
                random_state= 42,
                stratify= y_test
            )
        
        else:
            val_lables = []
            test_lables = []
            
            for i in tcv:
                val_lables += df_deleted.index[df_deleted['task'] ==  int(i)].tolist()

        
            df_val = df_deleted[df_deleted.index.isin(val_lables)]
            print(df_val)
            
        
            for i in tt:
                test_lables += df_deleted.index[df_deleted['task'] ==  int(i)].tolist()

            
            df_test = df_deleted[df_deleted.index.isin(test_lables)]
            df_train = df_deleted[~df_deleted.index.isin(val_lables + test_lables)]
            
            
            X_train = samples_mat[ df_train.index.tolist(), :]
            X_val = samples_mat[ df_val.index.tolist(), :]
            X_test = samples_mat[ df_test.index.tolist(), :]
            
            
            y_train = np.array(df_train.label.tolist()).reshape((len(df_train.index),1)) 
            y_val = np.array(df_val.label.tolist()).reshape((len(df_val.index),1))
            y_test = np.array(df_test.label.tolist()).reshape((len(df_test.index),1))
            

    else:
        print("Data set is not supported")
        exit()


        
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


#####------------------------------##########
#####---------load data -----------##########

codes_dir = os.getcwd()
print(codes_dir)
prjct_main_dir = os.path.abspath(os.path.join(codes_dir, ".."))
data_dir = os.path.join(prjct_main_dir, "processedData", dataset, 'readyToTrain')
model_save_dir = os.path.join(prjct_main_dir, "models", dataset)

print(prjct_main_dir)
print(data_dir)

if dataset == 'EMMI':
    path = os.path.join(data_dir, band +"__"+ str(int(window*1000)) + "ms_samples")
    with open(path, "rb") as fp:
        samples = pickle.load(fp)
    
    path = os.path.join(data_dir, band +"__"+ str(int(window*1000)) + "ms_info")
    with open(path, "rb") as fp:
        info = pickle.load(fp)

elif dataset == 'DEAP':
    path = os.path.join(data_dir, band +"__"+ str(int(window*1000)) + "ms_samples.npy")
    samples = np.load(path)

    path = os.path.join(data_dir, band +"__"+ str(int(window*1000)) + "ms_info")
    with open(path, "rb") as fp:
        info = pickle.load(fp)

elif dataset == 'MAHNOB':
    path = os.path.join(data_dir, band +"__"+ str(int(window*1000)) + "ms_samples.npy")
    samples = np.load(path)


    path = os.path.join(data_dir, band +"__"+ str(int(window*1000)) + "ms_info")
    with open(path, "rb") as fp:
        info = pickle.load(fp)

else:
    print('Data set is not defined')
    exit()
    
    
df = pd.DataFrame(info, columns =['label', 'task', 'band','state']) 


if dataset == 'EMMI':
    #####------------------------------##########
    #####-----delete crupted users-----##########
    deleted_labels = [88, 89, 100, 104, 106, 92]
    num_classes = int(109 - len(deleted_labels))
    
    unwanted_lables = []
    
    for i in deleted_labels:
        unwanted_lables += df.index[df['label'] ==  int(i)].tolist()
        
        
    
    #####------------------------------##########
    #####----delete unwanted states----##########
    if Action == False:
        unwanted_lables += df.index[df['state'] ==  "T2"].tolist()
        unwanted_lables += df.index[df['state'] ==  "T1"].tolist()
    
    df_deleted = df[~df.index.isin(unwanted_lables)]
    
    #####------------------------------##########
    #####----task base----########## 
    wanted_lables = []
    if TaskNum > 0:
        wanted_lables += df.index[df['task'] ==  TaskNum].tolist()
        df_deleted = df_deleted[df_deleted.index.isin(wanted_lables)]  
    
    print(df_deleted.shape)
    
    #####------------------------------##########
    #####---convert samples matrix ----##########
    samples_mat = np.array(samples, dtype = float)
    samples_mat = np.reshape(samples_mat, (samples_mat.shape[0], samples_mat.shape[2]))
    
    
    
    #####------------------------------##########
    #####--deviding data to 3 group----##########
    X_train, X_val, X_test, label_train, label_val, label_test = get_train_val_test(df_deleted, samples_mat, num_classes,task_dependent, tcv,  tt)

elif dataset == 'DEAP':
    num_classes = len(df.label.unique())
    #trials_ = df.task.unique()
    #print(num_classes)
    #print(trials_)
    #exit()
    X_train, X_val, X_test, label_train, label_val, label_test = get_train_val_test(df, samples, num_classes,task_dependent, tcv,  tt)
    
elif dataset == 'MAHNOB':
    df['X'] = list(samples)

    df_sorted =  df.sort_values(by=['label','task'])
    df_sorted["label"] = df_sorted["label"].replace(mapping_MAHNOB_USERid)
    num_classes = len(df_sorted.label.unique())
    
    trials_ = df_sorted.task.unique()
    print(df_sorted.label.unique())
    print(num_classes)
    print(trials_)
    print(samples.shape)

    X_train, X_val, X_test, label_train, label_val, label_test = get_train_val_test(df_sorted, samples, num_classes, 
                                                                                    dataset,task_dependent, tcv,  tt)

    #exit()

else:
    exit()

#####------------------------------##########
#####--convert label to one hot encoding----##########
y_train = to_categorical((label_train).astype(np.int32),num_classes=num_classes, dtype = int) 
y_test = to_categorical((label_test).astype(np.int32),num_classes=num_classes, dtype = int) 
y_val = to_categorical((label_val).astype(np.int32),num_classes=num_classes, dtype = int)





#####------------------------------##########
#####--normalizing input data using Standard Scaler----##########
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
'''

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
