import argparse

from . import MAHNOB_preprocess
from . import EMMI_preprocess
from . import DEAP_preprocess

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical             
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def get_train_val_test(df_deleted, samples_mat, num_classes, dataset, task_dependent = True, tcv = [0], tt = [0]):
    if dataset == 'EMMI' or dataset == 'DEAP':
        if task_dependent == True:
            index = df_deleted.index.tolist()
            index = np.array(index, dtype = int)
            X = samples_mat[index, :]
    
    
    
            y = np.zeros((len(df_deleted.index),1))
            samples_per_persons = int(len(df_deleted.index)/num_classes)
            for i in range(num_classes):
                y[i * samples_per_persons: (i+1) * samples_per_persons] = i
    
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
    
    
            

    return X_train, X_val, X_test, y_train, y_val, y_test



def main(args):
    
    
    band = args.band
    window = float(args.window)
    TaskNum = int(args.TaskNum)
    
    
    UnicornPlacement = False
    if args.UnicornPlacement == 'True':
        UnicornPlacement = True
        
    
    Action = False
    if args.Action == 'True':
        Action = True
        
    task_dependent = True
    if args.TaskDependent == 'False':
        task_dependent = False
    
    
    tcv = args.TCV
    tt = args.TT
    rn = args.RN
    
    
    dataset = args.dataset
    
    
    
    
    
    if task_dependent == True:
        print("Dataset: {}, band: {}, Sampling window: {}s, UnicornPlacement: {}, Action included: {},\
            random: {}, Run:  {}".format(dataset, band, window, UnicornPlacement, Action, task_dependent, rn))
    else:
        print("Dataset: {}, band: {}, Sampling window: {}s, UnicornPlacement: {}, Action included: {},\
            random: {}, CV_trials: {}, Test_trails: {}, Run: {}".format(dataset, band,\
                                                                        window, UnicornPlacement,\
                                                                        Action, task_dependent,\
                                                                        str(tcv), str(tt), rn))
    
    
    
    mapping_MAHNOB_USERid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,\
                             9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 16: 13,\
                             17: 14, 18: 15, 19: 16, 20: 17, 21: 18, 22: 19,\
                             23: 20, 24: 21, 25: 22, 27: 23, 28: 24, 29: 25, 30: 26,
                            }
       
    
    
    #####------------------------------##########
    #####---------load and preprocess data -----------##########
    
    if dataset == 'EMMI':
        samples, info = EMMI_preprocess.main(args)
        df = pd.DataFrame(info, columns =['label', 'task', 'band','state']) 

        #####------------------------------##########
        #####-----delete corrupted users-----##########
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
        X_train, X_val, X_test, label_train, label_val, label_test = get_train_val_test(df_deleted, samples_mat, num_classes,\
                                                                                        dataset,task_dependent, tcv,  tt)
    
    elif dataset == 'DEAP':
        samples, info = DEAP_preprocess.main(args)
        df = pd.DataFrame(info, columns =['label', 'task', 'band','state']) 
        

        samples_mat = np.array(samples, dtype = float)
        samples_mat = np.reshape(samples_mat, (samples_mat.shape[0], samples_mat.shape[2]))
        num_classes = len(df.label.unique())
        #trials_ = df.task.unique()
        #print(num_classes)
        #print(trials_)
        #exit()
        X_train, X_val, X_test, label_train, label_val, label_test = get_train_val_test(df, samples_mat, num_classes,\
                                                                                        dataset, task_dependent, tcv,  tt)
        
        
    
    elif dataset == 'MAHNOB':
        samples, info = MAHNOB_preprocess.main(args)
        df = pd.DataFrame(info, columns =['label', 'task', 'band','state']) 

        df['X'] = list(samples)
    
        df_sorted =  df.sort_values(by=['label','task'])
        df_sorted["label"] = df_sorted["label"].replace(mapping_MAHNOB_USERid)
        num_classes = len(df_sorted.label.unique())

        samples_mat = np.array(samples, dtype = float)
        samples_mat = np.reshape(samples_mat, (samples_mat.shape[0], samples_mat.shape[2]))
        
        trials_ = df_sorted.task.unique()
        print(df_sorted.label.unique())
        print(num_classes)
        print(trials_)
        print(samples.shape)
    
        X_train, X_val, X_test, label_train, label_val, label_test = get_train_val_test(df_sorted, samples_mat, num_classes, \
                                                                                        dataset,task_dependent, tcv,  tt)
        
    
    else:
        print('Data set is not defined')
        exit()
        
        
    #####------------------------------##########
    #####--convert label to one hot encoding----##########
    y_train = to_categorical((label_train).astype(np.int32),num_classes=num_classes) 
    y_test = to_categorical((label_test).astype(np.int32),num_classes=num_classes) 
    y_val = to_categorical((label_val).astype(np.int32),num_classes=num_classes)
    
    
    
    
    
    #####------------------------------##########
    #####--normalizing input data using Standard Scaler----##########
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes

if __name__ == "__main__":
       # Configure CLI parser early. This way we don't need to load TF if there's a missing arg.
    parser = argparse.ArgumentParser(description='different scenario',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Define CLI options.
    parser.add_argument('--band', default= 'allbands', help='Band Name: allbands ,delta, theta, alpha, beta, gamma')
    
    parser.add_argument('--window', default = 4, type = float, help='Window size of sampling')
    
    parser.add_argument('--UnicornPlacement', default = 'False', help='True or False')
    
    parser.add_argument('--Action', default = 'True', help='True or Flase')
    
    parser.add_argument('--TaskDependent', default = 'True', help='True or Flase, True means random, otherwise specifiy TCV and TT')
    
    parser.add_argument('--TaskNum', default = 0, type = int, help='0 means all task, for individual task enter number of that task example: 1, 2, 3,...,14')
    
    parser.add_argument('--TCV', type=int, nargs='+', default=[11, 12], help='Tasks for cross validation')
    parser.add_argument('--TT', type=int, nargs='+', default=[13, 14], help='Tasks for test')
    parser.add_argument('--RN', type=int, default = 1, help='Run Number')
    parser.add_argument('--dataset', default = 'EMMI', help='EMMI, DEAP, MAHNOB')
    parser.add_argument('--batchSize', default = 256, type = int ,help='EMMI, DEAP,(256 good) MAHNOB(64 good)')
    parser.add_argument('--verbose', default = 0 , type = int , help='0, 1')


    args = parser.parse_args()



    main(args)
    