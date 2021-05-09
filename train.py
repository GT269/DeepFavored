import re, os, sys, random, json
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import shuffle, LoadTrainData
from network import DeepFavored

def WrapLossFunc(lossFuncName, Y, Y_pred):
    if lossFuncName == 'BinaryCrossentropy':
        loss = tf.keras.losses.binary_crossentropy(Y, Y_pred)
    return loss

def WrapOpt(optName, lr_tensor):
    if optName in ['RMSProp', 'Adam', 'Adadelta', 'AdagradDA', 'Adagrad', 'Ftrl', 'GradientDescent', 'Momentum', 'ProximalAdagrad', 'ProximalGradientDescent', 'SyncReplicas']:
        opt = eval('tf.compat.v1.train.'+optName+'Optimizer')(learning_rate=lr_tensor)
    if optName in ['AdaMax', 'Nadam']:
        opt = eval('tf.contrib.opt.'+optName+'Optimizer')(learning_rate=lr_tensor)
    return opt

def Train(trainData, confArgs, modelDir):
    '''
    '''
    tf.compat.v1.disable_eager_execution()
    
    # Specify params
    model_hyparams, training_params = confArgs['modelHyparamDict'], confArgs['trainParamDict']
    X1_data, Y1_data, X2_data, Y2_data = trainData
    valid_split1 = int(X1_data.shape[0]*0.9)
    valid_split2 = int(X2_data.shape[0]*0.9)
    epochs = training_params['epochs']
    batch_size = training_params['batch_size']

    # Build network
    featureNum, y1_classNum, y2_classNum = X1_data.shape[1], Y1_data.shape[1], Y2_data.shape[1]
    net = DeepFavored(featureNum, model_hyparams, training_params)
    X, H_pred, O_pred = net.build()
    H = tf.compat.v1.placeholder("float", [None, y1_classNum], name="H_true")
    O = tf.compat.v1.placeholder("float", [None, y2_classNum], name="O_true")

    # Define loss
    lossFuncName = training_params['loss_func']
    H_loss = WrapLossFunc(lossFuncName, H, H_pred)
    O_loss = WrapLossFunc(lossFuncName, O, O_pred)

    # Learning rate
    lr_tensor = tf.Variable(training_params['lr'], shape=[], trainable=False)
    new_lr_tensor = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_lr_tensor")
    update_lr = tf.compat.v1.assign(lr_tensor, new_lr_tensor)
    
    # Define optimizer
    optHname = training_params['optimizer_H']
    optOname = training_params['optimizer_O']
    optH = WrapOpt(optHname, lr_tensor)
    optO = WrapOpt(optOname, lr_tensor)
    trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    H_opt = optH.minimize(H_loss, var_list=trainable_vars)
    O_opt = optO.minimize(O_loss, var_list=trainable_vars)

    # Run session and log the info of training
    model_file = modelDir+'/model'
    model_log_file = model_file+'.log'
    saver = tf.compat.v1.train.Saver(defer_build=False)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        best_loss_epoch_tuple = (-1, 0)#initialization
        training_on_best = 0
        new_lr = training_params['lr']
        log_info = ''
        for epoch in range(1,epochs+1):
            # Shuffle and Split data into train set and validation set
            X1_data, Y1_data = shuffle(X1_data, Y1_data)
            X2_data, Y2_data = shuffle(X2_data, Y2_data)
            X1_train, X1_valid = X1_data[:valid_split1], X1_data[valid_split1:]
            Y1_train, Y1_valid = Y1_data[:valid_split1], Y1_data[valid_split1:]
            X2_train, X2_valid = X2_data[:valid_split2], X2_data[valid_split2:]
            Y2_train, Y2_valid = Y2_data[:valid_split2], Y2_data[valid_split2:]
            X1_train_sampleNum, X2_train_sampleNum = X1_train.shape[0], X2_train.shape[0]
            sampleTotalNum = X1_train_sampleNum if X1_train_sampleNum < X2_train_sampleNum else X2_train_sampleNum

            # Training
            info = '==='
            print(info)
            log_info += info+'\n'
            batch_start, batch_end = 0, batch_size
            step = 0
            info_step = int((sampleTotalNum/3)/batch_size)
            while batch_end < sampleTotalNum:
                X1_batch, Y1_batch = X1_train[batch_start:batch_end], Y1_train[batch_start:batch_end]
                X2_batch, Y2_batch = X2_train[batch_start:batch_end], Y2_train[batch_start:batch_end]
                hopt, H_batch_loss = session.run([H_opt, H_loss], feed_dict={ X: X1_batch, H: Y1_batch })
                oopt, O_batch_loss = session.run([O_opt, O_loss], feed_dict={ X: X2_batch, O: Y2_batch })
                
                batch_start = batch_end
                batch_end += batch_size
                step += 1
                if step % info_step == 0:
                    info = 'Epoch %d/%d %d/%d, loss: H-%f O-%f' % (epoch, epochs, step*batch_size, sampleTotalNum, H_batch_loss.mean(), O_batch_loss.mean())
                    print(info)
                    log_info += info+'\n'
           
            H_train_loss = session.run(H_loss, feed_dict={ X: X1_train, H: Y1_train})
            O_train_loss = session.run(O_loss, feed_dict={ X: X2_train, O: Y2_train})
            H_valid_loss = session.run(H_loss, feed_dict={ X: X1_valid, H: Y1_valid})
            O_valid_loss = session.run(O_loss, feed_dict={ X: X2_valid, O: Y2_valid})
            H_train_loss, O_train_loss = H_train_loss.mean(), O_train_loss.mean()
            H_valid_loss, O_valid_loss = H_valid_loss.mean(), O_valid_loss.mean()
            info = 'Epoch %d/%d, train_loss: H-%f  O-%f  valid_loss: H-%f  O-%f'%(epoch, epochs, H_train_loss, O_train_loss, H_valid_loss, O_valid_loss)
            print(info)
            log_info += info+'\n'
            
            valid_loss = H_valid_loss+O_valid_loss#can have different weights
            best_valid_loss, best_epoch = best_loss_epoch_tuple
            if best_valid_loss is -1 or valid_loss < best_valid_loss:
                info = 'Decrease valid_loss from %f to %f, save the newest model.'%(best_valid_loss, valid_loss)
                print(info)
                log_info += info+'\n'
                valid_loss_e04 = round(valid_loss, 4)
                best_loss_epoch_tuple = (valid_loss, epoch)
                best_H_valid_loss, best_O_valid_loss = H_valid_loss, O_valid_loss
                saver.save(session, model_file)
            else:
                info = 'Not decrease valid_loss.'
                print(info)
                log_info += info+'\n'
                training_on_best += 1
                if training_on_best > 3:
                    # continue to train based on the values of trainable variables of the saved best model
                    saver.restore(session, model_file)#all of the values of saved (trained) variables have been restored
                    info = 'Training based on the values of trainable variables of the saved best model.'
                    print(info)
                    log_info += info+'\n'
                    training_on_best = 0

                if epoch - best_epoch > training_params['reduce_lr_epochs']:
                    # Update learning rate
                    old_lr = new_lr
                    new_lr = old_lr * 0.5
                    session.run(update_lr, feed_dict={new_lr_tensor: new_lr})
                    #opt.lr = new_lr
                    info = 'Reduce lr from %f to %f.'%(old_lr, new_lr)
                    print(info)
                    log_info += info+'\n'
                
                if epoch - best_epoch > training_params['early_stop_epochs']:
                    # Stop training
                    info = 'EarlyStopping! Because valid_loss has not been decreased through %d epochs.'%training_params['early_stop_epochs']
                    print(info)
                    log_info += info+'\n'
                    break

    info = 'Training done! Epoch %d has best valid_loss %f, in which H_valid_loss is %f, O_valid_loss is %f.'%(best_loss_epoch_tuple[1],best_loss_epoch_tuple[0],best_H_valid_loss,best_O_valid_loss)
    print(info)
    log_info += info
    
    # Save training log
    with open(model_log_file, 'w') as fw:
        fw.write(log_info)

def TrainWithArgs(args):
    print('===== Start to train DeepFavored =====')
    with open(args.config) as f:
        confArgs = json.load(f)

    # Save all of the hyper-params for the model to be trained
    if not os.path.exists(args.modelDir):
        os.makedirs(args.modelDir)
    confArgs['trainDataParamDict']['trainData'] = os.path.abspath(args.trainData)
    with open(args.modelDir+'/model.json','w') as f:
        f.write(json.dumps(confArgs, sort_keys=True,indent=4))

    # Load train dataset
    X1_train, Y1_train, X2_train, Y2_train = LoadTrainData(args.trainData, confArgs['trainDataParamDict'], labelType = '2dBinaryVector')
    trainData = (X1_train, Y1_train, X2_train, Y2_train)

    # Training
    Train(trainData, confArgs, args.modelDir)
    print('===== Training  Done! =====')