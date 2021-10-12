## load library
from __future__ import print_function, division
import scipy.io as sio
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import models, layers, initializers
from keras.layers import Input, Conv1D,  Dense,  MaxPooling1D,  Flatten, Activation,  Dropout, GaussianNoise, Reshape, BatchNormalization, Convolution2D,  MaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,  TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from keras.utils import np_utils

from scipy import stats
from scipy.signal import spectrogram
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def main(model_num, model_type):
    random_seed = 1

    #load the data ::
    Data= sio.loadmat('KSSL_new2.mat')
    X_train_MIR = Data['MIRc'];X_test_MIR=Data['MIRv']
    X_train_NIR = Data['NIRc'];X_test_NIR=Data['NIRv']

    X_train_MIRr = Data['MIRrc'];X_test_MIRr=Data['MIRrv']
    y_train = Data['Yc'];y_test = Data['Yv']

    # plot X.NIR
    lamda_NIR1=np.arange(350, 2501,1)
    # _=plt.plot(lamda_NIR1,NIR.T)

    #plot X.MIR
    lamda_MIR1=np.arange(7498.043,599.766,-(7498.043-599.766)/3578)
    # _=plt.plot(lamda_MIR1,MIR.T)

    #Split the dataset :
    X_cal_NIR, X_val_NIR, y_calX, y_valX = train_test_split(X_train_NIR, y_train, train_size=0.75,random_state=random_seed)
    X_cal_MIR, X_val_MIR= train_test_split(X_train_MIR, train_size=0.75,random_state=random_seed)

    #transform the Y
    sc_y = StandardScaler()
    y_cal = sc_y.fit_transform(y_calX)
    y_val = sc_y.transform(y_valX)
    
    #transform the X
    def filter_SNV(spectra):
      """ SNV spectra transformation  """
      M, N = spectra.shape
      treated_spec = np.zeros([M, N])
      for i in range(0, M):
          temp = spectra[i,:]
          treated_spec[i,:]=(temp - np.mean(temp))/np.std(temp)
      return treated_spec

    # set up model parameters
    EPOCH = 500
    B_size = 50
    this = []
    f_s =20

    #standardize the data
    def scaleme(cal_data,new_data):
        data_norm = (new_data - np.mean(cal_data))/np.std(cal_data)
        data_norm = np.expand_dims(data_norm, axis = 3)
        return data_norm

    #create a spectrogram
    def spectrogram_std(data, nperseg=100, noverlap=50, log_spectrogram = True):
        fs = 0.25
        X=np.empty((len(data),51,42),dtype=float)
        for i in range (len(data)):
            f, t, Sxx = spectrogram(data[i,], fs=fs,window=('hann'), nperseg=nperseg, noverlap=noverlap)
            if log_spectrogram: # log transform the spectrogram
                Sxx = abs(Sxx) # Make sure, all values are positive before taking log
                mask = Sxx > 0 # We dont want to take the log of zero
                Sxx[mask] = np.log(Sxx[mask])
            X[i,] = Sxx
        return X

    #create a one-dimension model
    def model1D(channel_number,nb_features,multOUT):
        if channel_number==1:  #using only one spectra input, i.e. MIR or NIR
            inputShape=(nb_features,1)
        elif channel_number ==2:  #using two spectra input; i.e. MIR&NIR
            inputShape=(nb_features,2)

        inputs= Input(shape=inputShape)

        #add the layers (can be modified)
        x = GaussianNoise(0.2)(inputs)
        x = Conv1D(32, f_s, padding="same",name="Conv1", kernel_initializer=initializers.he_normal(seed=0))(inputs)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling1D() (x)

        x = Conv1D(64, f_s, padding="same",name="Conv2")(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling1D(5) (x)

        x = Conv1D(128, f_s, padding="same",name="Conv3")(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling1D(5) (x)

        x = Conv1D(256, f_s, padding="same",name="Conv4")(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling1D(5) (x)

        x = Dropout (0.4) (x)
        x = Flatten () (x)

        #for multiple output predictions
        if (multOUT==True):
            do_rate=0.2

            y1 = Dense (100, activation='relu',name='Dense_TC.1') (x)
            y1 = BatchNormalization() (y1)
            y1 = Dropout (do_rate)(y1)
            y1 = Dense (1, name='Dense_TC.2') (y1)

            y2 = Dense (100, activation='relu',name='Dense_OC.1') (x)
            y2 = BatchNormalization() (y2)
            y2 = Dropout (do_rate)(y2)
            y2 = Dense (1, name='Dense_OC.2') (y2)

            y3 = Dense (100, activation='relu',name='Dense_CEC.1') (x)
            y3 = BatchNormalization() (y3)
            y3 = Dropout (do_rate)(y3)
            y3 = Dense (1, name='Dense_CEC.2') (y3)

            y4 = Dense (100, activation='relu',name='Dense_clay.1') (x)
            y4 = BatchNormalization() (y4)
            y4 = Dropout (do_rate)(y4)
            y4 = Dense (1, name='Dense_clay.2') (y4)

            y5 = Dense (100, activation='relu',name='Dense_sand.1') (x)
            y5 = BatchNormalization() (y5)
            y5 = Dropout (do_rate)(y5)
            y5 = Dense (1, name='Dense_sand.2') (y5)

            y6 = Dense (100, activation='relu',name='Dense_pH.1') (x)
            y6 = BatchNormalization() (y6)
            y6 = Dropout (do_rate)(y6)
            y6 = Dense (1, name='Dense_pH.2') (y6)

            model = Model(inputs=inputs,outputs=[y1,y2,y3,y4,y5,y6])

        #for single output predictions
        elif (multOUT == False):
            x = Dense (100, activation='relu', name='Dense_all.1') (x)
            x = BatchNormalization() (x)
            x = Dropout (0.2)(x)
            x = Dense (1, name='Dense_all.2') (x)

            model = Model (inputs=inputs,outputs=x)

        #compile the model
        model.compile(loss='mse', optimizer=Adam(lr=0.001),metrics=['mse'])
        return model

    # create a two-dimensional model (( using spectrogram ))
    def model2D(nb_features,multOUT=True):

        beg=32
        inputs= Input(shape=nb_features)
        do_rate=0.4

        #add the layers (can be modified)
        x = Convolution2D(beg*2**0, (3, 3),padding='same',name='conv0', kernel_initializer=initializers.he_normal(seed=0)) (inputs)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling2D(pool_size=(2,2)) (x)

        x = Convolution2D(beg*2**1, (3, 3),padding='same',name='conv2')(x)
        x = Convolution2D(beg*2**1, (3, 3),padding='same',name='conv3')(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling2D(pool_size=(2,2)) (x)

        x = Convolution2D(beg*2**2, (3, 3),padding='same',name='conv4')(x)
        x = Convolution2D(beg*2**2, (3, 3),padding='same',name='conv1')(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling2D(pool_size=(2,2)) (x)

        x = Dropout (do_rate) (x)
        x = Flatten () (x)

        #for multiple output predictions
        if (multOUT==True):
            do_rate=0.2

            y1 = Dense (100, activation='relu',name='Dense_TC.1') (x)
            y1 = BatchNormalization() (y1)
            y1 = Dropout (do_rate)(y1)
            y1 = Dense (1, name='Dense_TC.2') (y1)

            y2 = Dense (100, activation='relu',name='Dense_OC.1') (x)
            y2 = BatchNormalization() (y2)
            y2 = Dropout (do_rate)(y2)
            y2 = Dense (1, name='Dense_OC.2') (y2)

            y3 = Dense (100, activation='relu',name='Dense_CEC.1') (x)
            y3 = BatchNormalization() (y3)
            y3 = Dropout (do_rate)(y3)
            y3 = Dense (1, name='Dense_CEC.2') (y3)

            y4 = Dense (100, activation='relu',name='Dense_clay.1') (x)
            y4 = BatchNormalization() (y4)
            y4 = Dropout (do_rate)(y4)
            y4 = Dense (1, name='Dense_clay.2') (y4)

            y5 = Dense (100, activation='relu',name='Dense_sand.1') (x)
            y5 = BatchNormalization() (y5)
            y5 = Dropout (do_rate)(y5)
            y5 = Dense (1, name='Dense_sand.2') (y5)

            y6 = Dense (100, activation='relu',name='Dense_pH.1') (x)
            y6 = BatchNormalization() (y6)
            y6 = Dropout (do_rate)(y6)
            y6 = Dense (1, name='Dense_pH.2') (y6)

            model = Model(inputs=inputs,outputs=[y1,y2,y3,y4,y5,y6])

        #for single output predictions
        elif (multOUT == False):
            x = Dense (100, activation='relu', name='Dense_all.1') (x)
            x = BatchNormalization() (x)
            x = Dropout (0.2)(x)
            x = Dense (1, name='Dense_all.2') (x)
            model = Model (inputs=inputs,outputs=x)

        #compile the model
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mse'])
        return model

    #create matrix for the outer product analysis (OPA)
    def OPA_res(spec1,spec2):
        xx=5
        CC=np.empty((len(spec1),round(spec1.shape[1]/xx),int(np.ceil(spec2.shape[1]/xx))),dtype=float)
        for i in range (len(spec1)):
            A=np.outer(spec1[i,],spec2[i,])
            dc=A[:,range(0,A.shape[1],xx)]
            drdc=dc[range(0,A.shape[0],xx),:]
            CC[i,]=drdc
        CC=np.expand_dims(CC,axis=3)
        return CC

    #create model for the OPA matrix
    def model2D_OPA(nb_features,multOUT=True):
        beg=32
        inputs= Input(shape=nb_features)
        do_rate=0.4

        x = Convolution2D(beg*2**0, (5, 5), strides=(2,2), padding='same',name='conv0', kernel_initializer=initializers.he_normal(seed=0)) (inputs)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling2D(pool_size=(3,3)) (x)

        x = Convolution2D(beg*2**1, (5, 5), strides=(2,2), padding='same',name='conv2')(x)
        x = Convolution2D(beg*2**1, (5, 5), strides=(2,2), padding='same',name='conv3')(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling2D(pool_size=(3,3)) (x)

        x = Convolution2D(beg*2**2, (5, 5), strides=(2,2), padding='same',name='conv4')(x)
        x = Convolution2D(beg*2**2, (5, 5), strides=(2,2), padding='same',name='conv1')(x)
        x = BatchNormalization() (x)
        x = Activation ('relu') (x)
        x = MaxPooling2D(pool_size=(2,2)) (x)

        x = Dropout (do_rate) (x)
        x = Flatten () (x)

        #for multiple output predictions
        if (multOUT==True):
            do_rate=0.2

            y1 = Dense (100, activation='relu',name='Dense_TC.1') (x)
            y1 = BatchNormalization() (y1)
            y1 = Dropout (do_rate)(y1)
            y1 = Dense (1, name='Dense_TC.2') (y1)

            y2 = Dense (100, activation='relu',name='Dense_OC.1') (x)
            y2 = BatchNormalization() (y2)
            y2 = Dropout (do_rate)(y2)
            y2 = Dense (1, name='Dense_OC.2') (y2)

            y3 = Dense (100, activation='relu',name='Dense_CEC.1') (x)
            y3 = BatchNormalization() (y3)
            y3 = Dropout (do_rate)(y3)
            y3 = Dense (1, name='Dense_CEC.2') (y3)

            y4 = Dense (100, activation='relu',name='Dense_clay.1') (x)
            y4 = BatchNormalization() (y4)
            y4 = Dropout (do_rate)(y4)
            y4 = Dense (1, name='Dense_clay.2') (y4)

            y5 = Dense (100, activation='relu',name='Dense_sand.1') (x)
            y5 = BatchNormalization() (y5)
            y5 = Dropout (do_rate)(y5)
            y5 = Dense (1, name='Dense_sand.2') (y5)

            y6 = Dense (100, activation='relu',name='Dense_pH.1') (x)
            y6 = BatchNormalization() (y6)
            y6 = Dropout (do_rate)(y6)
            y6 = Dense (1, name='Dense_pH.2') (y6)

            model = Model(inputs=inputs,outputs=[y1,y2,y3,y4,y5,y6])

        #for single output prediction
        elif (multOUT == False):
            x = Dense (100, activation='relu', name='Dense_all.1') (x)
            x = BatchNormalization() (x)
            x = Dropout (0.2)(x)
            x = Dense (1, name='Dense_all.2') (x)
            model = Model (inputs=inputs,outputs=x)

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mse'])
        return model

    ############################################
	## set up the various spectra model input ##
    ############################################

	#model 1: MIR 1D MultipleY
    print('Running model: {} ({})'.format(model_num, model_type))
    if model_num== 1:
        m='MIR_1D'
	
        # get the standardized spectra
        cal_spec = X_cal_MIR; val_spec = X_val_MIR; test_spec = X_test_MIR

        # reshape the spectra to match input for the model
        cal_spec = cal_spec.reshape(cal_spec.shape[0],cal_spec.shape[1],1)
        val_spec = np.expand_dims(val_spec,axis=3)
        test_spec = np.expand_dims(test_spec,axis=3)
        nb_features=cal_spec.shape[1]; channel_number=1;

	#model 2: NIR 1D MultipleY
    elif model_num==2:
        m='NIR_1D'
         
	# get the standardized spectra
        cal_spec = X_cal_NIR; val_spec = X_val_NIR; test_spec = X_test_NIR

        # reshape the spectra to match input for the model
        cal_spec=cal_spec.reshape(cal_spec.shape[0],cal_spec.shape[1],1)
        val_spec=np.expand_dims(val_spec,axis=3)
        test_spec=np.expand_dims(test_spec,axis=3)
        nb_features=cal_spec.shape[1]; channel_number=1;

    #model 3: combined NIR MIR 2 channels
    elif model_num==3:
        m='NIRMIRres_1D';
        nb_features=X_train_MIRr.shape[1]; channel_number=2

        # use the resampled MIR data (ensuring both NIR and MIR are same lengths when you are loading it as two channels)
        X_cal_MIR, X_val_MIR= train_test_split(X_train_MIRr, train_size=0.75,random_state=random_seed)
        X_test_MIR = X_test_MIRr
	
	# get the standardized spectra
        cal_spec = np.zeros((len(X_cal_NIR), nb_features, 2))
	cal_spec[:, :, 0] = X_cal_NIR
	cal_spec[:, :, 1] = X_cal_MIR

	val_spec = np.zeros((len(X_val_NIR), nb_features, 2))
	val_spec[:, :, 0] = X_val_NIR
	val_spec[:, :, 1] = X_val_MIR

	test_spec = np.zeros((len(X_test_NIR), nb_features, 2))
	test_spec[:, :, 0] = X_test_NIR
	test_spec[:, :, 1] = X_test_MIR

    # model 4: MIR 2D (spectrogram)
    elif model_num==4:
        m='MIR_2D'

        X_cal_MIR, X_val_MIR= train_test_split(X_train_MIRr, train_size=0.75,random_state=random_seed)
        X_test_MIR = X_test_MIRr

        Sxx1 = spectrogram_std(X_cal_MIR); cal_spec=scaleme(Sxx1,Sxx1)
        Sxx = spectrogram_std(X_val_MIR); val_spec=scaleme(Sxx1,Sxx)
        Sxx = spectrogram_std(X_test_MIR); test_spec=scaleme(Sxx1,Sxx)
        nb_features=cal_spec.shape[1:]; channel_number=1

    # model 5: NIR 2D (spectrogram)
    elif model_num==5:
        m='NIR_2D'

        Sxx1 = spectrogram_std(X_cal_NIR); cal_spec=scaleme(Sxx1,Sxx1)
        Sxx = spectrogram_std(X_val_NIR); val_spec=scaleme(Sxx1,Sxx)
        Sxx = spectrogram_std(X_test_NIR); test_spec=scaleme(Sxx1,Sxx)
        nb_features=cal_spec.shape[1:]; channel_number=1

    #model 6: NIRMIR 2D (spectrogram)
    elif model_num==6:
        m='NIRMIRres_2D'
        X_cal_MIR, X_val_MIR= train_test_split(X_train_MIRr, train_size=0.75,random_state=random_seed)
        X_test_MIR = X_test_MIRr

        tempp1 = spectrogram_std(X_cal_NIR); temp1=scaleme(tempp1,tempp1)
        tempp2 = spectrogram_std(X_cal_MIR); temp2=scaleme(tempp2,tempp2)
        cal_spec = np.zeros((len(X_cal_NIR),temp1.shape[1],temp1.shape[2], 2))
        cal_spec[:,:,:, 0] = np.squeeze(temp1,axis=(3,))#NIR
        cal_spec[:,:,:, 1] = np.squeeze(temp2,axis=(3,)) #MIR

        temp1 = spectrogram_std(X_val_NIR); temp1=scaleme(tempp1,temp1)
        temp2 = spectrogram_std(X_val_MIR); temp2=scaleme(tempp2,temp2)
        val_spec = np.zeros((len(X_val_NIR),temp1.shape[1],temp1.shape[2], 2))
        val_spec[:,:,:, 0] = np.squeeze(temp1,axis=(3,))#NIR
        val_spec[:,:,:, 1] = np.squeeze(temp2,axis=(3,)) #MIR

        temp1 = spectrogram_std(X_test_NIR); temp1=scaleme(tempp1,temp1)
        temp2 = spectrogram_std(X_test_MIR); temp2=scaleme(tempp2,temp2)
        test_spec = np.zeros((len(X_test_NIR),temp1.shape[1],temp1.shape[2], 2))
        test_spec[:,:,:, 0] = np.squeeze(temp1,axis=(3,))#NIR
        test_spec[:,:,:, 1] = np.squeeze(temp2,axis=(3,)) #MIR
        nb_features=cal_spec.shape[1:]; channel_number=2

    #model 7: Outer Product Analysis
    elif model_num==7:
        m='OPA'
        def scaleme(cal_data,new_data):
            data_norm = (new_data - np.mean(cal_data))/np.std(cal_data)
            return data_norm

        temp1 = OPA_res(X_cal_MIR,X_cal_NIR); cal_spec = scaleme(temp1,temp1)
        temp = OPA_res(X_val_MIR,X_val_NIR); val_spec = scaleme(temp1,temp)
        temp = OPA_res(X_test_MIR,X_test_NIR); test_spec = scaleme(temp1,temp)
        nb_features = cal_spec.shape[1:];channel_number=1

    #model 7: Outer Product Analysis with SNV
    elif model_num==8:
        m='OPA wSNV'
        Data= sio.loadmat('KSSL_new3.mat') # with SNV
        X_train_MIR = Data['MIRc'];X_test_MIR=Data['MIRv']
        X_train_NIR = Data['NIRc'];X_test_NIR=Data['NIRv']

        #Split the dataset :
        X_cal_NIR, X_val_NIR, y_calX, y_valX = train_test_split(X_train_NIR, y_train, train_size=0.75,random_state=random_seed)
        X_cal_MIR, X_val_MIR= train_test_split(X_train_MIR, train_size=0.75,random_state=random_seed)

        cal_spec=OPA_res(X_cal_MIR,X_cal_NIR);
        val_spec=OPA_res(X_val_MIR,X_val_NIR);
        test_spec=OPA_res(X_test_MIR,X_test_NIR)
        nb_features = cal_spec.shape[1:];channel_number=1

    if model_type== 1: #for 1D model with multiple output predictions
        nn='MY';nb_outputs=y_cal.shape[1]

        # Create the 1D CNN model
        model=model1D(channel_number,nb_features, multOUT=True);model.summary()

        #create checkpoint
        monitor = EarlyStopping(monitor='val_loss', patience=30, verbose=1,min_delta=0.00001, mode='min')
        redLR=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1,  epsilon=0.0001,mode='min')
        checkpointer = ModelCheckpoint(filepath="best_weights_{}_{}.hdf5".format(model_num, model_type),verbose=0, save_best_only=True) # save best mod
        tb = TensorBoard(log_dir='logs/test_model_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

        # train the model
        h=model.fit(cal_spec,[y_cal[:,0],y_cal[:,1],y_cal[:,2],y_cal[:,3],y_cal[:,4],y_cal[:,5]], validation_data=(val_spec,[y_val[:,0],y_val[:,1],y_val[:,2],y_val[:,3],y_val[:,4],y_val[:,5]]),epochs=EPOCH, batch_size = B_size, callbacks=[monitor,checkpointer,redLR,tb])

        # create predictions using trained model
        TT=model.predict(test_spec)
        Y_pred=np.hstack(TT)

    elif model_type==2:  #for 1D model with single output prediction
        nn='1Y';nb_outputs=1
        for x in range(0,y_cal.shape[1]):
            print(x)

            # Create the 1D CNN model
            model=model1D(channel_number,nb_features, multOUT=False)
            monitor = EarlyStopping(monitor='val_loss', patience=30, verbose=1,min_delta=0.00001, mode='min')
            redLR=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1,  epsilon=0.0001,mode='min')
            checkpointer = ModelCheckpoint(filepath="best_weights_{}_{}.hdf5".format(model_num, model_type),verbose=0, save_best_only=True) # save best mod
            tb = TensorBoard(log_dir='logs/test_model_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

            # train the model
            h=model.fit(cal_spec, y_cal[:,x], validation_data=(val_spec,y_val[:,x]), epochs=EPOCH, batch_size = B_size, callbacks=[monitor,checkpointer,redLR,tb])

            # create predictions using trained model
            tmp=model.predict(test_spec)
            this.append([n for sublist in tmp for n in sublist])
        Y_pred = pd.DataFrame()
        for n in range(len(this)):
            Y_pred[n] = this[n]
            Y_pred

    elif model_type==3: #for 2D model with multiple output predictions
        nn='MY';nb_outputs=y_cal.shape[1]

        # Create the 2D CNN model
        model = model2D(nb_features,multOUT=True)
        monitor = EarlyStopping(monitor='val_loss', patience=30, verbose=1,min_delta=0.00001, mode='min')
        redLR=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1,  epsilon=0.0001,mode='min')
        checkpointer = ModelCheckpoint(filepath="best_weights_{}_{}.hdf5".format(model_num, model_type),verbose=0, save_best_only=True) # save best mod
        tb = TensorBoard(log_dir='logs/test_model_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

        # train the model
        h=model.fit(cal_spec,[y_cal[:,0],y_cal[:,1],y_cal[:,2],y_cal[:,3],y_cal[:,4],y_cal[:,5]], validation_data=(val_spec,[y_val[:,0],y_val[:,1],y_val[:,2],y_val[:,3],y_val[:,4],y_val[:,5]]),epochs=EPOCH, batch_size = B_size, callbacks=[monitor,checkpointer,redLR,tb])

        # create predictions using trained model
        TT=model.predict(test_spec)
        Y_pred=np.hstack(TT)

    elif model_type==4: #for 2D model with single output predictions
        nn='1Y';nb_outputs=1
        for x in range(0,y_cal.shape[1]):
            print(x)
            # Create the 2D CNN model
            model=model2D(nb_features,multOUT=False)
            monitor = EarlyStopping(monitor='val_loss', patience=30, verbose=1,min_delta=0.00001, mode='min')
            redLR=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1,  epsilon=0.0001,mode='min')
            checkpointer = ModelCheckpoint(filepath="best_weights_{}_{}.hdf5".format(model_num, model_type),verbose=0, save_best_only=True) # save best mod
            tb = TensorBoard(log_dir='logs/test_model_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

            # train the model
            h=model.fit(cal_spec, y_cal[:,x], validation_data=(val_spec,y_val[:,x]), epochs=EPOCH, batch_size = B_size, callbacks=[monitor,checkpointer,redLR,tb])

            # create predictions using trained model
            tmp=model.predict(test_spec)
            this.append([n for sublist in tmp for n in sublist])
        Y_pred = pd.DataFrame()
        for n in range(len(this)):
            Y_pred[n] = this[n]
            Y_pred

    elif model_type==5:
        nn='MY';
        model = model2D_OPA(nb_features,multOUT=True)
        monitor = EarlyStopping(monitor='val_loss', patience=30, verbose=1,min_delta=0.00001, mode='min')
        redLR=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1,  epsilon=0.0001,mode='min')
        checkpointer = ModelCheckpoint(filepath="best_weights_{}_{}.hdf5".format(model_num, model_type),verbose=0, save_best_only=True) # save best mod
        tb = TensorBoard(log_dir='logs/test_model_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

        # train the model
        h=model.fit(cal_spec,[y_cal[:,0],y_cal[:,1],y_cal[:,2],y_cal[:,3],y_cal[:,4],y_cal[:,5]], validation_data=(val_spec,[y_val[:,0],y_val[:,1],y_val[:,2],y_val[:,3],y_val[:,4],y_val[:,5]]),epochs=EPOCH, batch_size = B_size, callbacks=[monitor,checkpointer,redLR,tb])

        # create predictions using trained model
        TT=model.predict(test_spec)
        Y_pred=np.hstack(TT)

    #save the model
    model.save(str('KSSL')+str(m)+str(nn)+str('.h5'))

    #inverse back the results
    Y_predT=sc_y.inverse_transform(Y_pred)

    # export the results
    np.savetxt(str('Model')+str(m)+str(nn)+str('.csv'),Y_predT,delimiter=',')
