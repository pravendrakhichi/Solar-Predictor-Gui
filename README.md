# Solar-Predictor-Gui
A Keras model which can predict and retrain model for real time purpose.

Preprocessing Data/ you can change this according to your requirement of preprocessing.

        clean=pd.read_csv(self.csv_file)
        clean=clean.drop('Unnamed: 0',1)
        clean=clean.drop('DNI',1)
        clean=clean.drop('DIF',1)
        clean=clean.drop('Minutes',1)
        clean=clean.drop('Month',1)
        clean=clean.drop('Year',1)
        print('series to supervised')
        self.series_to_supervised(clean,n_in=4,n_out=0)
        shifted_frame=self.agg
        
        print('normalizing')
        print(shifted_frame)
        clean=clean[:-4]
        # normalize the dataset
        train_scaler = MinMaxScaler(feature_range=(-1 , 1))
        X_train = train_scaler.fit_transform(clean)
        # split into train and test sets
        
        y_train=shifted_frame.values
        print('scaled \n')
        print(y_train)
        print(X_train)
        print("%s %d" %(len(X_train),len(y_train)))
        look_back=30
        train_data_gen = TimeseriesGenerator(X_train, y_train,
                length=look_back, sampling_rate=1,stride=1,
                batch_size=1)
        print('generated train data')
Code for retraining the model.

        print('starting retraining ')
	#Model name must be given.
        model2 = load_model(self.csv_file1)
        model2.compile(loss='mean_squared_error', optimizer='RMSProp')
        t=datetime.now(pytz.utc)
        t=str(t)
        t=t[:10]
        g='Retrained Model '+t+'.h5'
        model2.save(g)
        history = model2.fit_generator(train_data_gen,epochs=10).history


        t=datetime.now(pytz.utc)
        t=str(t)
        t=t[:10]
        g='Retrained Model '+t+'.h5'
        model2.save(g)
	
Series to Supervised 

    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
            Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        print('started')
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        self.agg = concat(cols, axis=1)
        self.agg.columns = names
        # drop rows with NaN values
        if dropnan:
            self.agg.dropna(inplace=True)

Predictng data in which you will get a csv file according to no of predictions you want.

	train_scaler = MinMaxScaler(feature_range=(0, 1))
        clean_sc = train_scaler.fit_transform(clean)
        crow=clean.values
        X_train,X_test,y_train,y_test=train_test_split(clean_sc,crow,shuffle=False)
        ast=train_scaler.transform(clean)
        p=X_test[:-30]

        print('scaling\n')
        global new_data
        new_data=X_test[:-60]
        day_to_pre=self.spinbox_value
        print(day_to_pre)
        data_of_a_day=20
        pre=[]
        #print(*[p[i] for i in range(60)] ,sep='\n')
        print('predicting')
        look_back=30
        prediction_seqs = []
        curr_frame = new_data[-31:]

        print('time series checked')
        predict=[]
        l=0
        for i in range(int(day_to_pre*data_of_a_day/4)):
            cow_data=new_data[-31:]
            print(*cow_data,sep='\n')
            pre_gen = TimeseriesGenerator(cow_data, np.array([i for i in range(len(cow_data))]),
            length=look_back, sampling_rate=1,stride=1,
            batch_size=1)
            x,y=pre_gen[0]
            print('time series checked')
            print(*x,sep='\n')
            predict=model2.predict_generator(iter(pre_gen),1)
            print("\n model2 checked \n")
            print(*predict,sep='\n')
            predict=predict.reshape(4,8)
            print('reshaped')
            for j in range(4):
                pre.append(predict[j])
            print('appended')

            predict=train_scaler.transform(predict)
            new_data = np.vstack([new_data, predict])
            #print(new_data.shape)
            print('stacked')
            l+=1
        p=0
        print('saving data')
        print('%d   %f'%(day_to_pre,len(pre)))
        pred=np.array([pre[i][p] for i in range(20*day_to_pre)])

        # not aloud print("%d \n"%[pre[i][p] for i in range(40)])
        #print(*pred,sep="\n")
        #plt.plot(pred,'orange')
        #plt.plot(original,'blue')
        #plt.show()

        print(*pred,sep='\n')
      
        pred=pred.reshape(-1,1)
        t=datetime.now(pytz.utc)
        t=str(t)
        t=t[:10]
        g='Predicted Data '+t+'.csv'
        #self.predict_table_window(pred)
        np.savetxt(g,pred , delimiter=",")
        print('all done')
        print('%s %d'%(l,len(pre)))
