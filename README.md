# Solar-Predictor-Gui
A Keras model which can predict and retrain model for real time purpose.

Preprocessing Data

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
Code we developed for retraining the model.

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
