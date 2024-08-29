#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:24:58 2024

@author: qb
"""
import numpy as np 
import pandas as pd 
import h5py
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
import predictions
#it is crutial to use the same objects for laberEncoder for testing and training


class clean_data():
    def __init__(self, x_train_data, data):
        self.x = x_train_data
        self.data = data     
        
    def _clean(self):
        print('preprocessing')
        
        
        
        self.data = pd.DataFrame.from_records(data = self.data, columns = self.x.columns)
        
        print('**************recived data to clean', self.data)
        
        #replacing null/none with nana
        self.data.fillna(np.NAN, inplace=True)
        #if all the rows are null drop it
        
        self.data['is_na']=self.data.isnull().apply(lambda x: all(x), axis = 1)
        self.data.drop(self.data[self.data['is_na'] == True].index, inplace=True)
        
        self.data.drop(columns = {'is_na'}, inplace=True)
        
        print('***************** inprogresss', self.data)
        #handle preprocesing
        self.data.rename(columns = {'name':'company'}, inplace=True)
        
        
        
        print(self.data.columns)
        
        
        
        re = self.imputation_d(self.data)
        
        
        # general practice but in this case it does not matter.
                
        x_pr = pd.read_csv('requires/train.csv')
        x_pr['owner'].loc[x_pr[(x_pr['owner'] == 'Fourth & Above Owner') | (x_pr['owner'] == 'Third Owner')].index] = 'others'

        
        re = self.features(x=re[:-1], x_pr = x_pr)
        
    
        
    
        return re

    def imputation_d(self,x):
        if x.shape[0] > 0:            
            print('**** imputation process')
            x['engine']=x['engine'].apply(lambda x: str(x).split()[0]).astype('float')
            x['index'] = x.index
            r=x.groupby('company')['engine'].apply(lambda x: x.fillna(x.median()))
            x.set_index(['company', 'index'], inplace=True)
            x['engine'] = r
            x.reset_index(inplace=True)
            
        
            #even though the full distribution is centered to mean but so mean is close to median , 
            #but here filling up each distribution with median as per the company
            # filling missing values for milage
            x['index'] = x.index
            r = x.groupby('company')['mileage'].apply(lambda x: x.fillna(x.mean()))
            x.set_index(['company', 'index'], inplace=True)
            x['mileage'] = r
            x.reset_index(inplace=True)
            
            if x['engine'].max() > 1800:
                x['binned_engine']=pd.cut(x['engine'], [1, 1000, 1500, 1800, x['engine'].max()], labels = list(range(1,5))[::-1])
            else:
                x['binned_engine']=pd.cut(x['engine'], [1, 1000, 1500, 1800, 2500], labels = list(range(1,5))[::-1])            
            x['binned_engine'] = x.binned_engine.astype('int')
        
            #the distribution of max_power is not mean centered as whole so each group will be assumed (because the data is MNAR) to have median of each 
            #maxpower engine produced by companies
            x['index'] = x.index
            r = x.groupby('company')['max_power'].apply(lambda x: x.fillna(x.median()))
            x.set_index(['company', 'index'], inplace=True)
            x['max_power'] = r
            x.reset_index(inplace=True)
        
            # seat is not the ordinal data if that so enconding required
            # filling na values by filling in with a company not sure if this make sense 
            ix=x[x['seats'].isna()].index
            for i in ix:
                name =x.loc[i]['company']
                min_seats = x[x['company'] == name]['seats'].min()
                max_seats = x[x['company'] == name]['seats'].max()
                if not pd.isna(min_seats):
                    x['seats'].loc[i]=round(max_seats/min_seats) 
                    
            x.to_csv('requires/c.csv')
            return x
        
        
    def features(self,x, x_pr, phase='Traning', encoding='one_hot'):
   #     x['co_cc']=x.groupby(['company'])['year'].transform(lambda x: sum(x-x.min())).astype('float')
            
       
        gprs = x_pr.groupby(['company', 'year'])
        l = [(c,int(y)) for c in x['company'] for y in x[x['company'] == c]['year']]
        # print('LLLLLLLLLLLLLLLLLLLLLLLL',l)
        #x.set_index(['company', 'year'], inplace=True)
        x['year'] = x['year'].astype('int')
        x['company'] = x['company'].astype('str')
        for g in l:
            if g in gprs.groups.keys():
                ic = gprs.get_group(g)
                m = ic['_num_sales'].mean()
              
                idx =x[(x['company'] == g[0]) & (x['year'] == g[1])].index
                for i in idx:
                    x.at[i,'_num_sales'] = m
                    x.at[i, 'co_cc'] = ic['co_cc'].iloc[0]
                print('does it works here rrrrrrrrrrrr,', x['_num_sales'])
            #x.loc[g]['co_cc'] = ic['co_cc'].iloc[0]
                
        #x.reset_index(inplce=True)
        # x['_num_sales'] = x.groupby(['company', 'year'])['name'].transform('count')
            
        
        with open('requires/transmission.pickle', 'rb') as f:
            t = pickle.load(f)
        
        x['transmission'] = pd.Series(t.transform(x['transmission'].values.reshape(-1,1)).astype('int').flatten())
   
        with open('requires/fuel.pickle', 'rb') as f:
            f_= pickle.load(f)
            
        m = f_.transform(np.array(x['fuel']).reshape(-1,1)).toarray()
        x[['Diesel','petrol']] = pd.DataFrame.from_records(m , columns = x_pr['fuel'].unique())
    
        
        if encoding == 'one_hot':
            with open('requires/owner.pickle', 'rb') as f:
                b  = pickle.load(f)
            m = b.transform(np.array(x['owner']).reshape(-1,1)).toarray()
            x[['firstowner', 'secondowner', 'others']] = pd.DataFrame.from_records(m, columns = x_pr['owner'].unique())
        
        if encoding == 'oridinal':
            with open('requires/owner_ordinal.pickle', 'rb') as f:
                b  = pickle.load(f)
            x['_owner_encoded'] = pd.Series(b.transform(x['owner'].values.reshape(-1,1)).astype('int').flatten())

        
#     x['_owner_encoded']=x.groupby('company')['_owner_encoded'].transform(lambda x: x/x.max())
    
        with open('requires/company.pickle', 'rb') as f:
            o = pickle.load(f)
        x['_company'] = o.transform(x['company'].values.reshape(-1,1)).astype('int')

        r = x.groupby(['company', 'year'])['seats'].value_counts()
        x.set_index(['company', 'year', 'seats'], inplace = True)
        x['_seats_in_year'] = r
        x.reset_index(inplace=True)
        # x['km_distances']=x.groupby(['company', 'year','owner'])['km_driven'].transform(lambda x: x.mean())
        # x['km_distances'] = x['km_distances']
        l= [(c,y,o) for c in x['company'] for y in x[x['company'] == c]['year'] for o in x[(x['company'] == c) & (x['year'] == y)]['owner']]
        grps = x_pr.groupby(['company', 'year', 'owner'])
        
        ###the average might change but for prediction we are just taking the mean directly
        
        for g in l:
            if g in grps.groups.keys():
                ic = grps.get_group(g)
                idx = x[(x['company'] == g[0]) & (x['year'] == g[1]) & (x['owner'] == g[2])].index 
                for i in idx:
                    x.at[i,'km_distances'] = ic['km_distances'].iloc[0]
        
        
        return x
    
    def perfom_prediction(self,data):
        
        #drop all the columns
        data.to_csv('requires/data.csv')
        drop_columns = ['engine','seats','owner',
                'index', 'fuel', 'transmission', 'km_driven', 'seller_type']
        
        data[['year','mileage', 'max_power']] = data[['year','mileage', 'max_power']].astype('int')
        
        print('going for prediction', data.columns)
        data.drop(columns = drop_columns, inplace=True)

        o=predictions.predict(data)
        scaler =pickle.load(open('requires/scaler_train', 'rb'))
        preds=scaler.inverse_transform(o.predict().reshape(-1, 1))
        
        blended_pred = o.predict_blended_lm(0.04, scaler, pred_=preds)
        print(preds)
        print(blended_pred)
        
        return np.array(preds).ravel(), np.array(blended_pred[0]).ravel()