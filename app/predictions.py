#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:06:28 2024

@author: qb
"""

import pickle
import pandas as pd
import numpy as np 
import xgboost
import copy

class predict:
    def __init__(self, x_):
        self.x_ = x_
        self.reg = pickle.load(open('requires/xgb1', 'rb'))
        self.regid = pickle.load(open('requires/lm_.pkl', 'rb'))
        
    def predict(self):
        #simple prediction
        
        cols = self.reg.get_booster().feature_names
        self.x_=self.x_[cols]
        return self.reg.predict(self.x_)
        
    def predict_blended_lm(self, sigma, scaler, pred_):
        'pred_ from xgb'
        cols = self.regid.feature_names_in_
        x_prime = copy.deepcopy(self.x_[cols])
        'if data provided by the user does not exists then lm will not perfom as we have no '
        'mechanism of out of sample missing value handling right now'
        idx = x_prime.isna().index 
        x_prime = x_prime.drop(idx)
        
        if x_prime.shape[0] > 0:
            pred = self.regid.predict(x_prime)
            r_=scaler.fit_transform(pred.reshape([-1,1])).flatten()
            return pred_ + sigma*pred
        # else:
        #     r_=scaler.fit_transform([[pred]]).flatten()
        else:
            return pred_
        
    def pertubed_predict(self, sample_size, num_rep):
        #create quantiles here
        #random sample from test data =
        
        
        '''pertube perdiction'''
        '''sample_idx = []
        for i in range(sample_size):
            sample_idx.append(round(np.random.uniform(self.test.shape[0])))

        predictions =[]
        #mix data 
        for j in range(num_rep):
            for i in range(self.x_.shape[0]):
                idx = round(np.random.uniform(len(sample_idx)))
                test_ = self.test.loc[sample_idx]
                test_.loc[idx] = self.x_.loc[i]
                test_['year']=test_['year'].astype('int')
                test_['mileage'] = test_['mileage'].astype('float')
                pred=predictions.append(self.reg.predict(test_))
                predictions.append((pred[idx], 1))
        
        return predictions'''
    