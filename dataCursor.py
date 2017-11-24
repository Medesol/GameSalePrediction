# -*- coding: utf-8 -*-
"""
UVic SENG 474 Lab project data fetching script

Created on Thur Now 16 16:42:44 2017

@author: Medesol
"""
import pandas as pd
import numpy as np

def dataCursor():
    file_name = "data/merging_output.csv"
    origin_data = pd.read_csv(file_name, encoding = 'gbk')
    X = np.array([origin_data['Platform']], [origin_data['Genre']])
    sales = {
        'na_sales': origin_data['NA_Sales'],
        'eu_sales': origin_data['EU_Sales'],
        'jp_sales': origin_data['JP_Sales'],
        'other_sales': origin_data['Other_Sales'],
        'global_sales': origin_data['Global_Sales']
    }
    print("Success read data")
    return X, sales

if __name__ == "__main__":
    dataCursor()