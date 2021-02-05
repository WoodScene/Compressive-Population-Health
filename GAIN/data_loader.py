'''
Data loader for GAIN
'''

# Necessary packages
import os
import numpy as np
import pandas as pd
from utils import binary_sampler
from keras.datasets import mnist

def data_loader(miss_rate, yy):
  diease_list = ['Obesity Prevalence', 'Hypertension Prevalence', 'Diabetes Mellitus (Diabetes) Prevalence']
  year = yy
  diease_select_list = 0 # target disease
  #year = 2017
  N1 = 483
  N3 = 2017 - year + 1
  data_x = np.ones((N1, N3), dtype='float64')
  data_m = np.ones((N1, N3), dtype='float64')

  for y in range(year,2017+1):
    df = pd.read_csv("./data/Chronic_Diseases_Prevalence_Dataset.csv")
    ward_code_list=list(df['Ward Code'])
    # print(list(df))
    df = df[diease_list[diease_select_list]+"_"+str(y)]

    data_x[:,y - year] = df.values

  miss_data_x = data_x.copy()

  ward_number = int(N1 * (100 - miss_rate*100) / 100)

  for y in range(N3 - 1, N3):
    data_year = data_x[:,y]

    ward_list = []
    ward_nor_list = []
    num = 0
    df_ward = pd.read_csv("./data/Variance_2008_2017_" + diease_list[diease_select_list] + "_NORMALIZE.csv")

    df_diease = pd.read_csv("./data/Ward_code_list.csv")
    ward_code_old = list(df_diease['Ward Code'])

    ward_var = list(df_ward["Ward_id_" + str(year) + "_" + str(2017)])
    iii = 0
    while num < ward_number:
      id = ward_var[iii]
      iii += 1
      ward_code = ward_code_old[id]
      if ward_code in ward_code_list:
        index1 = ward_code_list.index(ward_code)
        diease_rate = data_year[index1]
        if diease_rate!=0:
          num += 1
          ward_list.append(index1)

    for i in range(N1):
      if i in ward_list:
        continue
      ward_nor_list.append(i)
      data_m[i,-1] = 0
    #print("ward_list", sorted(ward_list))

  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m, ward_nor_list