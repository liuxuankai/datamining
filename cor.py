# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import OrderedDict

from pyexcel_xls import get_data
from pyexcel_xls import save_data

import xlrd

workbook = xlrd.open_workbook('datamining.xlsx')
sheet2 = workbook.sheet_by_index(0)
nrows = sheet2.nrows

df=pd.read_excel('datamining.xlsx')

x=df.iloc[:,1]
for i in range(3,38):
    y = df.iloc[:, i]
    data = pd.DataFrame({'A': x, 'B': y})
    print(i)
    print(data.corr())
    print('\n')


