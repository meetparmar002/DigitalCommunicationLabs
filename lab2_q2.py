# Lab2 - Problem 2
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt

df = ps.read_excel('Gandhinagar_RainfallData.xls')  # importing xls file

# deleting first column which consists years
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# deleting first row which consists months names
df.drop([0], inplace=True)

dic = {  # to rename the column name
    'Unnamed: 1': 'Jan',
    'Unnamed: 2': 'Fab',
    'Unnamed: 3': 'Mar',
    'Unnamed: 4': 'Apr',
    'Unnamed: 5': 'May',
    'Unnamed: 6': 'Jun',
    'Unnamed: 7': 'Jul',
    'Unnamed: 8': 'Aug',
    'Unnamed: 9': 'Sep',
    'Unnamed: 10': 'Oct',
    'Unnamed: 11': 'Nev',
    'Unnamed: 12': 'Dec'
}

df.rename(columns=dic, inplace=True)  # just renaming the columns
mx = df.mean(axis=0)  # list of all means


months = ['Jan', 'Fab', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nev', 'Dec']

print('means of every months:')
print(mx)

n1, n2 = int(input('Enter index of first month(0 based): ')), int(
    input('Enter index second month(0 based): '))

# function to find the covariance of any two months
def covariance(n1, n2):
    cov_list = (df[months[n1]] - mx[n1]) * (df[months[n2]] - mx[n2])
    covar = cov_list.sum()
    return covar/102


print('Covariance of '+months[n1]+' and '+months[n2]+': ')
print('%.6f' % covariance(n1, n2))
