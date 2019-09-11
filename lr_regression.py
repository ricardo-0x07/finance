import matplotlib, os
matplotlib.use('Agg') 
import numpy as np
import pandas as pd
import argparse
# from sklearn import linear_model
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge,  Lasso
from sklearn.model_selection import train_test_split


dates = []
prices = []

def get_data(filename="AMZN", forecast_out = 14, start="2018-01-01"):
	if not os.path.exists(filename+'.csv'):
		yf.pdr_override()
		df_full = pdr.get_data_yahoo(filename, start=start).reset_index()
		df_full.to_csv(filename+'.csv',index=False)
	else:
		df_full = pd.read_csv(filename+'.csv')
	df_full['Date'] = pd.to_datetime(df_full['Date'])
	df_full.set_index('Date', inplace=True)
	df = df_full[['Adj Close']]
	forecast_out = 14
	df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
	X = np.array(df.drop(['Prediction'],1))
	X = X[:-forecast_out]
	y = np.array(df['Prediction'])
	y = y[:-forecast_out]
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	
	return df, x_train, x_test, y_train, y_test
	
def prediction(clf, df, x_train, x_test, y_train, y_test, name='Linear Regression', type='lr'):

	clf.fit(x_train, y_train)
	clf_score = clf.score(x_test, y_test)
	print('clf_score: ', clf_score)
	df['Prediction'] = df.apply(lambda row: clf.predict([[row.loc['Adj Close']]])[0] if  np.isnan((row.loc['Prediction'])) else row['Prediction'], axis=1)

	plt.rcParams['figure.figsize'] = (15,6)
	plt.plot(df.index.date, df['Adj Close'].values, label= 'Adj Close')
	plt.plot(df.index.shift(21, freq='d').date, df['Prediction'].values, label= 'Prediction')
	plt.title(name)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()
	plt.savefig('Plots/'+type+'.png')
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", help="select regression model. lr for LinearRegression, rr for Ridge, ls for Lasso ", default='lr')
	args = parser.parse_args()
	print('args: ', args)
	df, x_train, x_test, y_train, y_test = get_data()
	if args.model == 'lr':
		name = 'Linear Regression'
		clf = LinearRegression()
		prediction(clf, df, x_train, x_test, y_train, y_test, name=name, type=args.model)
	if args.model == 'rr':
		name = 'Ridge Regression'
		clf = Ridge(alpha=100)
		prediction(clf, df, x_train, x_test, y_train, y_test, name=name, type=args.model)
	if args.model == 'ls':
		name = 'Lasso Regression'
		clf = Lasso()
		prediction(clf, df, x_train, x_test, y_train, y_test, name=name, type=args.model)




if __name__ == '__main__':
    main()