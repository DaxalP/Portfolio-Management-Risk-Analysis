import pandas as pd
import numpy as np
import math
from scipy.stats import jarque_bera,norm
from datetime import datetime
from datetime import timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd

def run_cppi(risky_rets, safe_rets=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, Drawdown=None):
    
    n_steps = len(risky_rets.index)
    account_value = start
    floor_value = start*floor
    peak = account_value
    safe_rets = pd.DataFrame().reindex_like(risky_rets)
    safe_rets.values[:] = riskfree_rate/12 
    
    account_history = pd.DataFrame().reindex_like(risky_rets)
    risky_w_history = pd.DataFrame().reindex_like(risky_rets)
    cushion_history = pd.DataFrame().reindex_like(risky_rets)
    floorval_history = pd.DataFrame().reindex_like(risky_rets)
    peak_history = pd.DataFrame().reindex_like(risky_rets)
    
    dd_history = drawdown(risky_rets)

    for i in range(n_steps):
        peak = np.maximum(peak, account_value)
        floor_value = peak*(1-Drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        
        account_value = risky_alloc*(1+risky_rets.iloc[i]) + safe_alloc*(1+safe_rets.iloc[i])
       
        cushion_history.iloc[i] = cushion
        risky_w_history.iloc[i] = risky_w
        account_history.iloc[i] = account_value
        floorval_history.iloc[i] = floor_value
        peak_history.iloc[i] = peak
        
    risky_wealth = start*(1+risky_rets).cumprod()
    
    result_dataframe = {
        "account_hist": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "risky_rets":risky_rets,
        "safe_rets": safe_rets,
        "drawdown_hist": dd_history,
        "peak_hist": peak_history,
        "floor_hist": floorval_history
    }
    return result_dataframe

def msr(rfr, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights,rfr ,er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = port_ret(weights, er)
        vol = port_vol(weights, cov)
        return -(r-rfr)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(rfr, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def minimize_vol(target_return, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - erk.portfolio_return(weights,er)
    }
    weights = minimize(erk.portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(er, cov) for target_return in target_rs]
    return weights

def get_returns(stocks):
    df = pd.DataFrame()
    for file in stocks:
        data = pd.read_csv(f'{file}.csv', index_col=0, parse_dates=True)
        df[f'{file}'] = (data['Close']-data['Open'])*100/data['Open']
    return df

def volatility(data, date):
    delta = date[len(date)-1]-date[0]
    v = np.std(data) * (delta.days)**(0.5)
    return v

def get_stock_data(symbols, start_date, end_date):
    data = pd.DataFrame()  # Empty DataFrame to store the stock returns

    for symbol in symbols:
        ticker = symbol
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval='1mo')

        # Calculate returns (closing price divided by opening price)
        stock_returns = (stock_data['Close']-stock_data['Open'])*100 / stock_data['Open']

        # Rename the Series with the stock symbol
        stock_returns = stock_returns.rename(symbol)

        data = pd.concat([data, stock_returns], axis=1)  # Concatenate the returns for each stock
        
    data.index.name='Date'
    data.index = pd.to_datetime(data.index).date
    return data

# to convert index column into date :-
def read_with_date(data, format_of_date):
    data.index = pd.to_datetime(data.index, format = format_of_date)
    
# function to calculate Sharpe Ratio :-
def sharpe(series, Risk_free_return):
    size = len(series)
    diff = (series[size-1]-series[0])/series[0]
    std = np.std(series)
    S = round((diff-Risk_free_return)/std, 3)
    return S

# def sharpe_ratio(r, riskfree_rate, periods_per_year):
#     """
#     Computes the annualized sharpe ratio of a set of returns
#     """
#     # convert the annual riskfree rate to per period
#     rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
#     excess_ret = r - rf_per_period
#     ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
#     ann_vol = annualize_vol(r, periods_per_year)
#     return ann_ex_ret/ann_vol

# function to calculate Moment for different series and degree :-
def moment(series, degree):
    mean = np.mean(series)
    moment = np.mean((series - mean)**(degree))
    return moment

# function to calculate Kurtosis for given return series :-
def kurtosis(series):
    kurtosis = moment(series,4)/(moment(series,2)**2)
    return kurtosis

# function to calculate Skewness for given return series :-
def skewness(series):
    skewness = moment(series,3)/(moment(series,2)**1.5)
    return skewness 

# function to check weather return series is normal distribution or not :-
# output = 'True' means return series is example of normal distribution.
def jarque_bera_normality(series):
    norm = jarque_bera(series)
    return (norm[1]>0.05)

# function to calculate Annualize Returns for a given return series for a particular year :-
def annualized_returns(data):
    returns = []
    names = []
    for stock in data:
        names.append(stock)
        a = data[stock]
        flag=0
        prod = 1
        for b in a:
            c = (1+(b/100))
            prod = prod*c
            flag=flag+1
        ret = ((prod)**(1/flag))-1
        returns.append(ret)
    
    df = pd.DataFrame(returns, columns=['Annualized Returns'])
    df['Stocks']= names
    
    ar = df.groupby('Stocks')['Annualized Returns'].mean()*100
#     df = pd.DataFrame(ar)
    # We are grouping our returns with stocks so that later we can access the annualized returns by "ar[stock_name]".
    return ar

def port_sr(w, ar, cov, risk_free_returns):
    er = port_ret(w,ar)
    vol = port_ret(w,cov)
    sr = (er-risk_free_returns)/vol
    return er
    

def portfolio_plot(n_port, er, cov):
    expected_returns = np.zeros(n_port)
    expected_vol = np.zeros(n_port)
    n = er.shape[0]
    weights = np.zeros((n_port,n))
    for k in range(n_port):
        w = np.array(np.random.random(n))
        w =w /np.sum(w)
        weights[k,:] = w
    
        expected_returns[k]= port_ret(w,er)
        expected_vol[k] = port_vol(w, cov)

    opt_wt = msr(0.03, er, cov)
    x = port_vol(opt_wt, cov)
    y = port_ret(opt_wt, er) 
    plt.title("Efficient Frontier Plot")
#     plt.figure(figsize=(8,8))
#     plt.scatter(expected_vol,expected_returns)
    plt.scatter(x, y, color='red', marker='o', label='Point')
    plt.legend(['Optimal Portfolio'])
    
    max_ret = expected_returns.max()
    min_ret = expected_returns.min()
    Rets = np.linspace(min_ret,max_ret,50)
    opt_vol = []
    init_guess = np.repeat((1/n), n)
    bounds = ((0.,1.),)*n
    for a in Rets:
        constraints = ({'type': 'eq', 'fun': lambda w: 1-np.sum(w)},
                       {'type': 'eq', 'args':(er,), 'fun': lambda w,er: port_ret(w,er)-a})
        min_vol = minimize(port_vol, init_guess, args=(cov,), method='SLSQP', bounds=bounds, constraints = constraints)
        opt_vol.append(min_vol['fun'])
        
    plt.plot(opt_vol, Rets,'--', color= 'Green')

def cov(data):
    df = pd.DataFrame()
    for a in data:
        df[f'{a}'] = data[a]/100
    
    cov_mat = df.cov()
    return cov_mat


def port_vol(weights, cov):
    mat = weights.T @ cov
    mat = mat @ weights
    return np.sqrt(mat)
    
def port_ret(weights, er):
    return weights.T @ er

# function to calculate Annualize Volatility for a given return series for a particular year :-
def annualized_volatility(data_fund, year):
    a = data_fund[year]
    std = np.std(a)
    size = len(a)
    value = std*np.sqrt(size)
    return value

# function to calculate Drawdown for given return series from 1926 to 2018 :-
# we can calculate Drawdown for particular years by just giving input argument as "data['stock']['year']" ...

def drawdown(data):
    wealth = 1000*(1+(data/100)).cumprod()
    pp = wealth.cummax()
    value = (wealth-pp)/pp
    return value

# function to calculate Minimum Drawdown for given series :-
def min_drawdown(data):
    return min(drawdown(data))

# function to calculate Semi-Deviation for a given return series/data :-
def semi_deviation(data):
    mean = np.mean(data)
    deviations = [x - mean for x in data if x < mean]
    squared_deviations = [x ** 2 for x in deviations]
    semi_variance = np.sum(squared_deviations) / len(data)
    semi_deviation = np.sqrt(semi_variance)
    return semi_deviation

# function to calculate Historical Value At Risk for a given return series and for a particular confidence level :-
def hist_var(series,confidence_level):
    var = np.percentile(series,(100-confidence_level))
    return var

# function to calculate Historical Conditional Value At Risk for a given return series and for a particular confidence level :-
def hist_cvar(series, confidence_level):
    data = np.sort(series)
    index = int((1-(confidence_level/100))*len(data))
    returns = data[:index]
    var = np.mean(returns)
    return var

# function to calculate Gaussian Value At Risk for a given return series and for a particular confidence level :-
def gaussian_var(series, confidence_level):
    mean = np.mean(series)
    std = np.std(series)
    z = norm.ppf((100-confidence_level)/100)
    var = mean + std*z
    return var

# function to calculate Gaussian Conditional Value At Risk for a given return series and for a particular confidence level :-
def guassian_cvar(data,conf):
    old_z = norm.ppf(1-(conf/100))
    Z = -(1/(1-(conf/100)))*(1/(np.sqrt(2*(math.pi))))*(math.e**(-0.5*(old_z**2)))
    var = np.mean(data) + np.std(data)*Z
    return var

# function to calculate Cornish Fisher Value At Risk for a given return series and for a particular confidence level :-
def cornish_fisher_var(data,conf):
    s = skewness(data)
    k = kurtosis(data)-3  #excess kurtosis = kurtosis - 3
    z = norm.ppf(1-(conf/100))
    Z = -(1/(1-(conf/100)))*(1/(np.sqrt(2*(math.pi))))*(math.e**(-0.5*(z**2)))
    w = Z + (Z**2 - 1)*(s/6) + Z*(Z**2 - 3)*(k/24) - Z*(2*(Z**2) - 5)*((s**2)/36)
    return np.mean(data) + np.std(data)*w