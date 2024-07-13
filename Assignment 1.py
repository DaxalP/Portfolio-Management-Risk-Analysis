import pandas as pd
import math as math

stock = pd.read_csv('sample_prices.csv')

blue = stock['BLUE']
orange = stock['ORANGE']

def returns(stocks):
    market = []
    for a in stocks:
        market.append(a)
    
    i=2
    market_ret = [round((market[1]-market[0])*100/(market[0]),4)]
    while i<=(len(market)-1):
        diff = round((market[i]-market[i-1])*100/market[i],4)
        market_ret.append(diff)
        i=i+1
    
    return(market_ret)
        
print("Returns of stock BLUE is : ",returns(blue))
print("\nReturns of stock ORANGE is : ",returns(orange))

def sharpe(b):
    
    portret = (b[12]-b[0])/b[0]*100
    avgret = sum(returns(b))/12
    i=0
    vari = 0
    
    while i<12:
        vari = vari + (returns(b)[i]-avgret)*(returns(b)[i]-avgret)
        i=i+1
    
    std = math.sqrt(vari/12)
    
    sharp = (portret-6)/std
    #Assuming risk free return for these indian markets is 6 !!!
    
    return(round(sharp,2))
    
    
print("\nThe sharpe ratio for stock BLUE is : ",sharpe(blue))
print("\nThe sharpe ratio for stock ORANGE is : ",sharpe(orange))
      

def annualret(c):
    
    annualreturn = ((c[12]/c[0])**(1/12)-1)*100
    
    return round(annualreturn,2)

print("\nThe annual rate of return of stock BLUE is : ",annualret(blue))
print("\nThe annual rate of return of stock ORANGE is : ",annualret(orange))