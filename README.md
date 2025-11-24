# 📈 Portfolio Management & Risk Analysis (PMRA)

### 📘 Project Overview

---

This project explores how to build and evaluate investment portfolios using concepts from modern portfolio theory and quantitative risk management.
It focuses on understanding how risk and return interact, how portfolios behave under different market conditions, and how disciplined strategies like CPPI (Constant Proportion Portfolio Insurance) can protect investors from large losses.

The project includes:

Collecting real market data from Yahoo Finance

Analyzing risk/return characteristics of different assets

Building optimal portfolios using the Efficient Frontier

Designing a dynamically protected strategy (CPPI)

Comparing performance through charts, backtests, and statistical summaries

It aims to provide a practical, hands-on introduction to how professional portfolio managers balance growth with downside protection.

---

## 🚀 Features

### **1. Return & Risk Computation**
- Fetch historical stock data using Yahoo Finance.
- Compute:
  - Simple returns  
  - Annualized returns & volatility  
  - Skewness & kurtosis  
  - Sharpe ratio  
  - Drawdowns  

**Functions:**  
`get_returns()`, `annualize_rets()`, `annualize_vol()`, `sharpe_ratio()`, `drawdown()`, `skewness()`, `kurtosis()`

### **2. Portfolio Optimization (Markowitz Model)**
Tools to build and analyze optimal portfolios:

- Portfolio return & volatility  
- Minimum-variance portfolios  
- Maximum Sharpe Ratio (MSR) portfolio  
- Efficient Frontier visualization  

**Functions:**  
`portfolio_return()`, `portfolio_vol()`, `minimize_vol()`, `optimal_weights()`, `plot_ef()`, `msr()`


### **3. Constant Proportion Portfolio Insurance (CPPI)**
Implements dynamic allocation between risky and safe assets based on the cushion rule.

- Wealth evolution  
- Risk budget  
- Cushion and floor tracking  
- Backtested results  

**Function:**  
`run_cppi()`


### **4. Summary Statistics**
Generate full statistical metrics for any return series:

**Function:**  
`summary_stats()`

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- SciPy (SLSQP optimization)  
- Matplotlib  
- yfinance  

---

🎯 Purpose

This project showcases practical applications of:

Modern portfolio theory

Risk-adjusted portfolio construction

Dynamic downside protection (CPPI)

Real-market backtesting

Useful for quantitative finance learning, research, and investment simulations.