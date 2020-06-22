
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
#UXBAWQ4K8LK7WYKF free API key

  
financial_dir = {}

stocks=['ZEEL']
for ticker in stocks:
    
    #getting balance sheet data  for the given ticker
    temp_dir = {}
    url = 'https://in.finance.yahoo.com/quote/'+ticker+'.BO/balance-sheet?p='+ticker+'.BO'
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.find_all("div", {"class" : "M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)"})
    for t in tabl:
        rows = t.find_all("div", {"class" : "rw-expnded"})
        for row in rows:
            temp_dir[row.get_text(separator='|').split("|")[0]]=row.get_text(separator='|').split("|")[1]
    
    #getting income statement data for the given ticker
    url = 'https://in.finance.yahoo.com/quote/'+ticker+'.BO/financials?p='+ticker+'.BO'
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.find_all("div", {"class" : "M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)"})
    for t in tabl:
        rows = t.find_all("div", {"class" : "rw-expnded"})
        for row in rows:
            temp_dir[row.get_text(separator='|').split("|")[0]]=row.get_text(separator='|').split("|")[2]
    
    #getting cashflow statement data for the given ticker
    url = 'https://in.finance.yahoo.com/quote/'+ticker+'.BO/cash-flow?p='+ticker+'.BO'
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.find_all("div", {"class" : "M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)"})
    for t in tabl:
        rows = t.find_all("div", {"class" : "rw-expnded"})
        for row in rows:
            temp_dir[row.get_text(separator='|').split("|")[0]]=row.get_text(separator='|').split("|")[2]
    #getting key statistics data for the given ticker
    url = 'https://in.finance.yahoo.com/quote/'+ticker+'.BO/key-statistics?p='+ticker+'.BO'
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content,'html.parser')
    tabl = soup.findAll("table")
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            if len(row.get_text(separator='|').split("|")[0:2])>0:
                temp_dir[row.get_text(separator='|').split("|")[0]]=row.get_text(separator='|').split("|")[-1]    
    
    #combining all extracted information with the corresponding ticker
    financial_dir[ticker] = temp_dir
    
    
    
#storing information in pandas dataframe
combined_financials = pd.DataFrame(financial_dir)
combined_financials.dropna(how='all',axis=1,inplace=True) #dropping columns with all NaN values
tickers = combined_financials.columns #updating the tickers list based on only those tickers whose values were successfully extracted

# creating dataframe with relevant financial information for each stock using fundamental data
stats = ["EBITDA",
         "Accumulated depreciation",
         "Market cap (intra-day)",
         "Net income available to common shareholders",
         "Net cash provided by operating activities",
         "Capital expenditure",
         "Total current assets",
         "Total current liabilities",
         "Net property, plant and equipment",
         "Total stockholders' equity",
         "Long-term debt",
         "Forward annual dividend yield"] # change as required

indx = ["EBITDA","D&A","MarketCap","NetIncome","CashFlowOps","Capex","CurrAsset",
        "CurrLiab","PPE","BookValue","TotDebt","DivYield"]
all_stats = {}
for ticker in stocks:
    try:
        temp = combined_financials[ticker]
        ticker_stats = []
        for stat in stats:
            ticker_stats.append(temp.loc[stat])
        all_stats['{}'.format(ticker)] = ticker_stats
    except:
        print("can't read data for ",ticker)

all_stats_df = pd.DataFrame(all_stats,index=indx)
for ticker in stocks:
    all_stats_df[ticker] = all_stats_df[ticker].replace({'-':'0'},regex=True)
    all_stats_df[ticker] = all_stats_df[ticker].replace({'M': 'E+03'}, regex=True)
    all_stats_df[ticker] = all_stats_df[ticker].replace({'B': 'E+06'}, regex=True)
    all_stats_df[ticker] = all_stats_df[ticker].replace({'T': 'E+09'}, regex=True)
    all_stats_df[ticker] = all_stats_df[ticker].replace({'N/A': '0%'}, regex=True)
    all_stats_df[ticker] = all_stats_df[ticker].replace({'%': 'E-02'}, regex=True)
    

all_stats_df[tickers] = all_stats_df[tickers].replace({',': ''}, regex=True)
for ticker in stocks:
    all_stats_df[ticker] = pd.to_numeric(all_stats_df[ticker].values,errors='ignore')
all_stats_df.dropna(axis=1,inplace=True)

# calculating relevant financial metrics for each stock
transpose_df = all_stats_df.transpose()
final_stats_df = pd.DataFrame()
final_stats_df["EBIT"] = transpose_df["EBITDA"] - transpose_df["D&A"]
final_stats_df["TEV"] =  transpose_df["MarketCap"].fillna(0) \
                         +transpose_df["TotDebt"].fillna(0) \
                         -(transpose_df["CurrAsset"].fillna(0)-transpose_df["CurrLiab"].fillna(0))
final_stats_df["EarningYield"] =  final_stats_df["EBIT"]/final_stats_df["TEV"]
final_stats_df["FCFYield"] = (transpose_df["CashFlowOps"]-transpose_df["Capex"])/transpose_df["MarketCap"]
final_stats_df["ROC"]  = (transpose_df["EBITDA"] - transpose_df["D&A"])/(transpose_df["PPE"]+transpose_df["CurrAsset"]-transpose_df["CurrLiab"])
final_stats_df["BookToMkt"] = transpose_df["BookValue"]/transpose_df["MarketCap"]
final_stats_df["DivYield"] = transpose_df["DivYield"]



# finding value stocks based on Magic Formula
final_stats_val_df = final_stats_df.loc[stocks,:]
final_stats_val_df["CombRank"] = final_stats_val_df["EarningYield"].rank(ascending=False,na_option='bottom')+final_stats_val_df["ROC"].rank(ascending=False,na_option='bottom')
final_stats_val_df["MagicFormulaRank"] = final_stats_val_df["CombRank"].rank(method='first')
value_stocks = final_stats_val_df.sort_values("MagicFormulaRank").iloc[:,[2,4,8]]
print("------------------------------------------------")
print("Value stocks based on Greenblatt's Magic Formula")
print(value_stocks)


# finding highest dividend yield stocks
high_dividend_stocks = final_stats_df.sort_values("DivYield",ascending=False).iloc[:,6]
print("------------------------------------------------")
print("Highest dividend paying stocks")
print(high_dividend_stocks)


# # Magic Formula & Dividend yield combined
final_stats_df["CombRank"] = final_stats_df["EarningYield"].rank(ascending=False,method='first') \
                              +final_stats_df["ROC"].rank(ascending=False,method='first')  \
                              +final_stats_df["DivYield"].rank(ascending=False,method='first')
final_stats_df["CombinedRank"] = final_stats_df["CombRank"].rank(method='first')
value_high_div_stocks = final_stats_df.sort_values("CombinedRank").iloc[:,[2,4,6,8]]
print("------------------------------------------------")
print("Magic Formula and Dividend Yield combined")
print(value_high_div_stocks)
