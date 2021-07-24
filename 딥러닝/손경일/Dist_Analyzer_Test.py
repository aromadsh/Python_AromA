import Dist_Analyzer as Analyzer 
import pandas as pd
import pymysql
import re
import json

try:
    with open('name_code.json', 'r') as in_file:
        name_code = json.load(in_file)
except FileNotFoundError:
    from Crawling_TimeSeries_Class import name_code

mk = Analyzer.MarketDB()
stocks = ['005930', 'SK하이닉스', '현대자동차', 'NAVER']
df = pd.DataFrame()

for s in stocks:
    if len(re.findall('[0-9]', s)) == 6:
        df[name_code[s]] = mk.get_daily_price(s, start_date= '2017-12-24', end_date = '2021-06-14')['close']
    else:
        df[s] = mk.get_daily_price(s, start_date= '2017-12-24', end_date = '2021-06-14')['close']
print(df)