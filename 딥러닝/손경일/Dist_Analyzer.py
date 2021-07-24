import pymysql
from datetime import datetime
import pandas as pd
import re
import json

try:
    with open('name_code.json', 'r') as in_file:
        name_code = json.load(in_file)
        name_code = {v:k for (k,v) in name_code.items()}
except FileNotFoundError:
    from Crawling_TimeSeries_Class import name_code
    name_code = {v:k for (k,v) in name_code.items()}

class MarketDB:
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', user='root',
                                    password='1234', db='stock_data', charset='utf8')      
    def get_daily_price(self, code, start_date = None, end_date = None):
        with self.conn.cursor() as curs:
            if len(re.findall('[0-9]', code)) == 6:
                sql = f'''
                select * from daily_price as dp
                where dp.code = '{code}' and dp.date between '{start_date}' and '{end_date}';
                '''
            else:
                sql = '''
                select * from daily_price as dp
                where dp.code = '{}' and dp.date between '{}' and '{}';
                '''.format(name_code['{}'.format(code)], start_date, end_date)               
            curs.execute(sql)
            resultset1 = curs.fetchall()
            daily_df = pd.DataFrame(resultset1, columns = ['code', 'date', 'open', 'high', 'low', 'close', 'diff', 'volume'])
        self.conn.commit()
        return daily_df