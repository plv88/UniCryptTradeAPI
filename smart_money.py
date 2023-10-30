import pandas as pd
from binance import Binance_public
import numpy as np
import mplfinance as mpf
from multiprocessing import Process

"""
идеи
может сделать еще определение суб и майнор структруры внутри свинг структуры
"""
class SPFinder:
    """
    type_5_extremum:
    - n - normal
    - m - modified
    """

    def __init__(self, df_klines, lst_result, i_kline=None, type_extremum_5='m'):
        self.df = df_klines
        self.lst_result = lst_result
        self.i_kl = i_kline
        self.type_extremum_5 = type_extremum_5

    @classmethod
    def candles_match(cls, _prev, _curr, _next, type_k, _i, _klines):
        if _prev == _curr:
            if _i > 2:
                if _klines.iloc[_i - 2][type_k] == _curr:
                    if _i > 3:
                        if _klines.iloc[_i - 3][type_k] == _curr: _prev -= 1e-15
                        else: _prev = _klines.iloc[_i - 3][type_k]
                    else: _prev -= 1e-15
                else: _prev = _klines.iloc[_i - 2][type_k]
            else: _prev -= 1e-15
        if _curr == _next:
            if len(_klines) - (_i + 1) > 2:
                if _klines.iloc[_i + 2][type_k] == _curr:
                    if len(_klines) - (_i + 1) > 3:
                        if _klines.iloc[_i + 3][type_k] == _curr: _next += 1e-15
                        else: _next = _klines.iloc[_i + 3][type_k]
                    else: _next += 1e-15
                else: _next = _klines.iloc[_i + 2][type_k]
            else: _next += 1e-15
        return _prev, _curr, _next

    def find_swings(self):
        """
        t_HH - temp
        s_HH - strong если тестировали дискаунт или премиум маркете
        w_HH - weak
        b_HH - boss точка на которой произошел слом структуры
        c_HH - confirm
        d_HH - double тут по хорошему смотреть более младший таймфрейм и смотреть что там было
        hh - все тоже самое но для внутренней структуры
        """

        #Проверим тип свинга
        if len(self.df) == 3:
            prev_high, curr_high, next_high = self.df.iloc[0]['High'], self.df.iloc[1]['High'], self.df.iloc[2]['High']
            prev_low, curr_low, next_low = self.df.iloc[0]['Low'], self.df.iloc[1]['Low'], self.df.iloc[2]['Low']
            is_max = prev_high < curr_high > next_high
            is_min = prev_low > curr_low < next_low

        elif len(self.df) == 5:
            prev_high_1 = self.df.iloc[0]['High']
            prev_high_0 = self.df.iloc[1]['High']
            curr_high = self.df.iloc[2]['High']
            next_high_0 = self.df.iloc[3]['High']
            next_high_1 = self.df.iloc[4]['High']

            prev_low_1 = self.df.iloc[0]['Low']
            prev_low_0 = self.df.iloc[1]['Low']
            curr_low = self.df.iloc[2]['Low']
            next_low_0 = self.df.iloc[3]['Low']
            next_low_1 = self.df.iloc[4]['Low']

            if self.type_extremum_5 == 'n':
                is_max = prev_high_1 <= prev_high_0 < curr_high > next_high_0 >= next_high_1
                is_min = prev_low_1 >= prev_low_0 > curr_low < next_low_0 <= next_low_1
            elif self.type_extremum_5 == 'm':
                is_max = prev_high_1 < curr_high and prev_high_0 < curr_high > next_high_0 and curr_high > next_high_1
                is_min = prev_low_1 > curr_low and prev_low_0 > curr_low < next_low_0 and curr_low < next_low_1
            else:
                print(f'Error self.type_extremum_5 {self.type_extremum_5}')
                exit()

        else:
            print('Ошибка в длине len(self.df)')
            exit()

        is_dual_extr = True if is_max and is_min else False
        close_time = int(self.df.iloc[len(self.df)//2]['Close_time'])

        if len(self.lst_result) < 3:
            # Если одновременно и максимум и минимум то пропускаем
            if is_max and is_min and len(self.lst_result) == 0:
                print('Одновременно и максимум и минимум в самом начале работы')
                return []
            # Работаем с максимум
            elif is_max:
                if len(self.lst_result) == 0:
                    self.lst_result.append(('H', close_time, curr_high, self.i_kl))
                elif self.lst_result[-1][0] == 'H' and self.lst_result[-1][2] < curr_high:
                    self.lst_result[-1] = ('H', close_time, curr_high, self.i_kl)
                elif self.lst_result[-1][0] == 'L':
                    if len(self.lst_result) == 2:
                        if curr_high < self.lst_result[-2][2]:
                            self.lst_result.append(('t_LH', close_time, curr_high, self.i_kl))
                        elif curr_high > self.lst_result[-2][2]:
                            self.lst_result.append(('t_HH', close_time, curr_high, self.i_kl))
                    else:
                        self.lst_result.append(('H', close_time, curr_high, self.i_kl))
            # Работаем с минимум
            elif is_min:
                if len(self.lst_result) == 0:
                    self.lst_result.append(('L', close_time, curr_low, self.i_kl))
                elif self.lst_result[-1][0] == 'H':
                    if len(self.lst_result) == 2:
                        if curr_low < self.lst_result[-2][2]:
                            self.lst_result.append(('t_LL', close_time, curr_low, self.i_kl))
                        elif curr_low > self.lst_result[-2][2]:
                            self.lst_result.append(('t_HL', close_time, curr_low, self.i_kl))
                    else:
                        self.lst_result.append(('L', close_time, curr_low, self.i_kl))
                elif self.lst_result[-1][0] == 'L' and self.lst_result[-1][2] > curr_low:
                    self.lst_result[-1] = ('L', close_time, curr_low, self.i_kl)
        else:
            temp_swings = self.lst_result[-1][0].split('_')[1]

            if temp_swings == 'HH':
                # Cвеча is_dual_extr: приоритет low
                if is_min and self.lst_result[-2][2] > curr_low:
                    self.lst_result.append(('db_LL' if is_dual_extr else 'b_LL', close_time, curr_low, self.i_kl))
                elif is_min:
                    self.lst_result.append(('d_HL' if is_dual_extr else 't_HL', close_time, curr_low, self.i_kl))
                elif is_max and self.lst_result[-1][2] < curr_high:
                    self.lst_result[-1] = ('d_HH' if is_dual_extr else 't_HH', close_time, curr_high, self.i_kl)

            elif temp_swings == 'HL':
                # Cвеча is_dual_extr: приоритет low
                if is_min and self.lst_result[-3][2] > curr_low:
                    self.lst_result[-1] = ('db_LL' if is_dual_extr else 'b_LL', close_time, curr_low, self.i_kl)
                elif is_min and self.lst_result[-1][2] > curr_low:
                    self.lst_result[-1] = ('d_HL' if is_dual_extr else 't_HL', close_time, curr_low, self.i_kl)
                elif is_max and self.lst_result[-2][2] < curr_high:
                    self.lst_result.append(('d_HH' if is_dual_extr else 't_HH', close_time, curr_high, self.i_kl))

            elif temp_swings == 'LL':
                # Cвеча is_dual_extr: приоритет high
                if is_max and self.lst_result[-2][2] < curr_high:
                    self.lst_result.append(('db_HH' if is_dual_extr else 'b_HH', close_time, curr_high, self.i_kl))
                elif is_max:
                    self.lst_result.append(('d_LH' if is_dual_extr else 't_LH', close_time, curr_high, self.i_kl))
                elif is_min and self.lst_result[-1][2] > curr_low:
                    self.lst_result[-1] = ('d_LL' if is_dual_extr else 't_LL', close_time, curr_low, self.i_kl)

            elif temp_swings == 'LH':
                # Cвеча is_dual_extr: приоритет high
                if is_max and self.lst_result[-3][2] < curr_high:
                    self.lst_result[-1] = ('db_HH' if is_dual_extr else 'b_HH', close_time, curr_high, self.i_kl)
                elif is_max and self.lst_result[-1][2] < curr_high:
                    self.lst_result[-1] = ('d_LH' if is_dual_extr else 't_LH', close_time, curr_high, self.i_kl)
                elif is_min and self.lst_result[-2][2] > curr_low:
                    self.lst_result.append(('d_LL' if is_dual_extr else 't_LL', close_time, curr_low, self.i_kl))

            # Подумать как обрабатывать двойные экстремумы которые ломают структуру
            if self.lst_result[-1][0].split('_') == 'db':
                print('слом структуры свечкой с двойным экстремумы', '-' * 50)
                pass

        return self.lst_result



    def main_handler(self):
        return self.find_swings()


class LiqMonitor:
    """
    Тут будем анализировать ликвидность + имбаланс
    """
    def __init__(self, df_klines, lst_result=[]):
        self.df_klines = df_klines
        self.lst_result = lst_result

    def find_imbalance(self, threshold = 0.01):
        """
        max1 < min3 and max1 < max3 - бычий имбаланс
        min1 > max3 and min1 > min3 - медвежий имбаланс
        """
        if len(self.df_klines) != 3:
            raise ValueError("df_klines должен содержать только 3 строки")
        OHLC_1 = self.df_klines.iloc[0]
        OHLC_3 = self.df_klines.iloc[2]

        if OHLC_1['High'] < OHLC_3['Low'] and OHLC_1['High'] < OHLC_3['High']:
            print(f"бычий имбаланс {OHLC_3['Low'] - OHLC_1['High']}")
            pass
        elif OHLC_1['Low'] > OHLC_3['High'] and OHLC_1['Low'] > OHLC_3['Low']:
            print(f"медвежий имбаланс {OHLC_1['Low'] - OHLC_3['High']}")
            pass









def create_dataframe(klines):
    name_columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Asset_volume', 'n', 't_bb', 't_bq', 'ig']
    df = pd.DataFrame(klines, columns=name_columns)
    df = df[['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']]
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('Open_time', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    df['Close_time'] = df['Close_time'].astype(int)
    return df


b_pub = Binance_public('der')
work_coin = 'ethusdt'
work_interval = '1h'
klines = b_pub.get_kline(symbol_or_pair=work_coin, interval=work_interval, limit=100)
df_all_klines = create_dataframe(klines)

for i in range(2, len(klines) - 2):
    df_klines = create_dataframe(klines[i - 1: i + 2])
    LiqMonitor(df_klines=df_klines).find_imbalance()
pass


def plot_3point():
    list_result = []
    for i in range(2, len(klines) - 2):
        df_klines = create_dataframe(klines[i - 1: i + 2])
        list_result = SPFinder(df_klines=df_klines, lst_result=list_result, i_kline=i).main_handler()
    df = df_all_klines
    y_values_H = [np.nan] * len(df)
    y_values_L = [np.nan] * len(df)
    for el in list_result:
        if el[0] in ['H', 'L']:continue
        elif '_H' in el[0]: y_values_H[el[3]] = float(el[2])
        elif '_L' in el[0]: y_values_L[el[3]] = float(el[2])
    ap1 = mpf.make_addplot(y_values_H, scatter=True, markersize=15, marker='^', color='g')
    ap2 = mpf.make_addplot(y_values_L, scatter=True, markersize=15, marker='v', color='r')
    mpf.plot(df, type='candle', addplot=[ap1, ap2], title=f'3points {work_interval}')
# def plot_5point_n():
#     list_result = []
#     for i in range(2, len(klines) - 2):
#         work_klines = klines[i - 2: i + 3]
#         list_result = SPFinder(df_klines=work_klines, lst_result=list_result, i_kline=i, type_extremum_5='n').main_handler()
#     df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
#     df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
#     df['date'] = pd.to_datetime(df['date'], unit='ms')
#     df.set_index('date', inplace=True)
#     for col in ['open', 'high', 'low', 'close', 'volume']:
#         df[col] = df[col].astype(float)
#     y_values_H = [np.nan] * len(df)
#     y_values_L = [np.nan] * len(df)
#     for el in list_result:
#         if el[0] in ['H', 'L']:
#             continue
#         elif '_H' in el[0]:
#             y_values_H[el[3]] = float(el[2])
#         elif '_L' in el[0]:
#             y_values_L[el[3]] = float(el[2])
#     ap1 = mpf.make_addplot(y_values_H, scatter=True, markersize=15, marker='^', color='g')
#     ap2 = mpf.make_addplot(y_values_L, scatter=True, markersize=15, marker='v', color='r')
#     mpf.plot(df, type='candle', addplot=[ap1, ap2], title=f'5points_n {work_interval}')

# def plot_5point_m():
#     list_result = []
#     for i in range(2, len(klines) - 2):
#         work_klines = klines[i - 2: i + 3]
#         list_result = SPFinder(df_klines=work_klines, lst_result=list_result, i_kline=i, type_extremum_5='m').main_handler()
#     df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
#     df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
#     df['date'] = pd.to_datetime(df['date'], unit='ms')
#     df.set_index('date', inplace=True)
#     for col in ['open', 'high', 'low', 'close', 'volume']:
#         df[col] = df[col].astype(float)
#     y_values_H = [np.nan] * len(df)
#     y_values_L = [np.nan] * len(df)
#     for el in list_result:
#         if el[0] in ['H', 'L']:
#             continue
#         elif '_H' in el[0]:
#             y_values_H[el[3]] = float(el[2])
#         elif '_L' in el[0]:
#             y_values_L[el[3]] = float(el[2])
#     ap1 = mpf.make_addplot(y_values_H, scatter=True, markersize=15, marker='^', color='g')
#     ap2 = mpf.make_addplot(y_values_L, scatter=True, markersize=15, marker='v', color='r')
#     mpf.plot(df, type='candle', addplot=[ap1, ap2], title=f'5points_m {work_interval}')

if __name__ == "__main__":
    p1 = Process(target=plot_3point)
    # p2 = Process(target=plot_5point_m)
    # p3 = Process(target=plot_5point_n)
    # p4 = Process(target=plot_graph4)

    p1.start()
    # p2.start()
    # p3.start()
    # p4.start()

    p1.join()
    # p2.join()
    # p3.join()
    # p4.join()

# list_result = []
# list_result_2 = []
# for i in range(2, len(klines) - 2):
#     work_klines = klines[i-1: i+2]
#     list_result = SPFinder(klines=work_klines, lst_result=list_result, i_kline=i).main_handler()
#
# for i in range(2, len(klines) - 2):
#     work_klines = klines[i-2: i+3]
#     list_result_2 = SPFinder(klines=work_klines, lst_result=list_result_2, i_kline=i).main_handler()
#
# df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
# df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
# df['unix_date'] = df['date'].astype(int) / 1000 if len(str(int(df.iloc[0]['date']))) == 13 else df['date'].astype(int)
# df['date'] = pd.to_datetime(df['date'], unit='ms')
# df.set_index('date', inplace=True)
# for col in ['open', 'high', 'low', 'close', 'volume']:
#     df[col] = df[col].astype(float)
# y_values_H_3 = [np.nan] * len(df)
# y_values_L_3 = [np.nan] * len(df)
# y_values_H_5 = [np.nan] * len(df)
# y_values_L_5 = [np.nan] * len(df)
#
# for el in list_result:
#     if el[0] in ['H', 'L']: continue
#     elif '_H' in el[0]:
#         y_values_H_3[el[3]] = float(el[2])
#     elif '_L' in el[0]:
#         y_values_L_3[el[3]] = float(el[2])
#
# for el in list_result_2:
#     if el[0] in ['H', 'L']: continue
#     elif '_H' in el[0]:
#         y_values_H_5[el[3]] = float(el[2])
#     elif '_L' in el[0]:
#         y_values_L_5[el[3]] = float(el[2])
# ap1 = mpf.make_addplot(y_values_H_3, scatter=True, markersize=15, marker='^', color='g')
# ap2 = mpf.make_addplot(y_values_L_3, scatter=True, markersize=15, marker='v', color='r')
# ap3 = mpf.make_addplot(y_values_H_5, scatter=True, markersize=15, marker='*', color='b')
# ap4 = mpf.make_addplot(y_values_L_5, scatter=True, markersize=15, marker='o', color='m')
# mpf.plot(df, type='candle', addplot=[ap1, ap2, ap3, ap4], title=work_interval)






#По 4 точкам можно определить тренд HL-HH-HL-HH а лучше показывать числом сколько произошло после смены


# dict_result = {}
#
# for work_interval in ["3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]:
#     for coint_k in range(500, 1550, 50):
#         klines = b_pub.get_kline(symbol_or_pair=work_coin, interval=work_interval, limit=coint_k)
#         if len(klines) != coint_k:
#             print('кол-во свечек урезано')
#             pass
#         df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
#         df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
#         df['unix_date'] = df['date'].astype(int) / 1000 if len(str(int(df.iloc[0]['date']))) == 13 else df['date'].astype(int)
#         df['unix_date'] = df['unix_date'].astype(int)
#         df['date'] = pd.to_datetime(df['date'], unit='ms')
#         df.set_index('date', inplace=True)
#         for col in ['open', 'high', 'low', 'close', 'volume']:
#             df[col] = df[col].astype(float)
#         res_find_swings = find_swings(df)
#
#         t_res = ''.join([el[0].split('_')[1] for el in res_find_swings if '_' in el[0]])
#         if 'LLLL' in t_res:
#             pass
#
#         dict_result[coint_k] = ','.join([el[0] for el in res_find_swings[-10:]])
#
#         print(f'{work_interval}: {coint_k}')
#
#     old_val = None
#     for _coint, _val in dict_result.items():
#         if old_val:
#             if old_val != _val:
#                 print('Не равны')
#                 pass
#             else:
#                 print(f'{old_val} == {_val}')
#             old_val = _val
#         else:
#             old_val = _val

# print('OK')
#
# exit()

def plot_graph1(w_inter="15m"):
    klines = b_pub.get_kline(symbol_or_pair=work_coin, interval=w_inter, limit=320)
    df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    # df['unix_date'] = df['date']
    df['unix_date'] = df['date'].astype(int) / 1000 if len(str(int(df.iloc[0]['date']))) == 13 else df['date'].astype(int)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    y_values_H = [np.nan] * len(df)
    y_values_L = [np.nan] * len(df)
    # res_find_swings = find_swings(df)
    for el in [1,2]:
        if el[0] in ['H', 'L']: continue
        elif '_H' in el[0]: y_values_H[el[1]] = float(el[2])
        elif '_L' in el[0]: y_values_L[el[1]] = float(el[2])
    ap1 = mpf.make_addplot(y_values_H, scatter=True, markersize=15, marker='^', color='g')
    ap2 = mpf.make_addplot(y_values_L, scatter=True, markersize=15, marker='v', color='r')
    mpf.plot(df, type='candle', addplot=[ap1, ap2], title=w_inter)


