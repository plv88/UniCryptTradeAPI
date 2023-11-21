import pandas as pd
from binance import Binance_public
import numpy as np
import mplfinance as mpf
from multiprocessing import Process
# import datetime
import time
from datetime import datetime

"""
идеи
может сделать еще определение суб и майнор структруры внутри свинг структуры
"""


class UniOther:
    @staticmethod
    def create_dataframe(klines):
        name_columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Asset_volume', 'n', 't_bb', 't_bq', 'ig']
        df = pd.DataFrame(klines, columns=name_columns)
        df = df[['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time']]
        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
        df.set_index('Open_time', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        # for col in ['Open_time', 'Close_time']:
        #     df[col] = df[col].astype(int)
        df['Close_time'] = df['Close_time'].astype(int)


        return df

class SPFinder(UniOther):
    def __init__(self):
        pass

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

    def find_swings(self, w_df, lst_result, type_extr_5='m', i_kl=None):
        """
        t_HH - temp
        s_HH - strong если тетировали дискаунт или премиум маркете
        w_HH - weak
        b_HH - boss точка на которой произошел слом структуры
        c_HH - confirm
        d_HH - double тут по хорошему смотреть более младший таймфрейм и смотреть что там было
        hh - все тоже самое но для внутренней структуры

        type_5_extremum:
        - n - normal
        - m - modified
        """

        #Проверим тип свинга
        if len(w_df) == 3:
            prev_high, curr_high, next_high = w_df.iloc[0]['High'], w_df.iloc[1]['High'], w_df.iloc[2]['High']
            prev_low, curr_low, next_low = w_df.iloc[0]['Low'], w_df.iloc[1]['Low'], w_df.iloc[2]['Low']
            is_max = prev_high < curr_high > next_high
            is_min = prev_low > curr_low < next_low

        elif len(w_df) == 5:
            prev_high_1 = w_df.iloc[0]['High']
            prev_high_0 = w_df.iloc[1]['High']
            curr_high = w_df.iloc[2]['High']
            next_high_0 = w_df.iloc[3]['High']
            next_high_1 = w_df.iloc[4]['High']

            prev_low_1 = w_df.iloc[0]['Low']
            prev_low_0 = w_df.iloc[1]['Low']
            curr_low = w_df.iloc[2]['Low']
            next_low_0 = w_df.iloc[3]['Low']
            next_low_1 = w_df.iloc[4]['Low']

            if type_extr_5 == 'n':
                is_max = prev_high_1 <= prev_high_0 < curr_high > next_high_0 >= next_high_1
                is_min = prev_low_1 >= prev_low_0 > curr_low < next_low_0 <= next_low_1
            elif type_extr_5 == 'm':
                is_max = prev_high_1 < curr_high and prev_high_0 < curr_high > next_high_0 and curr_high > next_high_1
                is_min = prev_low_1 > curr_low and prev_low_0 > curr_low < next_low_0 and curr_low < next_low_1
            else:
                mes_to_log = f'Error type_extr_5 {type_extr_5}'
                raise TypeError(mes_to_log)

        else:
            mes_to_log = 'Ошибка в длине len(w_df)'
            raise TypeError(mes_to_log)

        is_dual_extr = True if is_max and is_min else False
        close_time = int(w_df.iloc[len(w_df)//2]['Close_time'])

        if len(lst_result) < 3:
            # Если одновременно и максимум и минимум то пропускаем
            if is_max and is_min and len(lst_result) == 0:
                print('Одновременно и максимум и минимум в самом начале работы')
                return []
            # Работаем с максимум
            elif is_max:
                if len(lst_result) == 0:
                    lst_result.append(('H', close_time, curr_high, i_kl))
                elif lst_result[-1][0] == 'H' and lst_result[-1][2] < curr_high:
                    lst_result[-1] = ('H', close_time, curr_high, i_kl)
                elif lst_result[-1][0] == 'L':
                    if len(lst_result) == 2:
                        if curr_high < lst_result[-2][2]:
                            lst_result.append(('t_LH', close_time, curr_high, i_kl))
                        elif curr_high > lst_result[-2][2]:
                            lst_result.append(('t_HH', close_time, curr_high, i_kl))
                    else:
                        lst_result.append(('H', close_time, curr_high, i_kl))
            # Работаем с минимум
            elif is_min:
                if len(lst_result) == 0:
                    lst_result.append(('L', close_time, curr_low, i_kl))
                elif lst_result[-1][0] == 'H':
                    if len(lst_result) == 2:
                        if curr_low < lst_result[-2][2]:
                            lst_result.append(('t_LL', close_time, curr_low, i_kl))
                        elif curr_low > lst_result[-2][2]:
                            lst_result.append(('t_HL', close_time, curr_low, i_kl))
                    else:
                        lst_result.append(('L', close_time, curr_low, i_kl))
                elif lst_result[-1][0] == 'L' and lst_result[-1][2] > curr_low:
                    lst_result[-1] = ('L', close_time, curr_low, i_kl)
        else:
            temp_swings = lst_result[-1][0].split('_')[1]

            if temp_swings == 'HH':
                # Cвеча is_dual_extr: приоритет low
                if is_min and lst_result[-2][2] > curr_low:
                    lst_result.append(('db_LL' if is_dual_extr else 'b_LL', close_time, curr_low, i_kl))
                elif is_min:
                    lst_result.append(('d_HL' if is_dual_extr else 't_HL', close_time, curr_low, i_kl))
                elif is_max and lst_result[-1][2] < curr_high:
                    lst_result[-1] = ('d_HH' if is_dual_extr else 't_HH', close_time, curr_high, i_kl)

            elif temp_swings == 'HL':
                # Cвеча is_dual_extr: приоритет low
                if is_min and lst_result[-3][2] > curr_low:
                    lst_result[-1] = ('db_LL' if is_dual_extr else 'b_LL', close_time, curr_low, i_kl)
                elif is_min and lst_result[-1][2] > curr_low:
                    lst_result[-1] = ('d_HL' if is_dual_extr else 't_HL', close_time, curr_low, i_kl)
                elif is_max and lst_result[-2][2] < curr_high:
                    lst_result.append(('d_HH' if is_dual_extr else 't_HH', close_time, curr_high, i_kl))

            elif temp_swings == 'LL':
                # Cвеча is_dual_extr: приоритет high
                if is_max and lst_result[-2][2] < curr_high:
                    lst_result.append(('db_HH' if is_dual_extr else 'b_HH', close_time, curr_high, i_kl))
                elif is_max:
                    lst_result.append(('d_LH' if is_dual_extr else 't_LH', close_time, curr_high, i_kl))
                elif is_min and lst_result[-1][2] > curr_low:
                    lst_result[-1] = ('d_LL' if is_dual_extr else 't_LL', close_time, curr_low, i_kl)

            elif temp_swings == 'LH':
                # Cвеча is_dual_extr: приоритет high
                if is_max and lst_result[-3][2] < curr_high:
                    lst_result[-1] = ('db_HH' if is_dual_extr else 'b_HH', close_time, curr_high, i_kl)
                elif is_max and lst_result[-1][2] < curr_high:
                    lst_result[-1] = ('d_LH' if is_dual_extr else 't_LH', close_time, curr_high, i_kl)
                elif is_min and lst_result[-2][2] > curr_low:
                    lst_result.append(('d_LL' if is_dual_extr else 't_LL', close_time, curr_low, i_kl))

            # Подумать как обрабатывать двойные экстремумы которые ломают структуру
            if lst_result[-1][0].split('_') == 'db':
                print('слом структуры свечкой с двойным экстремумы', '-' * 50)
                pass

        return lst_result


    def history_handler_binance(self, asset, time_frame, date_mes=None, n_klines=400):
        list_result = []
        if date_mes:
            date_mes = int(datetime.strptime(date_mes, '%Y-%m-%d %H:%M:%S').timestamp()*1000)

        klines = Binance_public('der').get_kline(symbol_or_pair=asset,
                                                 interval=time_frame,
                                                 endTime=date_mes,
                                                 limit=n_klines)
        if klines[-1][6] > int(time.time()) if len(str(int(time.time()))) == 13 else int(time.time()) * 1000:
            del klines[-1]
        for j in range(2, len(klines) - 2):
            df_kline = self.create_dataframe(klines[j - 1: j + 2])
            list_result = SPFinder().find_swings(df_kline, list_result, j)
        return list_result

    def calculate_pattern(self, _w_lst):
        """
        HL HH HL HH - 'uptrend'
        LL LH LL LH 'downtrend'
        OTHER 'consolidation'
        """
        temp_lst = [el[0].split('_')[1] for el in _w_lst if '_' in el[0]]
        _w_trend = ' '.join(temp_lst[-4:])
        match _w_trend:
            case 'HH HL HH HL':
                return 'uptrend'
            case 'HL HH HL HH':
                return 'uptrend'
            case 'LH LL LH LL':
                return 'downtrend'
            case 'LL LH LL LH':
                return 'downtrend'
        return 'consolidation'


    def identify_trend_pattern(self, asset, time_frame, date_mes, n_klines=400):
        work_list = self.history_handler_binance(asset, time_frame, date_mes, n_klines)
        trend_pattern = self.calculate_pattern(work_list)
        return trend_pattern



    # def main_handler(self, w_df, w_lst=[], i_kl=None):
    #     w_lst = self.find_swings(w_df, w_lst, i_kl)
    #     return w_lst



class ImbalanceKlineChecker(UniOther):
    """

    """
    WORK_INTERVAL = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"]
    def __init__(self, symbol, analysis_period=500):
        self.symbol = symbol
        self.analysis_period = analysis_period
        self.dict_resalt = self.load_history_date()



    def load_history_date(self):
        """
        """
        dict_resalt = {}
        b_pub = Binance_public('der')
        for temp_interval in self.WORK_INTERVAL:
            lst_imbalance = []
            klines = b_pub.get_kline(symbol_or_pair=self.symbol, interval=temp_interval, limit=self.analysis_period)
            #Проверяем последнюю свечку, не закрыла
            time_now = int(time.time()) if len(str(int(time.time()))) == 13 else int(time.time())*1000
            if klines[-1][6] > time_now:
                del klines[-1]
            for i in range(1, len(klines) - 1):
                temp_df = self.create_dataframe(klines[i - 1: i + 2])
                w_kline = temp_df.iloc[1]
                lst_imbalance = self.check_imbalance(w_kline, lst_imbalance, del_close_imbalance=True)
                tuple_imbalance_klines = self.find_imbalance_klines(temp_df)
                if tuple_imbalance_klines:
                    lst_imbalance.append(tuple_imbalance_klines)
            dict_resalt[temp_interval] = lst_imbalance

        for el in dict_resalt['4h']:
            print(f"{datetime.datetime.utcfromtimestamp(el[0]/1000).strftime('%Y-%m-%d %H:%M:%S')}", el)
        return dict_resalt

    @classmethod
    def check_imbalance(cls, w_kline, lst_imbalance, del_close_imbalance=True):
        for j in range(len(lst_imbalance)):
            close_time, type_imbalance, imbalance_open, imbalance_close, overlapped_by_50 = lst_imbalance[j]
            delta_imbalance = abs(imbalance_open - imbalance_close)
            if type_imbalance == 'UpShift':
                # Тут мы можем перекрыть минимумом
                if w_kline['Low'] < imbalance_open + delta_imbalance * 0.1:
                    lst_imbalance[j] = (close_time, type_imbalance, imbalance_open, imbalance_close, 'close')
                elif w_kline['Low'] < imbalance_open + delta_imbalance * 0.55:
                    lst_imbalance[j] = (close_time, type_imbalance, imbalance_open, imbalance_close, w_kline['Low'])
            elif type_imbalance == 'DownShift':
                # Тут мы можем перекрыть максимум
                if w_kline['High'] > imbalance_open + delta_imbalance * 0.1:
                    lst_imbalance[j] = (close_time, type_imbalance, imbalance_open, imbalance_close, 'close')
                elif w_kline['High'] > imbalance_close + (delta_imbalance * 0.45):
                    lst_imbalance[j] = (close_time, type_imbalance, imbalance_open, imbalance_close, w_kline['Low'])
        if del_close_imbalance:
            lst_imbalance = [_tup for _tup in lst_imbalance if _tup[4] != 'close']
        return lst_imbalance

    def find_imbalance_klines(self, df_klines, threshold = 0.1):
        """
        max1 < min3 and max1 < max3 - бычий имбаланс UpShift
        min1 > max3 and min1 > min3 - медвежий имбаланс DownShift
        return type_imbalance (close_time, imbalance_open, imbalance_close, overlapped_by_50)
        """
        if len(df_klines) != 3:
            raise ValueError("df_klines должен содержать только 3 строки")
        OHLC_1, OHLC_2, OHLC_3 = df_klines.iloc[0], df_klines.iloc[1], df_klines.iloc[2]
        max_delta = min([OHLC_1['High']-OHLC_1['Low'], OHLC_2['High']-OHLC_2['Low'], OHLC_3['High']-OHLC_3['Low']])

        if OHLC_1['High'] < OHLC_3['Low'] and OHLC_1['High'] < OHLC_3['High']:
            # print(f"бычий имбаланс {OHLC_3['Low'] - OHLC_1['High']}")
            if OHLC_3['Low'] - OHLC_1['High'] > max_delta * threshold:
                return (OHLC_2['Close_time'], 'UpShift', OHLC_1['High'], OHLC_3['Low'], False)
        elif OHLC_1['Low'] > OHLC_3['High'] and OHLC_1['Low'] > OHLC_3['Low']:
            # print(f"медвежий имбаланс {OHLC_1['Low'] - OHLC_3['High']}")
            if OHLC_1['Low'] - OHLC_3['High'] > max_delta * threshold:
                return (OHLC_2['Close_time'], 'DownShift', OHLC_1['Low'], OHLC_3['High'], False)

    def main(self, work_klines):
        print('Начинает работать майн на вебсокетах, пока пасс и exit')
        pass
        exit()
        temp_df = self.create_dataframe(work_klines)
        w_kline = temp_df.iloc[1]
        lst_imbalance = self.dict_resalt['timeframe']
        lst_imbalance = self.check_imbalance(w_kline, lst_imbalance, del_close_imbalance=True)

class LiqMonitorKlines:
    """
    - 1.0 Уровни открытия/закрытия прошлого дня/недели/месяца
    - 2.1 EQL EQH
    - 2.2 Компрессия
    - 2.3 Отдельные максимумы и минимумы

    Подумать деление на Внешняя и внутреннюю ликвидность (BSL SSL)

    """
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


# asset = 'ethusdt'
# time_frame = '1h'
# for el in ['02', '05', '07', '09', '13', '18']:
#     date_mes = f'2023-02-{el} 12:00:00'
#     trend_pattern = SPFinder().identify_trend_pattern(asset, time_frame, date_mes)
#     print(trend_pattern)
#
# exit()
#
# test = SPFinder()
# lst_result = test.history_handler_binance('ethusdt', '1h', '2023-02-15 12:00:00')
# f_pat = test.identify_trend_pattern(lst_result)



# b_pub = Binance_public('der')
# work_coin = 'ethusdt'
# work_interval = '1h'
# klines = b_pub.get_kline(symbol_or_pair=work_coin, interval=work_interval, limit=100)
# df_all_klines = create_dataframe(klines)
# name_dir = rf"klines.bin"
# with open(name_dir, "wb") as file:
#     pickle.dump(df_all_klines, file)

# _lst_result = []
#
# for i in range(2, len(klines) - 2):
#     df_klines = create_dataframe(klines[i - 1: i + 2])
#     _lst_result = SPFinder(df_klines, _lst_result, i).main_handler()
# #     LiqMonitor(df_klines=df_klines).find_imbalance()
# pass



# Пример использования
# candles = [(time, open, close, high, low), ...]
# structure_points = find_structure_points(candles)










# def plot_3point():
#     list_result = []
#     for i in range(2, len(klines) - 2):
#         df_klines = create_dataframe(klines[i - 1: i + 2])
#         list_result = SPFinder(df_klines=df_klines, lst_result=list_result, i_kline=i).main_handler()
#     df = ''
#     y_values_H = [np.nan] * len(df)
#     y_values_L = [np.nan] * len(df)
#     for el in list_result:
#         if el[0] in ['H', 'L']:continue
#         elif '_H' in el[0]: y_values_H[el[3]] = float(el[2])
#         elif '_L' in el[0]: y_values_L[el[3]] = float(el[2])
#     ap1 = mpf.make_addplot(y_values_H, scatter=True, markersize=15, marker='^', color='g')
#     ap2 = mpf.make_addplot(y_values_L, scatter=True, markersize=15, marker='v', color='r')
#     mpf.plot(df, type='candle', addplot=[ap1, ap2], title=f'3points {work_interval}')
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

# if __name__ == "__main__":
#     p1 = Process(target=plot_3point)
    # p2 = Process(target=plot_5point_m)
    # p3 = Process(target=plot_5point_n)
    # p4 = Process(target=plot_graph4)

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()

    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
#
# # list_result = []
# # list_result_2 = []
# # for i in range(2, len(klines) - 2):
# #     work_klines = klines[i-1: i+2]
# #     list_result = SPFinder(klines=work_klines, lst_result=list_result, i_kline=i).main_handler()
# #
# # for i in range(2, len(klines) - 2):
# #     work_klines = klines[i-2: i+3]
# #     list_result_2 = SPFinder(klines=work_klines, lst_result=list_result_2, i_kline=i).main_handler()
# #
# # df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
# # df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
# # df['unix_date'] = df['date'].astype(int) / 1000 if len(str(int(df.iloc[0]['date']))) == 13 else df['date'].astype(int)
# # df['date'] = pd.to_datetime(df['date'], unit='ms')
# # df.set_index('date', inplace=True)
# # for col in ['open', 'high', 'low', 'close', 'volume']:
# #     df[col] = df[col].astype(float)
# # y_values_H_3 = [np.nan] * len(df)
# # y_values_L_3 = [np.nan] * len(df)
# # y_values_H_5 = [np.nan] * len(df)
# # y_values_L_5 = [np.nan] * len(df)
# #
# # for el in list_result:
# #     if el[0] in ['H', 'L']: continue
# #     elif '_H' in el[0]:
# #         y_values_H_3[el[3]] = float(el[2])
# #     elif '_L' in el[0]:
# #         y_values_L_3[el[3]] = float(el[2])
# #
# # for el in list_result_2:
# #     if el[0] in ['H', 'L']: continue
# #     elif '_H' in el[0]:
# #         y_values_H_5[el[3]] = float(el[2])
# #     elif '_L' in el[0]:
# #         y_values_L_5[el[3]] = float(el[2])
# # ap1 = mpf.make_addplot(y_values_H_3, scatter=True, markersize=15, marker='^', color='g')
# # ap2 = mpf.make_addplot(y_values_L_3, scatter=True, markersize=15, marker='v', color='r')
# # ap3 = mpf.make_addplot(y_values_H_5, scatter=True, markersize=15, marker='*', color='b')
# # ap4 = mpf.make_addplot(y_values_L_5, scatter=True, markersize=15, marker='o', color='m')
# # mpf.plot(df, type='candle', addplot=[ap1, ap2, ap3, ap4], title=work_interval)
#
#
#
#
#
#
# #По 4 точкам можно определить тренд HL-HH-HL-HH а лучше показывать числом сколько произошло после смены
#
#
# # dict_result = {}
# #
# # for work_interval in ["3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]:
# #     for coint_k in range(500, 1550, 50):
# #         klines = b_pub.get_kline(symbol_or_pair=work_coin, interval=work_interval, limit=coint_k)
# #         if len(klines) != coint_k:
# #             print('кол-во свечек урезано')
# #             pass
# #         df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
# #         df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
# #         df['unix_date'] = df['date'].astype(int) / 1000 if len(str(int(df.iloc[0]['date']))) == 13 else df['date'].astype(int)
# #         df['unix_date'] = df['unix_date'].astype(int)
# #         df['date'] = pd.to_datetime(df['date'], unit='ms')
# #         df.set_index('date', inplace=True)
# #         for col in ['open', 'high', 'low', 'close', 'volume']:
# #             df[col] = df[col].astype(float)
# #         res_find_swings = find_swings(df)
# #
# #         t_res = ''.join([el[0].split('_')[1] for el in res_find_swings if '_' in el[0]])
# #         if 'LLLL' in t_res:
# #             pass
# #
# #         dict_result[coint_k] = ','.join([el[0] for el in res_find_swings[-10:]])
# #
# #         print(f'{work_interval}: {coint_k}')
# #
# #     old_val = None
# #     for _coint, _val in dict_result.items():
# #         if old_val:
# #             if old_val != _val:
# #                 print('Не равны')
# #                 pass
# #             else:
# #                 print(f'{old_val} == {_val}')
# #             old_val = _val
# #         else:
# #             old_val = _val
#
# # print('OK')
# #
# # exit()
#
# def plot_graph1(w_inter="15m"):
#     klines = b_pub.get_kline(symbol_or_pair=work_coin, interval=w_inter, limit=320)
#     df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume', '_c', '_v', '_t', '_b', '_q', '_g'])
#     df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
#     # df['unix_date'] = df['date']
#     df['unix_date'] = df['date'].astype(int) / 1000 if len(str(int(df.iloc[0]['date']))) == 13 else df['date'].astype(int)
#     df['date'] = pd.to_datetime(df['date'], unit='ms')
#     df.set_index('date', inplace=True)
#     for col in ['open', 'high', 'low', 'close', 'volume']:
#         df[col] = df[col].astype(float)
#     y_values_H = [np.nan] * len(df)
#     y_values_L = [np.nan] * len(df)
#     # res_find_swings = find_swings(df)
#     for el in [1,2]:
#         if el[0] in ['H', 'L']: continue
#         elif '_H' in el[0]: y_values_H[el[1]] = float(el[2])
#         elif '_L' in el[0]: y_values_L[el[1]] = float(el[2])
#     ap1 = mpf.make_addplot(y_values_H, scatter=True, markersize=15, marker='^', color='g')
#     ap2 = mpf.make_addplot(y_values_L, scatter=True, markersize=15, marker='v', color='r')
#     mpf.plot(df, type='candle', addplot=[ap1, ap2], title=w_inter)


