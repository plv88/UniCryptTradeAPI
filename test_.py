import pandas as pd


class SPFinder():
    def __init__(self, df_klines, lst_result, i_kline=None):
        self.df = df_klines
        self.lst_result = lst_result
        self.i_kl = i_kline

    def find_structural_points(self):
        # Проверка наличия достаточного количества свечей
        if len(self.df) < 3:
            return

        # Проходим по свечам, начиная с третьей, чтобы оценить экстремумы
        for i in range(2, len(self.df)):
            prev_candle = self.df.iloc[i - 2]
            current_candle = self.df.iloc[i - 1]
            next_candle = self.df.iloc[i]

            if current_candle['high'] > prev_candle['high'] and current_candle['high'] > next_candle['high']:
                # Экстремум максимума - HH (High High)
                self.lst_result.append(('H', current_candle['time'], current_candle['high'], i))

            if current_candle['low'] < prev_candle['low'] and current_candle['low'] < next_candle['low']:
                # Экстремум минимума - LL (Low Low)
                self.lst_result.append(('L', current_candle['time'], current_candle['low'], i))

            if current_candle['high'] > prev_candle['high'] and current_candle['low'] < next_candle['low']:
                # Экстремум максимума и минимума (double) - d_HH
                self.lst_result.append(('d_HH', current_candle['time'], current_candle['high'], i))

            if current_candle['low'] < prev_candle['low'] and current_candle['high'] > next_candle['high']:
                # Экстремум минимума и максимума (double) - d_LL
                self.lst_result.append(('d_LL', current_candle['time'], current_candle['low'], i))

    def get_result(self):
        return self.lst_result


# Пример использования
if __name__ == "__main__":
    # Замените эту часть кода на загрузку вашего датафрейма из файла или другим способом
    data = {'time': [1698836399999, 1698847199999, 1698850799999, 1698897599999, 1698976799999, 1699156799999,
                     1699167599999, 1699174799999],
            'high': [1792.81, 1829.0, 1782.73, 1876.26, 1775.36, 1896.5, 1876.7, 1898.96],
            'low': [1791.0, 1826.12, 1782.72, 1869.1, 1773.89, 1892.33, 1875.7, 1894.72]}
    df_klines = pd.DataFrame(data)

    lst_result = []
    sp_finder = SPFinder(df_klines, lst_result)

    sp_finder.find_structural_points()

    result = sp_finder.get_result()
    print(result)
