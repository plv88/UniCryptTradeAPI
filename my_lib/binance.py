import requests
import sys
from datetime import datetime
import json
import traceback
import time
import websocket
from my_lib.logger import Logger






class Binance_public:

    __main_url_spot = r'https://api.binance.com'
    __main_url_derv = r'https://fapi.binance.com'

    def __init__(self, market_type):
        self.market_type = market_type
        self.__lg = Logger('Binance_public', type_log='w')
        self.__logger = self.__lg.logger


    def __setattr__(self, key, value):
        if key == 'market_type' and value not in ['spot', 'der']:
            self.__logger.error(f'Неизвестный тип рынка {self.market_type}')
            raise TypeError(f"Неверный market_type {self.market_type}")
            sys.exit()
        object.__setattr__(self, key, value)


    def __request_template(self, end_point, par=None, method='get'):
        work_link = self.__main_url_spot if self.market_type == 'spot' else self.__main_url_derv
        match method.lower():
            case 'get':
                req = requests.get(work_link + end_point, params=par)
            case 'post':
                req = requests.post(work_link + end_point, params=par)
            case 'delete':
                req = requests.delete(work_link + end_point, params=par)
            case _:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} Неизвестный метод {method}'
                print(mes_to_log)
                self.__logger.error(mes_to_log)
                sys.exit()
                return None
        if req.ok:
            return req.json()

    @classmethod
    def check_startTime_endTime(cls, def_name, startTime, endTime):
        startTime = int(startTime) if startTime else startTime
        endTime = int(endTime) if endTime else endTime
        for i, temp_time in enumerate([startTime, endTime]):
            if temp_time:
                if len(str(temp_time)) == 10:
                    temp_time = int(temp_time * 1000)
                elif len(str(temp_time)) != 13:
                    mes_to_log = f"{def_name} ошибка в {'startTime' if i == 0 else 'endTime'} {temp_time}"
                    cls.__logger.error(mes_to_log)
                    sys.exit()
                if i == 0: startTime = temp_time
                if i == 1: endTime = temp_time
        return startTime, endTime

    def get_exchange_information(self):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#exchange-information
        derv: https://binance-docs.github.io/apidocs/futures/en/#exchange-information
        """
        end_point = '/api/v3/exchangeInfo' if self.market_type == 'spot' else '/fapi/v1/exchangeInfo'
        return self.__request_template(end_point=end_point, method='get')



    def get_order_book(self, symbol, limit=100):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#order-book
        derv: https://binance-docs.github.io/apidocs/futures/en/#order-book
        """
        if self.market_type == 'der' and limit not in [5, 10, 20, 50, 100, 500, 1000]:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} Неверная глубина книги дериватив {limit}'
            self.__logger.error(mes_to_log)
            sys.exit()
        end_point = '/api/v3/depth' if self.market_type == 'spot' else '/fapi/v1/depth'
        par = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_recent_trades_list(self, symbol, limit=500):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#recent-trades-list
        derv: https://binance-docs.github.io/apidocs/futures/en/#recent-trades-list
        """
        end_point = '/api/v3/trades' if self.market_type == 'spot' else '/fapi/v1/trades'
        par = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_old_trade_lookup(self, symbol, fromId=None, limit=500):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#old-trade-lookup
        derv: https://binance-docs.github.io/apidocs/futures/en/#old-trades-lookup-market_data
        """
        end_point = '/api/v3/historicalTrades' if self.market_type == 'spot' else '/fapi/v1/historicalTrades'
        par = {
            'symbol': symbol.upper(),
            'limit': limit,
            'fromId': fromId
        }
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_aggregate_trades_list(self, symbol, fromId=None, startTime=None, endTime=None, limit=500):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#compressed-aggregate-trades-list
        derv: https://binance-docs.github.io/apidocs/futures/en/#compressed-aggregate-trades-list
        Агрегируются трейды (возможно только мелкие) одного пользователя до 100мс
        """
        end_point = '/api/v3/aggTrades' if self.market_type == 'spot' else '/fapi/v1/aggTrades'
        par = {
            'symbol': symbol.upper(),
            'fromId': fromId,
            'startTime': startTime,
            'endTime': endTime,
            'limit': limit
        }
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_kline(self, symbol_or_pair, interval, startTime=None, endTime=None, contractType='PERPETUAL', limit=500):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        derv: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
        market_type: spot, der, contract, index, mark_price, premium_index
        contractType: ['PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER']
        """
        lst_kline_type = ['spot', 'der', 'contract', 'index', 'mark_price', 'premium_index']
        lst_contract_type = ['PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER']
        lst_work_interval = ["1s","1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]

        if self.market_type.lower() not in lst_kline_type:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} Неизвестный kline_type {self.market_type}, lst_kline_type: {lst_kline_type}'
            self.__logger.error(mes_to_log)
            sys.exit()
        if contractType.upper() not in lst_contract_type:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} Неизвестный contractType {contractType}, only {",".join(lst_contract_type)}'
            self.__logger.error(mes_to_log)
            sys.exit()
        if limit < 0 and ((self.market_type != 'spot' and abs(limit) < 1500) or (self.market_type == 'spot' and abs(limit) < 1000)):
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный limit {limit}'
            self.__logger.error(mes_to_log)
            sys.exit()
        name_symbol_or_pair = 'symbol'
        match self.market_type.lower():
            case 'spot':
                end_point = '/api/v3/klines'
            case 'der':
                end_point = '/fapi/v1/klines'
            case 'contract':
                end_point = '/fapi/v1/continuousKlines'
                name_symbol_or_pair = 'pair'
            case 'index':
                end_point = '/fapi/v1/indexPriceKlines'
                name_symbol_or_pair = 'pair'
            case 'mark_price':
                end_point = '/fapi/v1/markPriceKlines'
            case 'premium_index':
                end_point = '/fapi/v1/premiumIndexKlines'
        if interval not in lst_work_interval:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} Неизвестный интервал {interval}, work_interval: {lst_work_interval}'
            self.__logger.error(mes_to_log)
            sys.exit()
        startTime, endTime = self.check_startTime_endTime(sys._getframe().f_code.co_name, startTime, endTime)
        par = {
            name_symbol_or_pair: symbol_or_pair.upper(),
            'interval': interval,
            'startTime': startTime,
            'endTime': endTime,
            'limit': limit
        }
        if self.market_type.lower() == 'contract':
            par['contractType'] = contractType.upper()
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_current_average_price(self, symbol):
        """spot: https://binance-docs.github.io/apidocs/spot/en/#current-average-price"""
        if self.market_type != 'spot':
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} работает только со спотовым рынком'
            self.__logger.error(mes_to_log)
            sys.exit()
        end_point = '/api/v3/avgPrice'
        par = {
            'symbol': symbol.upper(),
        }
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_mark_price_and_funding_rate(self, symbol=None):
        """
        derv: https://binance-docs.github.io/apidocs/futures/en/#mark-price
        """
        end_point = '/fapi/v1/premiumIndex'
        par = {'symbol': symbol.upper()} if symbol else None
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_funding_rate_history(self, symbol=None, startTime = None, endTime = None, limit=100):
        """
        derv: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
        """
        end_point = '/fapi/v1/fundingRate'
        if limit < 0 and abs(limit) < 1000:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный limit {limit}'
            self.__logger.error(mes_to_log)
            sys.exit()
        startTime, endTime = self.check_startTime_endTime(sys._getframe().f_code.co_name, startTime, endTime)
        par = {
            'symbol': symbol.upper() if symbol else symbol,
            'startTime': startTime,
            'endTime': endTime,
            'limit': limit
        }
        return self.__request_template(end_point=end_point, par=par, method='get')



    def get_24hr_ticker_price_change(self, symbol=None, type_bin='FULL'):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#24hr-ticker-price-change-statistics
        derv: https://binance-docs.github.io/apidocs/futures/en/#24hr-ticker-price-change-statistics
        """
        if type_bin.lower() not in ['full', 'mini']:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный type_bin {type_bin} only: [full, mini]'
            self.__logger.error(mes_to_log)
            sys.exit()
        if symbol and ',' in symbol:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} если несклько символов то передавать листом [], а не строчкой через запятую'
            self.__logger.error(mes_to_log)
            sys.exit()
        end_point = '/api/v3/ticker/24hr' if self.market_type == 'spot' else '/fapi/v1/ticker/24hr'
        par = {}
        if type(symbol) is list and self.market_type == 'spot':
            par['symbols'] = '['+str(','.join([f'"{el.upper()}"' for el in symbol])) + ']'
        elif type(symbol) is str:
            par['symbol'] = symbol.upper()
        if self.market_type == 'spot':
            par['type'] = type_bin.upper()
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_price_ticker(self, symbol=None):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#symbol-price-ticker
        derv: https://binance-docs.github.io/apidocs/futures/en/#symbol-price-ticker
        """
        end_point = '/api/v3/ticker/price' if self.market_type == 'spot' else '/fapi/v1/ticker/price'
        par = {}
        if type(symbol) is list and self.market_type == 'spot':
            par['symbols'] = '['+str(','.join([f'"{el.upper()}"' for el in symbol])) + ']'
        elif type(symbol) is str:
            par['symbol'] = symbol.upper()
        elif symbol:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный symbol {symbol} для {self.market_type}'
            self.__logger.warning(mes_to_log)
            return None
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_symbol_order_book_ticker(self, symbol):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#symbol-order-book-ticker
        derv: https://binance-docs.github.io/apidocs/futures/en/#symbol-order-book-ticker
        """
        end_point = '/api/v3/ticker/bookTicker' if self.market_type == 'spot' else '/fapi/v1/ticker/bookTicker'
        par = {}
        if type(symbol) is list and self.market_type == 'spot':
            par['symbols'] = '[' + str(','.join([f'"{el.upper()}"' for el in symbol])) + ']'
        elif type(symbol) is str:
            par['symbol'] = symbol.upper()
        elif symbol:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный symbol {symbol} для {self.market_type}'
            self.__logger.warning(mes_to_log)
            return None
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_rolling_window_price_change_statistics(self, symbol_s, windowSize='1d', type_bin='MINI'):
        """https://binance-docs.github.io/apidocs/spot/en/#rolling-window-price-change-statistics"""
        end_point = '/api/v3/ticker'
        if type(symbol_s) is list: w_symbol = '[' + str(','.join([f'"{el.upper()}"' for el in symbol_s])) + ']'
        par = {'symbols': w_symbol} if type(symbol_s) is list else {'symbol': symbol_s.upper()}
        par['type'] = type_bin.upper()
        if 'm' in windowSize.lower():
            temp_windowSize = int(windowSize.replace('m', ''))
            if 1 > temp_windowSize > 59:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} ошибка в windowSize {windowSize}, минуты от 1 до 59'
                self.__logger.error(mes_to_log)
                sys.exit()
        elif 'h' in windowSize.lower():
            temp_windowSize = int(windowSize.replace('h', ''))
            if 1 > temp_windowSize > 23:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} ошибка в windowSize {windowSize}, часы от 1 до 23'
                self.__logger.error(mes_to_log)
                sys.exit()
        elif 'd' in windowSize.lower():
            temp_windowSize = int(windowSize.replace('d', ''))
            if 1 > temp_windowSize > 7:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} ошибка в windowSize {windowSize}, дни от 1 до 7'
                self.__logger.error(mes_to_log)
                sys.exit()
        elif windowSize != None:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не понятный windowSize {windowSize}'
            self.__logger.error(mes_to_log)
            sys.exit()
        par['windowSize'] = windowSize
        return self.__request_template(end_point=end_point, par=par, method='get')


    def get_open_interest(self, symbol):
        """
        https://binance-docs.github.io/apidocs/futures/en/#open-interest
        """
        end_point = '/fapi/v1/openInterest'
        par = {
            'symbol': symbol.upper()
        }
        return self.__request_template(end_point=end_point, par=par, method='get')


    @classmethod
    def get_default_def_5_end_point(cls, end_point, symbol, period, limit, startTime, endTime):
        lst_period = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if period not in lst_period:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный период period: {period} - {",".join(lst_period)}'
            cls.__logger.error(mes_to_log)
            sys.exit()
        if 30 > limit > 500:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный limit: {limit},  only: 30 - 500'
            cls.__logger.error(mes_to_log)
            sys.exit()
        if startTime or endTime:
            startTime, endTime = cls.check_startTime_endTime(sys._getframe().f_code.co_name, startTime, endTime)
            time_now = int(datetime.timestamp(datetime.utcnow())*1000)
            time_30_days = 30*24*60*60*1000
            if (time_now - startTime) > time_30_days:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} Only the data of the latest 30 days is available'
                cls.__logger.error(mes_to_log)
                sys.exit()
        par = {
            'symbol': symbol.upper(),
            'period': period,
            'startTime': startTime,
            'endTime': endTime,
            'limit': limit
        }
        return cls.__request_template(end_point=end_point, par=par, method='get')
    def get_open_interest_statistics(self, symbol, period, limit=30, startTime=None, endTime=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#open-interest-statistics
        """
        end_point = '/futures/data/openInterestHist'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, startTime, endTime)
    def get_top_trader_long_short_ratio_accounts(self, symbol, period, limit=30, startTime=None, endTime=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-accounts
        """
        end_point = '/futures/data/topLongShortAccountRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, startTime, endTime)
    def get_top_trader_long_short_ratio_positions(self, symbol, period, limit=30, startTime=None, endTime=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-positions
        """
        end_point = '/futures/data/topLongShortPositionRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, startTime, endTime)
    def get_long_short_ratio(self, symbol, period, limit=30, startTime=None, endTime=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#long-short-ratio
        """
        end_point = '/futures/data/globalLongShortAccountRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, startTime, endTime)
    def get_taker_buy_sell_volume(self, symbol, period, limit=30, startTime=None, endTime=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#taker-buy-sell-volume
        """
        end_point = '/futures/data/takerlongshortRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, startTime, endTime)


    def get_composite_index_symbol_information(self, symbol=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#composite-index-symbol-information
        """
        end_point = '/fapi/v1/indexInfo'
        par = {}
        if symbol:
            par['symbol'] = symbol.upper()
        return self.__request_template(end_point=end_point, par=par, method='get')



class Binance_websocket_public:
    __url = 'wss://fstream.binance.com'
    __url_auth = 'wss://fstream-auth.binance.com'
    __lg = Logger('Binance_websocket_public', type_log='w')
    __logger = __lg.logger

    def __init__(self, stream, queue, topics=None):
        self.queue = queue
        self.topics = topics
        self.stream = stream
        self.websocket_app = websocket.WebSocketApp(
            url=self.__url+stream,
            on_message=self.on_message,
            on_ping=self.on_ping,
            on_close=self.on_close,
            on_error=self.on_error,
            on_open=self.on_open,
        )

    def on_open(self, _wsapp):
        print("Connection opened")
        if self.stream == '/stream':
            data = {
                "method": "SUBSCRIBE",
                "params": self.topics,
                "id": int(time.time() * 1000)
            }
            _wsapp.send(json.dumps(data))

    def on_close(self, _wsapp, close_status_code, close_msg):
        if close_status_code is not None and close_msg is not None:
            print(f"Close connection by server, status {close_status_code}, close message {close_msg}")

    def on_error(self, _wsapp, error):
        def_name = sys._getframe().f_code.co_name
        mes_to_log = f'{def_name} Error: {error}, traceback: {traceback.format_exc()}'
        self.__logger.error(mes_to_log)
        print(mes_to_log)
        sys.exit()
    def on_ping(self, _wsapp, message):
        print(f"{str(datetime.now())} Got a ping! Ping msg is {message}")

    def stop(self):
        if self.websocket_app:
            self.websocket_app.close()

    def on_message(self, _wsapp, message):
        parsed = json.loads(message)
        self.queue.put(parsed)







