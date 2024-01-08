import requests
import sys
from datetime import datetime
import json
import traceback
import time
import websocket
from PlvLogger import Logger
# from queue import Queue
# import _thread


class SyncBinancePublic:
    _main_url_spot = r'https://api.binance.com'
    _main_url_derv = r'https://fapi.binance.com'

    def __init__(self, market_type):
        self.market_type = market_type
        self._logger = Logger('Binance_public', type_log='w').logger

    def __setattr__(self, key, value):
        if key == 'market_type' and value not in ['spot', 'der']:
            self._logger.error(f'Неизвестный тип рынка {self.market_type}')
            raise TypeError(f"Неверный market_type {self.market_type}")
        object.__setattr__(self, key, value)

    def _request_template(self, end_point, par=None, method='get'):
        work_link = self._main_url_spot if self.market_type == 'spot' else self._main_url_derv
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
                self._logger.error(mes_to_log)
                raise TypeError(f"Неверный method {mes_to_log}")
        if req.ok:
            return req.json()

    def check_start_time_end_time(self, def_name, start_time, end_time):
        def validate_time(time, time_name):
            if time:
                time = int(time)
                if len(str(time)) == 10:
                    time *= 1000
                elif len(str(time)) != 13:
                    mes_to_log = f"{def_name} ошибка в {time_name} {time}"
                    self._logger.error(mes_to_log)
                    raise TypeError(f"len(str(time)) != 13 {mes_to_log}")
            return time

        start_time = validate_time(start_time, "start_time")
        end_time = validate_time(end_time, "end_time")

        return start_time, end_time

    def get_exchange_information(self):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#exchange-information
        derv: https://binance-docs.github.io/apidocs/futures/en/#exchange-information
        """
        end_point = '/api/v3/exchangeInfo' if self.market_type == 'spot' else '/fapi/v1/exchangeInfo'
        return self._request_template(end_point=end_point, method='get')

    def get_order_book(self, symbol, limit=100):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#order-book
        derv: https://binance-docs.github.io/apidocs/futures/en/#order-book
        """
        if self.market_type == 'der' and limit not in [5, 10, 20, 50, 100, 500, 1000]:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} Неверная глубина книги дериватив {limit}'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        end_point = '/api/v3/depth' if self.market_type == 'spot' else '/fapi/v1/depth'
        par = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        return self._request_template(end_point=end_point, par=par, method='get')

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
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_old_trade_lookup(self, symbol, from_id=None, limit=500):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#old-trade-lookup
        derv: https://binance-docs.github.io/apidocs/futures/en/#old-trades-lookup-market_data
        """
        end_point = '/api/v3/historicalTrades' if self.market_type == 'spot' else '/fapi/v1/historicalTrades'
        par = {
            'symbol': symbol.upper(),
            'limit': limit,
            'fromId': from_id
        }
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_aggregate_trades_list(self, symbol, from_id=None, start_time=None, end_time=None, limit=500):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#compressed-aggregate-trades-list
        derv: https://binance-docs.github.io/apidocs/futures/en/#compressed-aggregate-trades-list
        Агрегируются трейды (возможно только мелкие) одного пользователя до 100мс
        """
        end_point = '/api/v3/aggTrades' if self.market_type == 'spot' else '/fapi/v1/aggTrades'
        par = {
            'symbol': symbol.upper(),
            'fromId': from_id,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_kline(self, symbol_or_pair, interval, start_time=None, end_time=None, contract_type='PERPETUAL', limit=500):
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
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        if contract_type.upper() not in lst_contract_type:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} Неизвестный contractType {contract_type}, only {",".join(lst_contract_type)}'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        if limit < 0 and ((self.market_type != 'spot' and abs(limit) < 1500) or (self.market_type == 'spot' and abs(limit) < 1000)):
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный limit {limit}'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        name_symbol_or_pair = 'symbol'
        end_point = None
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
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        start_time, end_time = self.check_start_time_end_time(sys._getframe().f_code.co_name, start_time, end_time)
        par = {
            name_symbol_or_pair: symbol_or_pair.upper(),
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        if self.market_type.lower() == 'contract':
            par['contractType'] = contract_type.upper()
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_current_average_price(self, symbol):
        """spot: https://binance-docs.github.io/apidocs/spot/en/#current-average-price"""
        if self.market_type != 'spot':
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} работает только со спотовым рынком'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        end_point = '/api/v3/avgPrice'
        par = {
            'symbol': symbol.upper(),
        }
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_mark_price_and_funding_rate(self, symbol=None):
        """
        derv: https://binance-docs.github.io/apidocs/futures/en/#mark-price
        """
        end_point = '/fapi/v1/premiumIndex'
        par = {'symbol': symbol.upper()} if symbol else None
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_funding_rate_history(self, symbol=None, start_time=None, end_time=None, limit=100):
        """
        derv: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
        """
        end_point = '/fapi/v1/fundingRate'
        if limit < 0 and abs(limit) < 1000:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный limit {limit}'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        start_time, end_time = self.check_start_time_end_time(sys._getframe().f_code.co_name, start_time, end_time)
        par = {
            'symbol': symbol.upper() if symbol else symbol,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_24hr_ticker_price_change(self, symbol=None, type_bin='FULL'):
        """
        spot: https://binance-docs.github.io/apidocs/spot/en/#24hr-ticker-price-change-statistics
        derv: https://binance-docs.github.io/apidocs/futures/en/#24hr-ticker-price-change-statistics
        """
        if type_bin.lower() not in ['full', 'mini']:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный type_bin {type_bin} only: [full, mini]'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        if symbol and ',' in symbol:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} если несклько символов то передавать листом [], а не строчкой через запятую'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        end_point = '/api/v3/ticker/24hr' if self.market_type == 'spot' else '/fapi/v1/ticker/24hr'
        par = {}
        if type(symbol) is list and self.market_type == 'spot':
            par['symbols'] = '['+str(','.join([f'"{el.upper()}"' for el in symbol])) + ']'
        elif type(symbol) is str:
            par['symbol'] = symbol.upper()
        if self.market_type == 'spot':
            par['type'] = type_bin.upper()
        return self._request_template(end_point=end_point, par=par, method='get')

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
            self._logger.warning(mes_to_log)
            return None
        return self._request_template(end_point=end_point, par=par, method='get')

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
            self._logger.warning(mes_to_log)
            return None
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_rolling_window_price_change_statistics(self, symbol_s, windowSize='1d', type_bin='MINI'):
        """https://binance-docs.github.io/apidocs/spot/en/#rolling-window-price-change-statistics"""
        end_point = '/api/v3/ticker'
        if type(symbol_s) is list:
            par = {'symbols': '[' + str(','.join([f'"{el.upper()}"' for el in symbol_s])) + ']'}
        else:
            par = {'symbol': symbol_s.upper()}
        par['type'] = type_bin.upper()
        if 'm' in windowSize.lower():
            temp_windowSize = int(windowSize.replace('m', ''))
            if 1 > temp_windowSize > 59:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} ошибка в windowSize {windowSize}, минуты от 1 до 59'
                self._logger.error(mes_to_log)
                raise TypeError(mes_to_log)
        elif 'h' in windowSize.lower():
            temp_windowSize = int(windowSize.replace('h', ''))
            if 1 > temp_windowSize > 23:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} ошибка в windowSize {windowSize}, часы от 1 до 23'
                self._logger.error(mes_to_log)
                raise TypeError(mes_to_log)
        elif 'd' in windowSize.lower():
            temp_windowSize = int(windowSize.replace('d', ''))
            if 1 > temp_windowSize > 7:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} ошибка в windowSize {windowSize}, дни от 1 до 7'
                self._logger.error(mes_to_log)
                raise TypeError(mes_to_log)
        elif windowSize != None:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не понятный windowSize {windowSize}'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        par['windowSize'] = windowSize
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_open_interest(self, symbol):
        """
        https://binance-docs.github.io/apidocs/futures/en/#open-interest
        """
        end_point = '/fapi/v1/openInterest'
        par = {
            'symbol': symbol.upper()
        }
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_default_def_5_end_point(self, end_point, symbol, period, limit, start_time, end_time):
        lst_period = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
        if period not in lst_period:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный период period: {period} - {",".join(lst_period)}'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        if 30 > limit > 500:
            def_name = sys._getframe().f_code.co_name
            mes_to_log = f'{def_name} не корректный limit: {limit},  only: 30 - 500'
            self._logger.error(mes_to_log)
            raise TypeError(mes_to_log)
        if start_time or end_time:
            start_time, end_time = self.check_start_time_end_time(sys._getframe().f_code.co_name, start_time, end_time)
            time_now = int(datetime.timestamp(datetime.utcnow())*1000)
            time_30_days = 30*24*60*60*1000
            if (time_now - start_time) > time_30_days:
                def_name = sys._getframe().f_code.co_name
                mes_to_log = f'{def_name} Only the data of the latest 30 days is available'
                self._logger.error(mes_to_log)
                raise TypeError(mes_to_log)
        par = {
            'symbol': symbol.upper(),
            'period': period,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        return self._request_template(end_point=end_point, par=par, method='get')

    def get_open_interest_statistics(self, symbol, period, limit=30, start_time=None, end_time=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#open-interest-statistics
        """
        end_point = '/futures/data/openInterestHist'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, start_time, end_time)

    def get_top_trader_long_short_ratio_accounts(self, symbol, period, limit=30, start_time=None, end_time=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-accounts
        """
        end_point = '/futures/data/topLongShortAccountRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, start_time, end_time)

    def get_top_trader_long_short_ratio_positions(self, symbol, period, limit=30, start_time=None, end_time=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-positions
        """
        end_point = '/futures/data/topLongShortPositionRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, start_time, end_time)

    def get_long_short_ratio(self, symbol, period, limit=30, start_time=None, end_time=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#long-short-ratio
        """
        end_point = '/futures/data/globalLongShortAccountRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, start_time, end_time)

    def get_taker_buy_sell_volume(self, symbol, period, limit=30, start_time=None, end_time=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#taker-buy-sell-volume
        """
        end_point = '/futures/data/takerlongshortRatio'
        return self.get_default_def_5_end_point(end_point, symbol, period, limit, start_time, end_time)

    def get_composite_index_symbol_information(self, symbol=None):
        """
        https://binance-docs.github.io/apidocs/futures/en/#composite-index-symbol-information
        """
        end_point = '/fapi/v1/indexInfo'
        par = {}
        if symbol:
            par['symbol'] = symbol.upper()
        return self._request_template(end_point=end_point, par=par, method='get')


class SyncBinanceWebsocketPublic:
    _dict_urls = {
        'spot': 'wss://stream.binance.com:9443',
        'der': 'wss://fstream.binance.com',
        'der_auth': 'wss://fstream-auth.binance.com',
        'spot_auth': ''
    }

    def __init__(self, trade_type, stream, queue, topics=None):
        if trade_type not in self._dict_urls:
            raise ValueError(f"trade_type '{trade_type}' is not valid.")
        self._logger = Logger('Binance_websocket_public', type_log='w').logger
        self.queue = queue
        self.topics = topics
        self.stream = stream
        self.websocket_app = websocket.WebSocketApp(
            url=self._dict_urls.get(trade_type)+stream,
            on_message=self.on_message,
            on_ping=self.on_ping,
            on_close=self.on_close,
            on_error=self.on_error,
            on_open=self.on_open,
        )

    def __del__(self):
        self.websocket_app.close()

    def on_open(self, _wsapp):
        print("Connection opened")
        if self.stream == '/stream':
            data = {
                "method": "SUBSCRIBE",
                "params": self.topics,
                "id": int(time.time() * 1000)
            }
            _wsapp.send(json.dumps(data))

    @staticmethod
    def on_close(_wsapp, close_status_code, close_msg):
        if close_status_code is not None and close_msg is not None:
            print(f"Close connection by server, status {close_status_code}, close message {close_msg}")

    def on_error(self, _wsapp, error):
        def_name = sys._getframe().f_code.co_name
        mes_to_log = f'{def_name} Error: {error}, traceback: {traceback.format_exc()}'
        self._logger.error(mes_to_log)
        print(mes_to_log)
        raise TypeError(mes_to_log)

    @staticmethod
    def on_ping(_wsapp, message):
        print(f"{str(datetime.now())} Got a ping! Ping msg is {message}")

    def stop(self):
        if self.websocket_app:
            self.websocket_app.close()

    def on_message(self, _wsapp, message):
        parsed = json.loads(message)
        # print(len(parsed))
        self.queue.put(parsed)