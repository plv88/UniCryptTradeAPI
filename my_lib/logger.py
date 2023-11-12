import logging
import os



class Logger:
    LOG_LEVELS = {
        'w': logging.WARNING,
        'i': logging.INFO
    }

    def __init__(self, logger_name, type_log, log_directory="logs"):
        if type_log not in self.LOG_LEVELS:
            raise EmptyResultException(f"Invalid type_log provided: {type_log}. Valid options are: {', '.join(self.LOG_LEVELS.keys())}")
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.LOG_LEVELS[type_log])
        log_path = os.path.join(log_directory, f'{logger_name}.log')
        handler = logging.FileHandler(log_path)
        handler.setLevel(self.LOG_LEVELS[type_log])
        formatter = logging.Formatter('%(levelname)s %(name)s %(asctime)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)



class EmptyResultException(Exception):
    """Исключение, возникающее, когда результат запроса к базе данных оказывается пустым, но ожидалось получение данных."""

    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.args[0]}"