from .config import Config
import time
import sys
import logging
from datetime import datetime
import inspect
import os

"""NASKit's Custom Logger and Formatter
"""

class _ColoredFormatter(logging.Formatter):
    """
    Custom color formatter for the :class:`~Logger` class
    """

    ANSI_COLORS = {
        'none': '\x1b[0m',
        'black': '\x1b[30m',
        'red': '\x1b[31m',
        'green': '\x1b[32m',
        'yellow': '\x1b[33m',
        'blue': '\x1b[34m',
        'magenta': '\x1b[35m',
        'cyan': '\x1b[36m',
        'white': '\x1b[37m',

        'lavender': '\x1b[38;5;147m',
        'pink': '\x1b[38;5;201m'
    }

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(_ColoredFormatter, self).__init__(fmt, datefmt, style)

    def colorize(self, message, color, start_idx, end_idx):
        if color in self.ANSI_COLORS:
            # reset the color after colorizing the given message
            return f"{self.ANSI_COLORS[color]}{message[start_idx:end_idx]}" + \
            f"{self.ANSI_COLORS['none']}{message[end_idx:]}"
        else:
            return message

    def format(self, record):

        formatted = ''
        # if the message starts with \n, shift the whole record down 1 line
        if record.msg[0] == '\n' or record.msg[0:3] == ': \n':
            formatted = '\n' + super(_ColoredFormatter,
                                     self).format(record).replace('\n', '', 1)
        else:
            formatted = super(_ColoredFormatter, self).format(record)

        levelname = record.levelname
        header_idx = formatted.index(levelname)
        end_idx = header_idx + formatted[header_idx:].index(' ')

        if record.levelno == logging.ERROR:
            return self.colorize(formatted, 'red', 0, end_idx)
        elif record.levelno == logging.WARNING:
            return self.colorize(formatted, 'yellow', 0, end_idx)
        elif record.levelno == logging.INFO:
            return self.colorize(formatted, 'blue', 0, end_idx)
        elif record.levelno == logging.DEBUG:
            return self.colorize(formatted, 'yellow', 0, end_idx)
        elif record.levelno == logging.CRITICAL:
            return self.colorize(formatted, 'pink', 0, end_idx)
        else:
            return formatted

class Logger:
    """
    Responsible for formatting and managing all logging-related calls
    """

    def setup_logger(colored=True):
        """
        Initializes the `logger` member once. All arguments default to a \
        predefined style

        Args:
            colored (optional, bool): whether or not the logger distinguishes \
            levels with predefined colors
        """

        if hasattr(Logger, '__logger') and hasattr(Logger, '__progress_logger'):
            return

        handler = logging.StreamHandler()
        prg_handler = logging.StreamHandler()

        # init standard logger with function name from inspect
        Logger.__logger = logging.Logger('std')
        # init progress logger (used for updates such as running loss per epoch
        # during training, etc.)
        Logger.__progress_logger = logging.Logger('PROGRESS')
        # init file logging dict
        Logger.__file_logger = {}

        # standard logger formatting
        fmt = '[%(asctime)s:%(msecs)03d] \033[1m%(levelname)s\033[0m%(message)s'
        datefmt = '%d/%m %H:%M:%S'

        # progress logger formatting
        prg_fmt = (
            '\x1b[36m[%(asctime)s.%(msecs)03d] \033[1m%(name)s'
            '\033[0m\x1b[0m: %(message)s'
        )
        prg_datefmt = '%d/%m %H:%M:%S'
        prg_formatter = logging.Formatter(prg_fmt, datefmt=prg_datefmt)
        prg_handler.setFormatter(prg_formatter)

        if colored:
            handler.setFormatter(_ColoredFormatter(fmt, datefmt, '%'))
        else:
            formatter = logging.Formatter(fmt, datefmt=datefmt)
            handler.setFormatter(formatter)

        Logger.__logger.addHandler(handler)
        Logger.__progress_logger.addHandler(prg_handler)


    def debug(*msg, caller=True, line=True):
        prepend = (
            f' \033[3m(caller: {inspect.stack()[1][3]}'
            f'{", line: " + str(inspect.stack()[1][2]) if line else ""})'
            '\033[23m: ' if caller else ': '
        )
        if msg is None:
            Logger.__logger.debug(prepend)
            return
        Logger.__logger.debug(prepend + ' '.join([repr(m) \
                                                  if not isinstance(m, str) \
                                                  else m for m in msg]))

    def info(*msg, caller=False, line=False):
        prepend = (
            f' \033[3m(caller: {inspect.stack()[1][3]}' + \
            f'{", line: " + str(inspect.stack()[1][2]) if line else ""})' + \
            ('\033[23m: ' if caller else ': ')
        )
        Logger.__logger.info(prepend + ' '.join(msg))

    def warning(*msg, caller=True, line=True):
        prepend = (
            f' \033[3m(caller: {inspect.stack()[1][3]}' + \
            f'{", line: " + str(inspect.stack()[1][2]) if line else ""})' + \
            ('\033[23m: ' if caller else ': ')
        )
        Logger.__logger.warning(prepend + ' '.join(msg))

    def error(*msg, caller=True, line=True):
        prepend = (
            f' \033[3m(caller: {inspect.stack()[1][3]}' + \
            f'{", line: " + str(inspect.stack()[1][2]) if line else ""})' + \
            ('\033[23m: ' if caller else ': ')
        )
        Logger.__logger.error(prepend + ' '.join(msg))

    def critical(*msg, caller=False, line=False):
        prepend = (
            f' \033[3m(caller: {inspect.stack()[1][3]}' + \
            f'{", line: " + str(inspect.stack()[1][2]) if line else ""})' + \
            ('\033[23m: ' if caller else ': ')
        )
        Logger.__logger.critical(prepend + ' '.join(msg))

    def progress(*msg):
        Logger.__progress_logger.info(' '.join(msg))

    def separator(color='\x1b[32m', length=80, symbol='▬'):
        """
        Symbols could be any string, some suggestions are:
        ● ■ ◆ ▬ | ~ - = + # & ^
        """
        reset_color = '\x1b[0m'

        separator = f'\n{color}{symbol * length}{reset_color}\n'

        sys.stdout.write(separator)
        sys.stdout.flush()

    def success(*msg, color='\x1b[32m'):
        """
        """
        reset_color = '\x1b[0m'
        content = str(' '.join(msg))

        log = f'\n{color}{content}{reset_color}\n'

        sys.stdout.write(log)
        sys.stdout.flush()



class TrainingLogger:

    __total_epochs = 0
    __total_train_batches = 0
    __total_val_batches = 0
    __task_id = 0
    __task_version = 0
    __task_name = ''
    __log_to_file = False
    __log_file_content = []

    def progress(*msg, is_last=False, delay=0.00):
        """
        Logs without Python's :class:`logging` to apply carriage return when
        applicable

        Args:
            carriage (optional, bool): whether or not to apply carriage \
            return, defaults to `False`
            delay (optional, float): amount of time to sleep for readability \
            purposes when logs are processed too quickly. Defaults to 0.00
        """

        current_datetime = datetime.now()
        time_str = current_datetime.strftime('%d/%m %H:%M:%S.%f')[:-3]

        text = f'\x1b[36m[{time_str}]'
        text += f' \033[1mPROGRESS\033[0m\x1b[0m: {"".join(msg)}'

        if TrainingLogger.__log_to_file is not None:
            # re-init log, but stripped of ANSI colors for file-logging
            file_text = f'[time_str] PROGRESS: {"".join(msg)}'
            TrainingLogger.__log_file_content.append(file_text)

        if delay > 0:
            time.sleep(delay)

        sys.stdout.write('\r' + text + ('\n' if is_last else ''))
        sys.stdout.flush()

    def reset(task_id, task_version, task_name, total_epochs,
              total_train_batches, total_val_batches, log_to_file):
        TrainingLogger.__total_epochs = total_epochs
        TrainingLogger.__total_train_batches = total_train_batches
        TrainingLogger.__total_val_batches = total_val_batches
        TrainingLogger.__task_id = task_id
        TrainingLogger.__task_name = task_name
        TrainingLogger.__task_version = task_version
        TrainingLogger.__log_to_file = log_to_file

    def log_training(epoch, batch, avg_loss, avg_acc):
        task = f'Task ({TrainingLogger.__task_id} '
        task += f'v.{TrainingLogger.__task_version}: '
        task += f'"{TrainingLogger.__task_name}") | '
        epochs = task + f'Epoch {epoch+1}/{TrainingLogger.__total_epochs}, '
        epochs += f'Batch {batch+1}/{TrainingLogger.__total_train_batches}'
        running_metrics = f'Loss: {avg_loss}, Acc.: {avg_acc}'
        is_last = (batch == TrainingLogger.__total_train_batches - 1)
        # running_metrics += '\n' if is_last else ''
        TrainingLogger.progress(epochs + ' - ' + running_metrics,
                                is_last=is_last)

    def log_validation(epoch, batch, avg_loss, avg_acc):
        task = f'Task ({TrainingLogger.__task_id} '
        task += f'v.{TrainingLogger.__task_version}: '
        task += f'"{TrainingLogger.__task_name}") | '
        epochs = task + f'Epoch {epoch+1}/{TrainingLogger.__total_epochs}, '
        epochs += f'Batch {batch+1}/{TrainingLogger.__total_val_batches}'
        running_metrics = f'Loss: {avg_loss}, Acc.: {avg_acc}'
        is_last = (batch == TrainingLogger.__total_val_batches - 1)
        # running_metrics += '\n' if is_last else ''
        TrainingLogger.progress(epochs + ' - ' + running_metrics,
                                is_last=is_last)

    def commit_file(filename, path='./training_logs/'):

        dir_path = os.path.join(Config.BASE_PATH, path)
        full_path = os.path.join(dir_path, filename)

        ensure_dir(dir_path, True)

        try:
            with open(full_path, 'w') as file:
                file.write('\n'.join(TrainingLogger.__log_file_content))
            # reset uncommitted content
            TrainingLogger.__log_file_content = []
        except Exception as e:
            Logger.debug(f'Error writing log to {full_path}: {e}')


Logger.setup_logger()

