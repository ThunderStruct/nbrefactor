


class Logger:

    H_RULER_SYMBOL = '▬'

    class Color:
        RED     = '\x1b[31m'
        GREEN   = '\x1b[32m'
        BLUE    = '\x1b[34m'
        YELLOW  = '\x1b[33m'
        MAGENTA = '\x1b[35m'
        RESET   = '\x1b[0m'

    @staticmethod
    def log(msg, tag=None, color=None):
        if color:
            if tag:
                print((
                    f'{color}{tag}{Logger.Color.RESET}: {msg}'
                ))
            else:
                print(msg)
        else:
            # un-colored log
            if tag:
                print(f'{tag}: {msg}')
            else:
                print(msg)


    @staticmethod
    def horizontal_separator(length=50, symbol=None, color=Color.RESET):
        print('\n', color,
              ((symbol if symbol else Logger.H_RULER_SYMBOL) * length), 
              Logger.Color.RESET, '\n', sep='')