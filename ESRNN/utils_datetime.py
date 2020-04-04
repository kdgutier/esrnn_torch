from pandas.tseries.offsets import DateOffset


def custom_offset(freq):

    if freq == 'Y':
        return lambda x: DateOffset(years = x)
    elif freq == 'M':
        return lambda x: DateOffset(months = x)
    elif freq == 'W':
        return lambda x: DateOffset(weeks = x)
    elif freq == 'H':
        return lambda x: DateOffset(hours = x)
    elif freq == 'Q':
        return lambda x: DateOffset(months = 3*x)
