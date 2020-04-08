from pandas.tseries.offsets import DateOffset
from datetime import timedelta

def custom_offset(freq, x):
    """
    Returns a custom offset of x according to freq.

    Parameters
    ----------
    freq: str
        Frequency of data, allowed freqs:  'Y', 'M', 'W', 'H', 'Q', 'D'.
    x: int
        number of periods
    """
    allowed_freqs= ('Y', 'M', 'W', 'H', 'Q', 'D')
    if freq not in allowed_freqs:
        raise ValueError(f'kind must be one of {allowed_kinds}')

    if freq == 'Y':
        return DateOffset(years = x)
    elif freq == 'M':
        return DateOffset(months = x)
    elif freq == 'W':
        return DateOffset(weeks = x)
    elif freq == 'H':
        return DateOffset(hours = x)
    elif freq == 'Q':
        return DateOffset(months = 3*x)
    elif freq == 'D':
        return DateOffset(days = x)

def date_to_start_week(col, week_starts_in=0):
    """Function that takes data values and returns week number
    Parameters
    ----------
    col: Series or array (datetime)
        Values to be processed
    week_starts_in: int
        Week number of first day (start_day).
        Monday is 0, thursday is 3
    Returns
    -------
    Series of dates with the closest week start behind
    Example
    -------
    datos['fecha_week_start'] = date_to_start_week(datos['fecha'])
    """
    col = col - col.dt.weekday.apply(lambda x: timedelta(days=(x + week_starts_in + 1) % 7))
    return col
