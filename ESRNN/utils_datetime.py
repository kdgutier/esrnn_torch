from pandas.tseries.offsets import DateOffset
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd

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

def fix_date(col, freq):
    """Function that takes data values and a frequency and returns fixed date
    Parameters
    ----------
    col: Series or array (datetime)
        Values to be processed
    freq: str
        frequency to fix
    Returns
    -------
    Series of dates with the closest week start behind
    Example
    -------
    datos['date_week_start'] = fix_date(datos['date'], 'W')
    """
    if freq=='W':
      return date_to_start_week(col)
    if freq=='M':
      return date_to_start_month(col)
    if freq=='Q':
      return date_to_start_quarter(col)
    if freq=='Y':
      return date_to_start_year(col)
    else:
      return col

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
    datos['date_week_start'] = date_to_start_week(datos['date'])
    """
    col = col - col.dt.weekday.apply(lambda x: timedelta(days=(x + week_starts_in + 1) % 7))
    return col

def date_to_start_month(col):
    """Function that takes data values and returns initial date for months
    Parameters
    ----------
    col: Series or array (datetime)
        Values to be processed
    Returns
    -------
    Series of dates with the initial date of month
    Example
    -------
    datos['date_week_start'] = date_to_start_week(datos['date'])
    """
    col = col.apply(lambda x: x.replace(day=1))
    return col

def date_to_start_quarter(col):
    """Function that takes data values and returns initial date for quarter
    Parameters
    ----------
    col: Series or array (datetime)
        Values to be processed
    Returns
    -------
    Series of dates with the initial date of quarters
    Example
    -------
    datos['date_quarter_start'] = date_to_start_week(datos['date'])
    """
    col = col - col.dt.month.apply(lambda x: custom_offset('M', (x-1) % 3))
    col = col.apply(lambda x: x.replace(day=1))
    return col

def date_to_start_year(col):
    """Function that takes data values and returns initial date for year
    Parameters
    ----------
    col: Series or array (datetime)
        Values to be processed
    Returns
    -------
    Series of dates with the initial date of year
    Example
    -------
    datos['date_year_start'] = date_to_start_year(datos['date'])
    """
    col = col.apply(lambda x: x.replace(month=1).replace(day=1))
    return col
