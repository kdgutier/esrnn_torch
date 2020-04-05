from pandas.tseries.offsets import DateOffset


def custom_offset(freq, x):
    """
    Returns a custom offset of x according to freq.

    Parameters
    ----------
    freq: str
        Frequency of data, allowed freqs:  'Y', 'M', 'W', 'H', 'Q', 'D'.
    x:
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
