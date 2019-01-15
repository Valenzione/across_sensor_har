import pandas as pd
import numpy as np
from datetime import timedelta


def get_intervals(y: pd.Series):
    min_delta = timedelta(seconds=10)
    intervals = []
    row_generator = y.iteritems()
    datetime, label = next(row_generator)
    for datetime_next, label_next in row_generator:
        if label_next != label:
            if datetime_next-datetime > min_delta:
                intervals.append((datetime, datetime_next, label))
            datetime = datetime_next
            label = label_next
    return intervals