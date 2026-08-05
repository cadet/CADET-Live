# -*- coding: utf-8 -*-

import copy
from datetime import datetime


def time_to_relative(data: dict, reference_time: datetime) -> dict:
    new_data = copy.deepcopy(data)
    for key, value in new_data.items():
        for item in value:
            item[0] -= reference_time
    return new_data


def time_to_absolute(data: dict, reference_time: datetime) -> dict:
    new_data = copy.deepcopy(data)
    for key, value in new_data.items():
        for item in value:
            item[0] += reference_time
    return new_data
