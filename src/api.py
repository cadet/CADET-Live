# -*- coding: utf-8 -*-

import time
from datetime import datetime, timedelta

import config
import requests
from apscheduler.schedulers.background import BackgroundScheduler

raw_config = config.get_config()


def post(url: str, path: str, payload: dict | None = None) -> None:
    print(datetime.now())
    resp = requests.post(f"{url}{path}", json=payload, timeout=10)
    resp.raise_for_status()


# def get(path: str) -> dict:
#    resp = requests.get(f"{url}{path}", timeout=10)
#    resp.raise_for_status()
#    return resp.json()


def control(type: str, config: dict, target_value: int = None) -> None:
    mapping = config.get("mapping")
    for item in mapping:
        if item.get("name") == type:
            value = target_value or str(item.get("value"))
            print(value)
            post(config.get("url"), item.get("path"), {"options": {item.get("label"): value}})


def control_sequence(sequences: dict, raw_config: dict) -> None:
    api_config = config.get_control_api(raw_config)

    scheduler.remove_all_jobs()

    for key in sequences:
        sequence = sequences.get(key)
        for timed_instruction in sequence:
            print(timed_instruction)
            scheduler.add_job(control, 'date', run_date=timed_instruction[0], args=[key, api_config, timed_instruction[1]])


scheduler = BackgroundScheduler()
scheduler.start()

api_config = config.get_control_api(raw_config)

# control("stirring", api_config, 0)

# sequence = {'stirring': [[datetime.now(), 100]], 'heating': [[datetime.now(), 20], [datetime.now(), 25]]}
sequence = {'stirring': [[datetime.now() + timedelta(seconds=10), 400], [datetime.now(), 700], [datetime.now() + timedelta(seconds=20), 400]]}
# sequence = {'stirring': [[datetime.now() + timedelta(seconds=10), 400]]}
print(datetime.now())
control_sequence(sequence, raw_config)

time.sleep(30)
