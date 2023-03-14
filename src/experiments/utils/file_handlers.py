import os
import pandas as pd
import datetime
import json, csv, pickle
from typing import List, Tuple, Dict, Any, Union


# read .csv as pandas.DataFrame
def read_csv_as_df(path:str, header=None, encoding='utf-8') -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=header, encoding=encoding, engine="python")
    except:
        df = pd.read_csv(path, header=header, engine="python")
    return df


# write .csv with header
def write_csv(path:str, data:List, header:List=None, encoding='utf-8') -> None:
    with open(path, 'w', newline="",  encoding=encoding) as f:
        writer = csv.writer(f)
        if type(header)==str: writer.writerow(header)
        writer.writerows(data)
    print(f"Successfully dumped {path} !")


# read .tsv
def read_tsv(path:str, header:List=None, encoding='utf-8') -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=header, encoding=encoding, engine="python", delimiter='\t')
    except:
        df = pd.read_csv(path, header=header, engine="python", delimiter='\t')
    return df


# write .tsv with header
def write_tsv(path:str, data:List, header:List=None, encoding='utf-8') -> None:
    with open(path, 'w', newline="",  encoding=encoding) as f:
        writer = csv.writer(f, delimiter='\t')
        if type(header)==str: writer.writerow(header)
        writer.writerows(data)
    print(f"Successfully dumped {path} !")


# read .json
def read_json(path:str, encoding='utf-8') -> Dict:
    with open(path, encoding=encoding) as f:
        return json.load(f)


# write .json
def write_json(path:str, data:dict, encoding='utf-8') -> None:
    with open(path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Successfully dumped {path} !")


# read .pickle
def read_pickle(path:str) -> Any:
    with open(path, 'rb') as f:
        any_type_instance = pickle.load(f)
    return any_type_instance


# write .pickle
def write_pickle(path:str, data:Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Successfully dumped {path} !")


# make dir if not exists
def makedirs(path:str):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_8char_datetime() -> str:
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    date_time = now.strftime('%y%m%d%H%M%S')
    return date_time
