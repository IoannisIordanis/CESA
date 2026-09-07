# utils.py
import sys
try:
    import pdmlabs
    sys.modules['OnlineADEngine'] = pdmlabs
except ImportError:
    pass

import pandas as pd
import numpy as np
from OnlineADEngine.utils.dataset import Dataset


def get_scania_train_sources():
    return ['52', '197', '7905', '2089', '690', '18926', '2205', '10111', '273', '3307', '2187', '25625', '821', '161', '16328', '1246', '234', '206', '3942', '185', '7684', '1638', '7961', '11593', '1783', '20417', '775', '20951', '1876', '241', '1988', '2257', '1787', '1709', '4686', '269']


def get_scania_val_sources():
    return ['2122', '18297', '20021', '2265', '1968', '7854', '84', '1646', '116', '8771', '3271', '2213']

def get_scania_test_sources():
    return ['1661', '11251', '255', '1656', '1301', '32842', '1644', '228', '4304', '721', '1950', '11496', '2672']



dataset_info = {
    "PBC": {
        "path": "Data/pbc2.csv",
        "non_actionable_original": ["sex", "age"],
    },
    "HNEI": {
        "path": "Data/HNF/HNEI_combined.csv",
        "timestamp_column": "Artificial_timestamp",
        "target": "RUL",
        "train_sources": ["a","b","c","d","e","f","g","n"],
        "val_sources": ["l","j","o"],
        "test_sources": ["t","s","p"],
        "non_actionable_features": {'Decrement 3.6-3.4V (s)'},
        "non_actionable_original": ['Decrement 3.6-3.4V (s)'],
        # "non_actionable_features": {},
        # "non_actionable_original": [],
    },
    "SCANIA": {
        "path": "Data/SCANIA/SCANIA_rtf.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
    "SCANIAn": {
        "path": "Data/SCANIA/SCANIA_rtfn.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
    "SCANIA10": {
        "path": "Data/SCANIA/SCANIA_rtfn_10.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
    "SCANIA25": {
        "path": "Data/SCANIA/SCANIA_rtfn_25.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
    "SCANIA50": {
        "path": "Data/SCANIA/SCANIA_rtfn_50.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
    "SCANIA75": {
        "path": "Data/SCANIA/SCANIA_rtfn_75.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
    "SCANIA100": {
        "path": "Data/SCANIA/SCANIA_rtfn_100.csv",
        "timestamp_column": "time_step",
        "target": "RUL",
        "train_sources": get_scania_train_sources(),
        "val_sources": get_scania_val_sources(),
        "test_sources": get_scania_test_sources(),
        "no_event_value": None,
        "limit": False,
        "non_actionable_features": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
        "non_actionable_original": ["Spec_0_Cat1","Spec_0_Cat2","Spec_1_Cat1","Spec_1_Cat10","Spec_1_Cat11","Spec_1_Cat12","Spec_1_Cat13","Spec_1_Cat14","Spec_1_Cat15","Spec_1_Cat16","Spec_1_Cat17","Spec_1_Cat18","Spec_1_Cat2","Spec_1_Cat22","Spec_1_Cat23","Spec_1_Cat3","Spec_1_Cat4","Spec_1_Cat5","Spec_1_Cat6","Spec_1_Cat7","Spec_1_Cat8","Spec_1_Cat9","Spec_2_Cat1","Spec_2_Cat10","Spec_2_Cat13","Spec_2_Cat14","Spec_2_Cat15","Spec_2_Cat2","Spec_2_Cat3","Spec_2_Cat5","Spec_2_Cat6","Spec_2_Cat7","Spec_2_Cat8","Spec_2_Cat9","Spec_3_Cat1","Spec_3_Cat2","Spec_3_Cat3","Spec_4_Cat1","Spec_5_Cat1","Spec_5_Cat2","Spec_5_Cat3","Spec_5_Cat4","Spec_6_Cat1","Spec_6_Cat12","Spec_6_Cat2","Spec_6_Cat3","Spec_6_Cat4","Spec_6_Cat5","Spec_6_Cat9","Spec_7_Cat1","Spec_7_Cat2","Spec_7_Cat3","Spec_7_Cat4","Spec_7_Cat5","Spec_7_Cat6","Spec_7_Cat7"],
    },
}




def _get_non_actionable_columns_from_original(original_names, all_columns):
    """
    Given a list of original feature names (e.g., ['sex','age']) and the list of all column names
    after preprocessing, return a set of column names that should be excluded.
    Rules:
      - If the original name appears exactly as a column, include it.
      - If the original name is categorical and one‑hot encoded, include all columns starting with
        f"{original_name}_" (e.g., 'sex_').
      - For PBC's 'age' which becomes 'age_combined', we handle that by looking for a column
        containing 'age' (case‑insensitive) that is not already captured.
    """
    excluded = set()
    for orig in original_names:
        # exact match
        if orig in all_columns:
            excluded.add(orig)
        # one‑hot encoded pattern: name + '_'
        pattern = f"{orig}_"
        excluded.update([c for c in all_columns if c.startswith(pattern)])
        # special handling for age -> age_combined (if no exact match and no underscore pattern)
        if orig.lower() == 'age' and 'age' not in excluded:
            age_cols = [c for c in all_columns if 'age' in c.lower()]
            excluded.update(age_cols)
    return excluded

def load_PBC(keep_identifiers=False, rul_sa="sa", path="Data/"):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv(f"{path}pbc2.csv")
    data['histologic'] = data['histologic'].astype(str)
    dat_cat = data[['drug', 'sex', 'ascites', 'hepatomegaly',
                    'spiders', 'edema', 'histologic']]
    dat_num = data[['serBilir', 'serChol', 'albumin', 'alkaline',
                    'SGOT', 'platelets', 'prothrombin']]
    age = data['age'] + data['years']

    # One-hot encode categoricals
    x1 = pd.get_dummies(dat_cat).values
    x2 = dat_num.values
    x3 = age.values.reshape(-1, 1)
    x = np.hstack([x1, x2, x3])

    time = (data['years'] - data['year']).values
    event = data['status2'].values
    sources = data['id'].values

    x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
    x_ = StandardScaler().fit_transform(x)

    # Create meaningful column names
    dummies = pd.get_dummies(dat_cat).columns.tolist()
    num_cols = dat_num.columns.tolist()
    age_col = ['age_combined']
    all_cols = dummies + num_cols + age_col

    df = pd.DataFrame(x_, columns=all_cols)
    df["RUL"] = time * 12
    df["event"] = event
    df["source"] = [str(s) for s in sources]
    df = df[df['source'].map(df['source'].value_counts()) >= 5]

    # Determine non‑actionable columns based on dataset_info["PBC"]["non_actionable_original"]
    non_act_orig = dataset_info["PBC"].get("non_actionable_original", [])
    non_actionable_cols = _get_non_actionable_columns_from_original(non_act_orig, df.columns)

    dfevent = df[df["event"] == 1]
    df_non_event = df[df["event"] == 0]

    import random
    random.seed(42)
    unid = dfevent["source"].unique().tolist()
    unid_non_event = df_non_event["source"].unique().tolist()[:len(unid)]
    while df[df["source"].isin(unid_non_event)].shape[0] / df[df["source"].isin(unid)].shape[0] > 1.05:
        unid_non_event.pop()

    train_sources = unid[:int(len(unid) * 0.6)] + unid_non_event[:int(len(unid_non_event) * 0.6)]
    val_sources = unid[int(len(unid) * 0.6):int(len(unid) * 0.8)] + unid_non_event[int(len(unid_non_event) * 0.6):int(len(unid_non_event) * 0.8)]
    test_sources = unid[int(len(unid) * 0.8):] + unid_non_event[int(len(unid_non_event) * 0.8):]

    df = df[df["source"].isin(train_sources + val_sources + test_sources)].copy()

    start_date = pd.Timestamp("2025-01-01 00:00:00")
    df["Artificial_timestamp"] = [start_date + pd.Timedelta(days=int(i)) for i in range(df.shape[0])]

    handler = Dataset(
        data=df,
        datetime_column="Artificial_timestamp",
        train_sources=train_sources,
        val_sources=val_sources,
        test_sources=test_sources
    )

    if rul_sa == "rul":
        dataset, test_dataset = handler.get_rul_dataset(keep_sources="vehicle_id" if keep_identifiers else None)
    else:
        dataset, test_dataset = handler.get_SA_dataset(keep_sources="vehicle_id" if keep_identifiers else None)

    # Attach the non‑actionable column set to the dataset dicts
    dataset['non_actionable_columns'] = non_actionable_cols
    test_dataset['non_actionable_columns'] = non_actionable_cols

    return dataset, test_dataset

def generic(info):
    if "pbc2" in info["path"]:
        train_data, test_data = load_PBC(rul_sa="sa", path="Data/")
    else:
        df = pd.read_csv(info["path"])
        dataset_obj = Dataset(
            data=df,
            datetime_column=info["timestamp_column"],
            event_indicator='event',
            source_column='source',
            train_sources=info["train_sources"],
            val_sources=info["val_sources"],
            test_sources=info["test_sources"]
        )
        print("Preparing SA Dataset format...")
        train_data, test_data = dataset_obj.get_SA_dataset()

        # For non‑PBC datasets (like HNEI), the non‑actionable columns are given by exact names
        exact_names = info.get("non_actionable_features", set())
        # Ensure they are a set
        if isinstance(exact_names, (list, tuple)):
            exact_names = set(exact_names)
        # Attach to dataset dicts
        train_data['non_actionable_columns'] = exact_names
        test_data['non_actionable_columns'] = exact_names

    return train_data, test_data