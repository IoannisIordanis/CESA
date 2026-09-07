import numpy as np
import pandas as pd
from typing import List, Set
from collections import Counter
from utils.survival_utils import robust_corr

def group_features(window: pd.DataFrame, actionable_set: Set[str], threshold: float = 0.7) -> List[List[str]]:
    if not actionable_set:
        #print("Feature grouping: no actionable features")
        return []
    feat_list = list(actionable_set)
    data = window[feat_list]
    corr_mat = robust_corr(data)   # safe, no warnings
    corr = np.abs(corr_mat)
    remaining = set(range(len(feat_list)))
    groups = []
    while remaining:
        start = next(iter(remaining))
        group = {start}
        remaining.remove(start)
        changed = True
        while changed:
            changed = False
            to_add = []
            for i in remaining:
                if all(corr[i, j] >= threshold for j in group):
                    to_add.append(i)
            if to_add:
                changed = True
                for i in to_add:
                    group.add(i)
                    remaining.remove(i)
        groups.append([feat_list[i] for i in sorted(group)])
    sizes = [len(g) for g in groups]
    if sizes:
        size_counts = Counter(sizes)
        #print(f"Feature grouping: total features={len(feat_list)}, groups={len(groups)}, sizes: {dict(size_counts)}")
    # else:
    #  print("Feature grouping: no groups formed")
    return groups