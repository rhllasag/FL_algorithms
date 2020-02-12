from numpy import unique
import random
def balanced_sample_maker(X, y, examples_per_class, random_seed=None):
    """ return a balanced data set by oversampling minority class 
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}
    if not random_seed is None:
        random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    less=10000000
    level_less=-1
    exclude_list = []

    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
        if(less>len(obs_idx)):
          less=len(obs_idx)
          level_less=level
    # oversampling on observations of positive label
    sample_size = level_less
    print(len(groupby_levels[sample_size]))
    balanced_copy_idx=[]
    for level in range(0,len(uniq_levels)):
      if (examples_per_class*len(uniq_levels))<=len(groupby_levels[sample_size]):
        over_sample_idx = random.sample(groupby_levels[level], k=examples_per_class)
        balanced_copy_idx+=over_sample_idx
        random.shuffle(balanced_copy_idx)
    return balanced_copy_idx