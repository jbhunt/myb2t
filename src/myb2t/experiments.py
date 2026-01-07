from myb2t.model import BrainToTextDecoder
from myb2t.helpers import make_default_config, SubsetWithAttrs, seed_everything
import numpy as np
import copy

def run_mtl_experiment(
    ds,
    config,
    alphas=[0, 0.5, 1.0],
    subset_size=0.3,
    train_size=0.8,
    n_runs=1,
    split_seed=42
    ):
    """
    """

    # Force early stopping to be false
    base_config = copy.deepcopy(config)
    base_config["early_stopping"] = False

    # Seed NumpPy's RNG so that I can reproduce/fix the train test split
    rng = np.random.default_rng(split_seed)

    # Compute the number of samples in each split
    n_all = int(round(subset_size * len(ds)))
    n_train = int(round(train_size * n_all))
    n_test  = n_all - n_train

    # Create training dataset
    all_idxs = np.arange(len(ds))
    train_idxs = rng.choice(all_idxs, size=n_train, replace=False)
    ds_train = SubsetWithAttrs(ds, train_idxs)

    # Create test dataset
    leftover_idxs = np.setdiff1d(all_idxs, train_idxs, assume_unique=False)
    test_idxs = np.random.choice(leftover_idxs, size=n_test, replace=False)
    ds_test = SubsetWithAttrs(ds, test_idxs)

    #
    scores = np.full([n_runs, len(alphas), 2], np.nan)
    seeds = [i + 1 for i in range(n_runs)]
    n_sessions = len(alphas) * n_runs
    i_session = 0
    for i, s in enumerate(seeds):
        seed_everything(s)
        est = BrainToTextDecoder(config=config, out_dir=None, verbosity=0)
        for j, a in enumerate(alphas):
            print(f"Running session {i_session + 1} out of {n_sessions}")
            est.config["alpha"] = a
            est.fit(ds_train, reset=True)
            wer, cer = est.score(ds_test, print_progress=False)
            scores[i, j, 0] = wer
            scores[i, j, 1] = cer
            i_session += 1
    print("All done!")

    return alphas, scores
    