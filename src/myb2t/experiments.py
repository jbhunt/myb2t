from myb2t.model import BrainToTextDecoder
from myb2t.helpers import make_default_config, SubsetWithAttrs, seed_everything
import numpy as np
import copy

def run_mtl_experiment(
    ds,
    config,
    alphas=[0, 0.5, 1.0],
    train_size=0.8,
    max_samples=300,
    n_runs=1,
    split_seed=42
    ):
    """
    """

    # Force early stopping to be false
    base_config = copy.deepcopy(config)
    base_config["early_stopping"] = False

    #
    rng = np.random.default_rng(split_seed)

    #
    n_train = int(round(train_size * len(ds)))
    n_test  = len(ds) - n_train
    if max_samples is not None:
        n_train = min(n_train, max_samples)
        n_test  = min(n_test,  max_samples)

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
    seeds = np.arange(n_runs)
    n_sessions = len(alphas) * n_runs
    for i, s in enumerate(seeds):
        print(f"Running session {i + 1} out of {n_sessions}")
        seed_everything(s)
        est = BrainToTextDecoder(config=config, out_dir=None, verbosity=0)
        for j, a in enumerate(alphas):
            est.config["alpha"] = a
            est.fit(ds_train, reset=True)
            wer, cer = est.score(ds_test, print_progress=False)
            scores[i, j, 0] = wer
            scores[i, j, 1] = cer
    print("All done!")

    return alphas, scores
    