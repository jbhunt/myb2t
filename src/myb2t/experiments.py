from myb2t.model import BrainToTextDecoder
from myb2t.helpers import make_default_config, SubsetWithAttrs, seed_everything
import numpy as np

def run_mtl_experiment(
    ds_train,
    config,
    alphas=[0, 0.5, 1.0],
    n_samples=300,
    n_runs=1
    ):
    """
    """

    #
    if (n_samples * 2) > len(ds_train):
        raise Exception("N samples must be less than 2x the size of the dataset")

    #
    if config is None:
        config = make_default_config()

    # Force early stopping to be false
    config["early_stopping"] = False

    #
    all_idxs = np.arange(len(ds_train))
    train_idxs = np.random.choice(all_idxs, size=n_samples, replace=False)
    all_idxs = np.delete(all_idxs, train_idxs)
    test_idxs = np.random.choice(all_idxs, size=n_samples, replace=False)
    ds_train_small = SubsetWithAttrs(ds_train, train_idxs)
    ds_test_small = SubsetWithAttrs(ds_train, test_idxs)

    #
    scores = np.full([n_runs, len(alphas)], np.nan)
    seeds = np.arange(n_runs)
    for i, s in enumerate(seeds):
        seed_everything(s)
        est = BrainToTextDecoder(config=config, out_dir=None, verbosity=0)
        for j, a in enumerate(alphas):
            est.fit(ds_train_small, reset=True)
            wer = est.score(ds_test_small, print_progress=False)
            scores[i, j] = wer

    return alphas, scores
    