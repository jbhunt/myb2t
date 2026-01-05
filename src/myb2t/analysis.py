from myb2t.model import BrainToTextDecoder
from myb2t.helpers import make_default_config, SubsetWithAttrs
import numpy as np

def tune_alpha(
    ds_train,
    config=None,
    alpha=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    max_iter=50,
    lr=0.00005,
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
    config["max_iter"] = max_iter
    config["lr"] = lr # Bump up the lr just a bit for a shorter training session
    config["early_stopping"] = False
    est = BrainToTextDecoder(config=config, out_dir=None)

    #
    all_idxs = np.arange(len(ds_train))
    train_idxs = np.random.choice(all_idxs, size=n_samples, replace=False)
    all_idxs = np.delete(all_idxs, train_idxs)
    test_idxs = np.random.choice(all_idxs, size=n_samples, replace=False)
    ds_train_small = SubsetWithAttrs(ds_train, train_idxs)
    ds_test_small = SubsetWithAttrs(ds_train, test_idxs)

    #
    scores = np.full([n_runs, len(alpha)], np.nan)

    #
    for i_run in range(n_runs):
        for i_a, a in enumerate(alpha):
            est.config["alpha"] = a
            est.fit(ds_train_small, new_model=True)
            wer = est.score(ds_test_small, print_progress=False)
            scores[i_run, i_a] = wer

    return alpha, scores
    