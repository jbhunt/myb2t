from myb2t.model import BrainToTextDecoder
from myb2t.helpers import make_default_config, SubsetWithAttrs, seed_everything
import numpy as np
import copy
import polars as pl

def run_mtl_experiment(
    ds,
    config,
    alphas=[0, 0.5, 1.0],
    subset_size=0.3,
    train_size=0.8,
    split_seed=42,
    run_seeds=[1,],
    dst=None
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
    leftover_idxs = np.setdiff1d(all_idxs, train_idxs)
    test_idxs = rng.choice(leftover_idxs, size=n_test, replace=False)
    ds_test = SubsetWithAttrs(ds, test_idxs)

    #
    n_runs = len(run_seeds)
    scores = np.full([n_runs, len(alphas), 2], np.nan)
    n_sessions = len(alphas) * n_runs
    i_session = 0
    for i, s in enumerate(run_seeds):
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

    # Package results into a table
    n_alphas = len(alphas)
    df = pl.DataFrame({
        "run_index": np.repeat(np.arange(n_runs), n_alphas * 2),
        "run_seed":  np.repeat(np.array(run_seeds), n_alphas * 2),  # optional
        "alpha":     np.tile(np.repeat(np.array(alphas), 2), n_runs),
        "metric":    np.tile(np.array(["wer", "cer"]), n_runs * n_alphas),
        "score":     scores.reshape(-1),
    })

    # Save the scores to a table
    if dst is not None:
        df.write_csv(dst)

    return alphas, scores, df
    