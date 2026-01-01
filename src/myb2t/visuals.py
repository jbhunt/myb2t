from myb2t import datasets
from matplotlib import pyplot as plt

def visualize_datasets(root_b2t, root_opus, n_seqs=100, cmap_chr="tab20", cmap_pho="tab20", cmap_neural="binary", figsize=(11, 5)):
    """
    """

    #
    ds_b2t = datasets.BrainToText2025(root_b2t, T=int(30 / 0.02), split="train")
    ds_b2t.load(p_skip=0.98)
    Z_chr_b2t = ds_b2t.y_2[:n_seqs]
    Z_pho_b2t = ds_b2t.y_1[:n_seqs]
    Z_spk = ds_b2t.X[0, :, :, 0].T
    Z_lfp = ds_b2t.X[0, :, :, 1].T
    ds_opus = datasets.OpusDataset(root_opus)
    ds_opus.load(n_seqs=n_seqs)
    Z_chr_opus = ds_opus.X

    #
    fig, axs = plt.subplots(ncols=5)
    axs[0].pcolor(Z_chr_b2t, cmap=cmap_chr)
    axs[1].pcolor(Z_chr_opus, cmap=cmap_chr)
    axs[2].pcolor(Z_pho_b2t, cmap=cmap_pho)
    axs[3].pcolor(Z_spk, cmap=cmap_neural)
    axs[4].pcolor(Z_lfp, cmap=cmap_neural)

    #
    titles = (
        r"$Characters_{B2T}$",
        r"$Characters_{OPUS}$",
        r"$Phonemes_{B2T}$",
        r"$Spikes$",
        r"$LFP$"
    )
    ylabels = (
        "Sequence index",
        "Sequence index",
        "Sequence index",
        "Channel index",
        "Channel index"
    )
    xlabels = (
        "Tokens",
        "Tokens",
        "Tokens",
        "Time (20 ms bins)",
        "Time (20 ms bins)"
    )
    for ax, title, ylabel, xlabel in zip(axs, titles, ylabels, xlabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    #
    for ax in (axs[1], axs[2], axs[4]):
        ax.set_ylabel("")
        ax.set_yticklabels([])

    #
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()

    return fig