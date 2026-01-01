from myb2t import datasets
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
import glasbey as gb
import numpy as np

def visualize_datasets(
    root_b2t,
    root_opus,
    n_seqs=30,
    figsize=(11, 7),
    trial_index=0
    ):
    """
    Visualize datasets
    """

    # Load datasets
    ds_b2t = datasets.BrainToText2025(root_b2t, T=int(30 / 0.02), split="train")
    ds_b2t.load(p_skip=0.98)
    Z_chr_b2t = ds_b2t.y_2[:n_seqs]
    Z_pho_b2t = ds_b2t.y_1[:n_seqs]
    Z_spk = ds_b2t.X[trial_index, :, :, 0].T
    Z_lfp = ds_b2t.X[trial_index, :, :, 1].T
    ds_opus = datasets.OpusDataset(root_opus)
    ds_opus.load(n_seqs=n_seqs)
    Z_chr_opus = ds_opus.X

    # 
    chr_ids = np.array(sorted(ds_b2t.v_chr._out_map.keys()), dtype=int)
    pho_ids = np.array(sorted(ds_b2t.v_pho.map.keys()), dtype=int)

    chr_labels = [ds_b2t.v_chr._out_map[int(k)] for k in chr_ids]
    pho_labels = [ds_b2t.v_pho.map[int(k)] for k in pho_ids]

    # 
    chr_to_idx = {int(tid): i for i, tid in enumerate(chr_ids)}
    pho_to_idx = {int(tid): i for i, tid in enumerate(pho_ids)}

    # Remap arrays for plotting ONLY
    Z_chr_b2t_plot = np.vectorize(lambda t: chr_to_idx.get(int(t), 0))(Z_chr_b2t)
    Z_chr_opus_plot = np.vectorize(lambda t: chr_to_idx.get(int(t), 0))(Z_chr_opus)
    Z_pho_b2t_plot = np.vectorize(lambda t: pho_to_idx.get(int(t), 0))(Z_pho_b2t)

    n_chr_tok = len(chr_ids)
    n_pho_tok = len(pho_ids)

    #
    chr_pal_lst = gb.extend_palette("tab20", palette_size=n_chr_tok)
    pho_pal_lst = gb.extend_palette("tab20", palette_size=n_pho_tok)
    cmap_chr = ListedColormap(chr_pal_lst, name="chr_tokens")
    cmap_pho = ListedColormap(pho_pal_lst, name="pho_tokens")

    #
    norm_chr = BoundaryNorm(np.arange(-0.5, n_chr_tok + 0.5, 1), ncolors=n_chr_tok)
    norm_pho = BoundaryNorm(np.arange(-0.5, n_pho_tok + 0.5, 1), ncolors=n_pho_tok)

    # Figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=3,
        ncols=6,
        height_ratios=[20, 2.2, 2.2],      # a bit taller so labels fit
        width_ratios=[1, 1, 1, 0.15, 1, 1],
        hspace=0.65,
        wspace=0.25,
    )

    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 4]),  # spikes
        fig.add_subplot(gs[0, 5]),  # lfp
    ]

    #
    m_chr_b2t  = axs[0].pcolor(Z_chr_b2t_plot, cmap=cmap_chr, norm=norm_chr)
    m_chr_opus = axs[1].pcolor(Z_chr_opus_plot, cmap=cmap_chr, norm=norm_chr)
    m_pho_b2t  = axs[2].pcolor(Z_pho_b2t_plot, cmap=cmap_pho, norm=norm_pho)
    m_spk      = axs[3].pcolor(Z_spk, cmap="binary")
    m_lfp      = axs[4].pcolor(Z_lfp, cmap="binary")

    #
    titles = (
        r"$Characters_{B2T}$",
        r"$Characters_{OPUS}$",
        r"$Phonemes_{B2T}$",
        r"$Spikes$",
        r"$LFP$"
    )
    ylabels = ("Sequence index", "Sequence index", "Sequence index", "Channel index", "Channel index")
    xlabels = ("Tokens", "Tokens", "Tokens", "Time (20 ms bins)", "Time (20 ms bins)")

    for ax, title, ylabel, xlabel in zip(axs, titles, ylabels, xlabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    for ax in (axs[1], axs[2], axs[4]):
        ax.set_ylabel("")
        ax.set_yticklabels([])

    # Colorbars
    cax_chr = fig.add_subplot(gs[1, :])
    cax_pho = fig.add_subplot(gs[2, :])
    chr_ticks = np.arange(n_chr_tok)   # centers of blocks in index space
    pho_ticks = np.arange(n_pho_tok)
    cb_chr = fig.colorbar(
        m_chr_b2t,
        cax=cax_chr,
        orientation="horizontal",
        ticks=chr_ticks,
        spacing="proportional",
        drawedges=False
    )
    cb_chr.set_label("Character token", labelpad=10)
    cb_chr.set_ticklabels(chr_labels)
    cb_pho = fig.colorbar(
        m_pho_b2t,
        cax=cax_pho,
        orientation="horizontal",
        ticks=pho_ticks,
        spacing="proportional",
        drawedges=False
    )
    cb_pho.set_label("Phoneme token", labelpad=10)
    cb_pho.set_ticklabels(pho_labels)

    # Tick styling: centered under blocks
    for cb in (cb_chr, cb_pho):
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.set_ticks_position("bottom")  # keep ticks/labels below
        cb.ax.tick_params(axis="x", length=0, pad=4)
        for lbl in cb.ax.get_xticklabels():
            lbl.set_rotation(90)
            lbl.set_ha("center")
            lbl.set_va("top")
        for spine in cb.ax.spines.values():
            spine.set_visible(False)

        # remove tick marks (but keep labels)
        cb.ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)
        cb.ax.tick_params(axis="x", labelbottom=True, labeltop=False, labelrotation=90)

        # remove divider lines between categories (these often look like ticks)
        if hasattr(cb, "solids") and cb.solids is not None:
            cb.solids.set_edgecolor("face")
            cb.solids.set_linewidth(0)
    
    # Adjust layout
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.15
    )

    return fig

