from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

from fals.utils.stats import filter_data, calculate_expression_diff

insitro_colors_wt = ["#2e898dff", "#136e72ff", "#81c4c7ff", "#abebeeff", "#bdd0d1ff", "#ff80ff"]
insitro_colors_disease = [
    "#ffd447",
    "#e6acc9",
    "#f9bf3aff",
    "#f8d177ff",
    "#fce5b0ff",
    "#fab20cff",
    "#f5d58bff",
]
insitro_alternating = [x for x in zip(insitro_colors_wt, insitro_colors_disease) for x in x]


def get_cell_line_palette(cell_line_pairs: Sequence[Tuple[str, str]]) -> dict[str, str]:
    """Generate cell lines color palette based on insitro colors"""
    cell_line_palette = {}

    for i, (wt, dis) in enumerate(cell_line_pairs):
        cell_line_palette[wt] = insitro_colors_wt[i % len(insitro_colors_wt)]
        cell_line_palette[dis] = insitro_colors_disease[i % len(insitro_colors_disease)]

    return cell_line_palette


def plot_dose_response(
    df: pd.DataFrame,
    feat: str,
    compound: str,
    by: Optional[str] = "cell_line",
    compound_control: str = "Dimethyl sulfoxide [MilliporeSigma]",
    concentrations: Optional[list[float]] = None,
    palette: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Dose response across concentrations with optional break out
    """

    df_compound = df[df.compound_name.isin([compound, compound_control])]
    if concentrations:
        df_compound = df_compound[df_compound.concentration_uM.isin([0.0] + concentrations)]
    # Rename control to compound with 0 concentration for simplifying hue
    df_compound.loc[:, "compound_name"].replace(compound_control, compound)
    # alternate wt/mut colors
    if not palette:
        palette = [x for x in zip(insitro_colors_wt, insitro_colors_disease) for x in x]
    fig = plt.figure()
    graph = sns.lineplot(
        data=df_compound.reset_index(),
        x="concentration_uM",
        y=feat,
        hue=by,
        palette=palette,
        marker="o",
    )
    graph.set(xscale="symlog")
    handles, labels = graph.get_legend_handles_labels()
    order = np.argsort(labels)
    graph.tick_params(axis="both", which="both", direction="in", labelleft=True)
    graph.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
    )
    return fig


def plot_line_pairs(
    df: pd.DataFrame,
    line_pairs: List[Tuple[str, str]],
    feat: str,
    compound: str,
    compound_control: str = "Dimethyl sulfoxide [MilliporeSigma]",
    concentrations: Optional[list[float]] = None,
    figsize: Tuple[int, int] = (40, 5),
) -> plt.Figure:
    """Plots paired dose response curves for each set of paired cell lines"""
    f, axes = plt.subplots(1, len(line_pairs), figsize=figsize, sharey="row")
    for i, (control_line, mutant_line) in enumerate(line_pairs):
        df_lines = df[df.cell_line.isin([control_line, mutant_line])]
        df_compound = df_lines[df_lines.compound_name.isin([compound, compound_control])]
        if concentrations:
            df_compound = df_compound[df_compound.concentration_uM.isin([0.0] + concentrations)]
        # Rename control to compound with 0 concentration for simplifying hue
        df_compound.loc[:, "compound_name"].replace(compound_control, compound)
        if len(line_pairs) > 1:
            ax = axes[i]
        else:
            ax = axes
        graph = sns.lineplot(
            data=df_compound.reset_index(),
            x="concentration_uM",
            y=feat,
            hue="cell_line",
            ax=ax,
            palette=get_cell_line_palette(cell_line_pairs=line_pairs),
            marker="o",
        )
        graph.set(xscale="symlog")
        handles, labels = ax.get_legend_handles_labels()
        order = np.argsort(labels)
        ax.tick_params(axis="both", which="both", direction="in", labelleft=True)
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
        )
    f.suptitle(compound, fontsize=18, y=1.25)
    return f


def boxplot_data(
    df_data: pd.DataFrame,
    feat: str,
    hue: str,
    compound: str,
    concentrations: Optional[float] = None,
) -> plt.Figure:
    """Plot boxplots for feature of a given dataframe grouped by a hue parameter.

    Parameters
    ----------
    df_data: Pandas DataFrame containing the data.
    feat: feature to be plotted.
    hue: The parameter to be used for grouping the data.
    compound: Compound to plot
    concentrations: Optional set of concentrations to plot.  Defaults to all

    Returns:
        Fiture
    """

    df_compound = df_data[df_data.compound_name == compound]
    if not concentrations:
        concentrations = sorted(set(df_compound.concentration_uM))
    f, ax = plt.subplots(1, 4, figsize=(40, 5), sharey="row")
    for i, concentration in enumerate(concentrations):
        df_concentration = df_compound[df_compound.concentration_uM == concentration].reset_index()
        hue_arg = hue
        if hue == "alive_in_well":
            hue_arg = ((df_concentration.alive_in_well // 100) + 1) * 100
        sns.boxplot(
            data=df_concentration,
            x="cell_line",
            y=feat,
            hue=hue_arg,
            showfliers=False,
            order=sorted(set(df_data.cell_line)),
            ax=ax[i],
        )
        ax[i].tick_params(axis="both", which="both", direction="in", labelleft=True)
        ax[i].title.set_text(f"{compound} / {concentration}")
        _ = ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
    return f


def plot_scatter(df, x, y, name, hue, hue_norm, title, plot_label=True):
    """Plot RNA imputation scatter plot with labels."""

    g = sns.scatterplot(data=df, x=x, y=y, hue=hue, hue_norm=hue_norm)
    pr = pearsonr(df[x], df[y])
    sr = spearmanr(df[x], df[y])

    def label_point(x, y, val, ax):
        a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point["x"] + 0.02, point["y"], str(point["val"]), size=10)

    if plot_label:
        label_point(df[x], df[y], df[name], plt.gca())
        x0, x1 = g.get_xlim()
        y0, y1 = g.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.plot(lims, lims, "-r")

    plt.title(title + f"\n Pearson corr={pr[0]} p={pr[1]}" + f"\n Spearman corr={sr[0]} p={sr[1]}")
    plt.show()

    return pr, sr


def plot_imputations(all_dfs, stressors, mutant_wt_pairs, perform_dict, plot_label):
    """Plot multiple RNA imputation scatter plot with labels."""

    results = []

    for stressor in stressors:
        for mutant, wt in mutant_wt_pairs:
            df2use = filter_data(all_dfs, stressor, mutant, wt)
            df = calculate_expression_diff(df2use, mutant, wt, perform_dict, stressor)

            if len(df) < 1:
                print("no data for", stressor, mutant, wt, "when assessing performance difference")
            else:
                t_out = plot_scatter(
                    df,
                    x="observed",
                    y="DINO-imputed",
                    name="gene",
                    hue="DINO_pred_performance",
                    hue_norm=(0, 1),
                    title=f"expr diff merged across donors: {stressor}-{mutant}-{wt}",
                    plot_label=plot_label,
                )
                results.append(
                    [stressor, mutant, t_out[0][0], t_out[0][1], t_out[1][0], t_out[1][1]]
                )

    return results
