import numpy as np
import matplotlib.pyplot as plt
import psychofit as psy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns


def fit_psychometric(block_df, x_col="signed_contrast", 
                     y_col="choice_right"):
    """
    Fit a psychometric curve.
    pars : fitted parameters: [bias, slope, lapse low, lapse high]
    xx : contrast grid.
    yy : P(right) in %.
    """
    df_fit = (
        block_df.groupby(x_col)
        .agg(
            ntrials=(y_col, "size"),
            frac=(y_col, "mean"),
        )
        .reset_index()
        .sort_values(x_col)
    )

    data_for_fit = df_fit[[x_col, "ntrials", "frac"]].to_numpy().T
    x_min = df_fit[x_col].min()
    x_max = df_fit[x_col].max()

    pars, L = psy.mle_fit_psycho(
        data_for_fit,
        P_model="erf_psycho_2gammas",
        parstart=np.array([0, 20.0, 0.05, 0.05]),
        parmin=np.array([x_min, 1.0, 0.0, 0.0]),
        parmax=np.array([x_max, 100.0, 1.0, 1.0]),
    )

    xx = np.linspace(x_min - 5, x_max + 5, 200)
    yy = psy.erf_psycho_2gammas(pars, xx) * 100  # in %
    return pars, xx, yy

def _normalize_palette(palette, group_values):
    """Accept a dict {group_value: color} or a sequence of colors aligned
    with group_values, and return a dict."""
    if isinstance(palette, dict):
        missing = [v for v in group_values if v not in palette]
        if missing:
            raise ValueError(f"palette is missing colors for: {missing}")
        return palette
    if len(palette) != len(group_values):
        raise ValueError("palette sequence must have the same length as group_values")
    return dict(zip(group_values, palette))
 
 
def _normalize_labels(labels, group_values):
    """Accept a dict {group_value: label}, a sequence aligned with
    group_values, or None (falls back to str(value))."""
    if labels is None:
        return {v: str(v) for v in group_values}
    if isinstance(labels, dict):
        return {v: labels.get(v, str(v)) for v in group_values}
    if len(labels) != len(group_values):
        raise ValueError("labels sequence must have the same length as group_values")
    return dict(zip(group_values, labels))
 
 
def subject_scatter(
    sub_df,
    x_col="signed_contrast",
    y_col="choice_right",
    subject_col="subject",
    jitter_scale=2,
):
    """
    Compute subject-level fraction-of-y_col for scatter points.
    x_scatter : jittered x_col values.
    y_scatter : subject-level mean of y_col, in percent.
    """
    df_scatter = (
        sub_df.groupby([subject_col, x_col])
        .agg(frac=(y_col, "mean"))
        .reset_index()
    )
 
    jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(df_scatter))
    x_scatter = df_scatter[x_col] + jitter
    y_scatter = df_scatter["frac"] * 100
    return x_scatter, y_scatter
 
 
def plot_two_curves_on_ax(
    ax,
    df,
    group_col,
    group_values,
    palette,
    labels=None,
    title="",
    title_color="black",
    x_col="signed_contrast",
    y_col="choice_right",
    subject_col="subject",
    show_ylabel=False,
    show_legend=False,
    add_subj_number=True,
    scatter=True
):
    """
    Plot pooled psychometric curves and subject-level scatter for two
    groups defined by `group_col` taking the two values in `group_values`.
 
    df : trial data for this panel (must contain group_col, x_col, y_col,
         subject_col).
    group_col : column defining which of the two curves a row belongs to.
    group_values : the two values of group_col to plot, e.g. ["Left", "Right"]
                   or ["easy", "hard"]. Order matters for plot_panel_with_inset
                   (it sets the sign of the delta).
    palette : dict {value: color} or 2-element sequence aligned with group_values.
    labels : optional dict {value: label} or 2-element sequence for the legend;
             defaults to str(value).
    """
    if df.empty:
        ax.set_axis_off()
        return
 
    if len(group_values) != 2:
        raise ValueError("group_values must contain exactly two values")
 
    palette = _normalize_palette(palette, group_values)
    labels = _normalize_labels(labels, group_values)
 
    n_subjs = df[subject_col].nunique()
 
    for value in group_values:
        sub_df = df[df[group_col] == value]
        if sub_df.empty:
            continue
 
        _, xx, yy = fit_psychometric(sub_df, x_col=x_col, y_col=y_col)
        ax.plot(xx, yy, color=palette[value], lw=2, label=labels[value])
        if scatter:
            x_scatter, y_scatter = subject_scatter(
                sub_df, x_col=x_col, y_col=y_col, subject_col=subject_col
            )
            ax.scatter(
                x_scatter,
                y_scatter,
                color=palette[value],
                s=10,
                alpha=0.25,
                edgecolor="none",
            )
            # One value per subject at each contrast
            subject_points = (
                sub_df.groupby([subject_col, x_col])
                .agg(p_right=(y_col, "mean"))
                .reset_index()
            )
            subject_points["p_right"] *= 100

            # Solid mean points + 95% CIs
            sns.lineplot(
                data=subject_points,
                x=x_col,
                y="p_right",
                ax=ax,
                estimator="mean",
                errorbar=("ci", 95),
                err_style="bars",
                n_boot=1000,
                seed=42,
                color=palette[value],
                marker="o",
                markersize=5,
                markeredgewidth=0,
                linestyle="None",
                err_kws={"capsize": 2, "linewidth": 1},
                legend=False,
            )
 
    ax.set_xlabel("Signed contrast (%)")
    ax.set_xlim([-110, 110])
    ax.set_ylim([0, 102])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, color=title_color)
 
    if show_ylabel:
        ax.set_ylabel("P(right) (%)")
    if show_legend:
        ax.legend(frameon=False, loc="upper left")
 
    if add_subj_number:
        ax.text(
            0.98,
            0.05,
            f"n = {n_subjs}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
 
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
 
def plot_panel_with_inset(
    ax,
    df_panel,
    group_col,
    group_values,
    palette,
    labels=None,
    title="",
    x_col="signed_contrast",
    y_col="choice_right",
    subject_col="subject",
    scatter=True,
):
    """
    Plot one group-level panel with:
      - main psychometric curves + scatter for the two groups in group_values
      - inset showing delta P(right) = group_values[1] - group_values[0],
        averaged across subjects, per x_col value.
    """
    if df_panel.empty:
        ax.set_axis_off()
        return
 
    if len(group_values) != 2:
        raise ValueError("group_values must contain exactly two values")
 
    labels = _normalize_labels(labels, group_values)
 
    plot_two_curves_on_ax(
        ax,
        df_panel,
        group_col=group_col,
        group_values=group_values,
        palette=palette,
        labels=labels,
        title=title,
        x_col=x_col,
        y_col=y_col,
        subject_col=subject_col,
        show_ylabel=False,
        show_legend=False,
        add_subj_number=False,
        scatter=scatter,
    )
 
    value_a, value_b = group_values  # delta = value_b - value_a
 
    df_frac = (
        df_panel.groupby([subject_col, group_col, x_col])
        .agg(frac=(y_col, "mean"))
        .reset_index()
    )
    wide = df_frac.pivot_table(
        index=[subject_col, x_col],
        columns=group_col,
        values="frac",
    )
 
    # need both groups present to compute a delta
    wide = wide.dropna(subset=[value_a, value_b])
    if wide.empty:
        return
 
    wide["delta"] = wide[value_b] - wide[value_a]
    df_delta = wide.reset_index()
 
    inset_ax = inset_axes(
        ax,
        width="26%",
        height="26%",
        loc="lower right",
        bbox_to_anchor=(0, 0.09, 1, 1),
        bbox_transform=ax.transAxes,
    )
 
    mean_delta = (
        df_delta.groupby(x_col)["delta"]
        .mean()
        .reset_index()
        .sort_values(x_col)
    )
    inset_ax.plot(
        mean_delta[x_col],
        mean_delta["delta"],
        linestyle="--",
        marker=".",
        linewidth=1,
        color="0.3",
    )
 
    inset_ax.set_xlabel("Signed contrast (%)", fontsize=7)
    inset_ax.set_ylabel(f"Δ ({labels[value_b]} - {labels[value_a]})", fontsize=7)
    inset_ax.tick_params(labelsize=7)
    inset_ax.set_xlim([-110, 110])
 
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)