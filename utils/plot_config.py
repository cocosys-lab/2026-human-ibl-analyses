# plot_config.py

import matplotlib.pyplot as plt
import seaborn as sns

GROUP_NAMES = ['Instructed', 'Uninstructed'] # or Discovery

EXAMPLE_SESSIONS = {
    GROUP_NAMES[0]: 94, # instructions
    GROUP_NAMES[1]: 27 # no instructions
}

# Named colors, keyed for semantic use
COLORS = {
    "left_block": "#5D3B92",
    "right_block": "#EF6528",
    GROUP_NAMES[0]: "#3B6F92", # instructions
    GROUP_NAMES[1]: "#FF9AE6", # no instructions
    'correct': 'darkgreen',
    'incorrect': 'firebrick',
    'cursor_early_trial': 'yellow',
    'cursor_late_trial': 'limegreen'
}

# Ordered palettes
CONTINUOUS_PALETTE = sns.color_palette("viridis", as_cmap=True)
DIVERGING_PALETTE = sns.color_palette("coolwarm", as_cmap=True)
CONTRAST_PALETTE = sns.color_palette('Greys', n_colors=10)[-5:]

# Font sizes, centralized
FONT_SIZES = {
    "title": 16,
    "label": 13,
    "tick": 11,
    "legend": 11,
    "annotation": 10,
}

FONT_FAMILY = "Arial"  

def apply_style():
    """Call this once at the top of a script/notebook."""
    # sns.set_theme(context=context, style=style, palette=PALETTE)
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "axes.facecolor": "white",
        "figure.figsize": (8, 5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })