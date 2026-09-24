# plot_config.py

import matplotlib.pyplot as plt
import seaborn as sns

EXAMPLE_SESSIONS = {
    'instructions':'094', 
    'no_instructions': '010'
}

# Named colors, keyed for semantic use
COLORS = {
    "left_block": "#5D3B92",
    "right_block": "#EF6528",
    "instructions": "mediumturquoise",
    "no_instructions": "palevioletred",
    'correct': 'darkgreen',
    'incorrect': 'firebrick'
}

# Ordered palettes
CONTINUOUS_PALETTE = sns.color_palette("viridis", as_cmap=True)
DIVERGING_PALETTE = sns.color_palette("coolwarm", as_cmap=True)

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