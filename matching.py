# matching.py
# Functions for matching counties and scoring them.
# Needs: a scaled table of counties (one row per county, one column per feature),
# plus the cat_map and direction dictionaries.

import pandas as pd
import numpy as np


def match(z, cat_map, cat_weights, target, k=10):
    # z = the scaled table. cat_map = feature -> category.
    # cat_weights = category -> weight. target = the profile to match.
    # k = how many closest counties to return.

    # Count how many features are in each category.
    n_per_cat = pd.Series(cat_map).value_counts()

    # Each feature's weight = its category's weight divided by the number
    # of features in that category.
    w = pd.Series({f: cat_weights[c] / n_per_cat[c] for f, c in cat_map.items()})
    w = w.reindex(z.columns)   # put the weights in the same order as the columns

    # Weighted distance from every county to the target:
    # square the differences, multiply by the weights, add up, take the square root.
    d = np.sqrt((((z - target) ** 2) * w).sum(axis=1))

    # Return the k counties with the smallest distance (the closest matches).
    return d.nsmallest(k)


def quality_score(z, cat_map, direction, cat_weights):
    # Keep only features that have a direction (+1 or -1). Skip the 0s.
    scored = [f for f in z.columns if direction[f] != 0]

    # Multiply each kept feature by its direction. This flips the lower-is-better
    # features, so a higher number is always better.
    signed = z[scored] * pd.Series({f: direction[f] for f in scored})

    # Label each kept feature with its category.
    cats = pd.Series({f: cat_map[f] for f in scored})

    # Category score = average of that category's features, for each county.
    cat_scores = signed.T.groupby(cats).mean().T

    # Overall score = weighted average of the category scores.
    w = pd.Series(cat_weights)[cat_scores.columns]
    return (cat_scores * w).sum(axis=1) / w.sum()


def match_ideal(z, cat_map, direction, cat_weights, ideal, k=10):
    # ideal = the level the user wants in each category (in z units).
    # Keep only the features that have a direction (+1 or -1).
    scored = [f for f in z.columns if direction[f] != 0]

    # Each feature's target = its category's ideal level times its direction.
    target = pd.Series({f: ideal[cat_map[f]] * direction[f] for f in scored})

    # Reuse match() on the kept features only.
    cat_map_s = {f: cat_map[f] for f in scored}
    return match(z[scored], cat_map_s, cat_weights, target, k)