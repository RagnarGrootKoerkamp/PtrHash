#!/usr/bin/env python3

import json
import pandas as pd
import matplotlib.pyplot as plt
from math import log, floor, ceil
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tabulate


def bucket_fn_plots():

    def linear(x):
        return x

    def optimal(x):
        return x + (1 - x) * log(1 - x)

    def skew(x):
        beta = 0.6
        gamma = 0.3
        if x < beta:
            return x * gamma / beta
        else:
            return gamma + (x - beta) * (1 - gamma) / (1 - beta)

    def square(x):
        return x * x

    def cubic(x):
        return x * x * (1 + x) / 2

    def invert(f, y):
        l = 0
        h = 1
        for _ in range(20):
            m = (l + h) / 2
            if f(m) < y:
                l = m
            else:
                h = m
        return l

    lmbda = 4

    def bucket_sz(f, y):
        y1 = y
        y2 = y + 0.001
        x1 = invert(f, y1)
        x2 = invert(f, y2)
        return lmbda * (x2 - x1) * 1000

    xs = [x / 1000 for x in range(1000)]

    ## PLOT 1

    # Set plot size
    plt.figure(figsize=(4.1, 2.7))
    # Plot the functions from 0 to 1
    plt.plot(xs, [linear(x) for x in xs], label="Linear  $\\,x$")
    plt.plot(xs, [skew(x) for x in xs], label="Skewed $0.6\\mapsto0.3$")
    plt.plot(xs, [optimal(x) for x in xs], label="Optimal $x + (1{-}x)\\ln(1{-}x)$")
    plt.plot(xs, [square(x) for x in xs], label="Square $x^2$")
    plt.plot(xs, [cubic(x) for x in xs], label="Cubic   $(x^3+x^2)/2$")
    # plt.legend()
    # Only show x and y ax
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    # x and y from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Normalized hash")
    plt.ylabel("Normalized bucket index")
    # Add horizontal grid lines
    plt.grid(axis="y", lw=0.5)

    plt.savefig("plots/bucket-fn.svg", bbox_inches="tight")
    plt.show()
    plt.close()

    ## PLOT 2

    # Set plot size
    plt.figure(figsize=(4.1, 2.7))
    plt.plot(xs, [bucket_sz(linear, x) for x in xs], label="Linear   $\\,x$")
    plt.plot(xs, [bucket_sz(skew, x) for x in xs], label="Skewed $0.6\\mapsto0.3$")
    plt.plot(
        xs,
        [bucket_sz(optimal, x) for x in xs],
        label="Optimal $x + (1{-}x)\\ln(1{-}x)$",
    )
    plt.plot(xs, [bucket_sz(square, x) for x in xs], label="Square  $x^2$")
    plt.plot(xs, [bucket_sz(cubic, x) for x in xs], label="Cubic    $(x^3+x^2)/2$")
    plt.legend()
    # Only show x and y ax
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    # x and y from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 11.5)
    plt.xlabel("Normalized bucket index")
    plt.ylabel("Expected bucket size for $\\lambda = 4$")
    # Add horizontal grid lines
    plt.grid(axis="y", lw=0.5, which="major")
    plt.grid(axis="y", lw=0.5, which="minor", alpha=0.4)
    # Add minor tickes lines every 1
    plt.yticks(range(0, 12, 1), minor=True)

    plt.savefig("plots/bucket-size.svg", bbox_inches="tight")
    plt.show()
    plt.close()


def build_stats():
    out = "plots/bucket_fn_stats.svg"
    f1 = "data/bucket_fn_stats_l35.json"
    f2 = "data/bucket_fn_stats_l40.json"

    alpha = 0.99

    with open(f1) as f:
        data1 = json.load(f)
    with open(f2) as f:
        data2 = json.load(f)

    rows = 2
    cols = 3

    pcts = range(100)
    xs = [x / 100 for x in range(0, 101)]

    # Figure with 3 subplots
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 2.7 + 0.75, rows * 2.2), layout="constrained"
    )
    if cols == 1:
        axs = [axs]

    keys = ["linear", "skewed", "optimal", "square", "cubic"]
    all_keys = [(k, 3.5) for k in keys] + [("cubic", 4.0)]

    flat_ax = [ax for row in axs for ax in row]

    for i, (ax1, (name, l)) in enumerate(zip(flat_ax, all_keys)):
        if l == 3.5:
            data = data1[name]["by_pct"]
        else:
            data = data2[name]["by_pct"]

        elem = 0
        elems = [0]
        for pct in pcts:
            elem += data[pct]["elements"]
            elems.append(elem)
        slots = elems[-1] / alpha
        load = [x / slots for x in elems]

        # Set plot size
        ax3 = ax1.twinx()
        ax2 = ax1.twinx()

        ax1.set_ylim(0, 1.0)
        ax3.set_ylim(0, 1)
        ax2.set_ylim(0, 20)
        ax2.set_yticks([0, 4, 8, 12, 16, 20], minor=False)

        ax1.set_title(name.capitalize() + f", $\\lambda = {l}$")

        if i >= 3:
            ax1.set_xlabel("Normalized bucket index")
        ax1.set_ylabel("Evictions per bucket")
        ax2.set_ylabel("Bucket size")
        ax2.yaxis.label.set_color("red")
        ax1.xaxis.set_minor_locator(MultipleLocator(0.1))

        p1 = ax1.plot(
            xs[1:],
            [data[pct]["evictions"] / data[pct]["buckets"] for pct in pcts],
            label="Evictions per bucket",
            color="black",
        )
        # Plot load factor
        p3 = ax3.fill(xs + [1], load + [0], color="red", alpha=0.1, label="Load factor")
        ax3.yaxis.set_visible(False)

        # Plot bucket size on secondary axis
        p2 = ax2.plot(
            xs[1:],
            [data[pct]["elements"] / data[pct]["buckets"] for pct in pcts],
            color="red",
            label="Bucket size",
        )

        ax1.set_zorder(ax3.get_zorder() + 1)
        ax1.set_frame_on(False)

        # Only show x and y ax
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.spines["bottom"].set_visible(False)
        ax3.spines["left"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        # x and y from 0 to 1
        ax1.set_xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.xlabel("Normalized hash")
        # plt.ylabel("Normalized bucket index")
        # Add horizontal grid lines
        ax1.grid(axis="y", lw=0.5)

        # Keep only leftmost and rightmost y-axis
        # First
        if i % 3 == 0:
            ax1.yaxis.set_visible(True)
        else:
            ax1.set_yticklabels([])
            ax1.yaxis.set_ticks_position("none")
            ax1.set_ylabel(None)
        # Last
        if i % 3 == 2:
            ax2.yaxis.set_visible(True)
        else:
            ax2.yaxis.set_visible(False)
        if i == 2:
            ax1.legend(handles=p1 + p2 + p3, loc="best")

    plt.savefig(out, bbox_inches="tight")
    plt.show()
    plt.close()


def space(f, out):
    with open(f) as f:
        data = json.load(f)
    # Convert json to dataframe
    df = pd.DataFrame(data)

    df["c"] = 1000000000 * df["construction_6"] / df["n"]

    # A plot with lambda on the x-axis, and two y-axes with build time and size.
    # Create 6 lines, for 3 different alpha values and 2 different bucket functions.
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5), layout="constrained")
    ax2 = ax.twinx()

    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Construction time (ns/key)")
    ax2.set_ylabel("Size (bits/key)")

    alpha_size = {
        0.998: 0.8,
        0.995: 1.4,
        0.99: 2.0,
        0.98: 2.6,
    }

    alpha_style = {
        0.998: "solid",
        0.995: "dotted",
        0.99: "dashed",
        0.98: "solid",
    }

    groups = df.groupby(["n", "alpha", "bucketfn", "real_alpha"])
    for k, g in groups:
        n, alpha, bucketfn, real_alpha = k
        ls = alpha_style[alpha]
        lw = alpha_size[alpha]
        a = 0.8
        if bucketfn == "Linear":
            color = "blue"
        else:
            color = "red"
        ax.plot(g["lambda"], g["c"], label=f"{k}", ls=ls, color=color, lw=lw, alpha=a)

    ls = df["lambda"].unique()
    ys = [8 / l for l in ls]
    ax2.plot(ls, ys, lw=1, label="Space (pilots)", color="black")
    # Plot space with remap
    for alpha in df.alpha.unique():
        lw = alpha_size[alpha]
        lst = alpha_style[alpha]
        ax2.plot(
            ls,
            [8 / l + 32 * (1 / alpha - 1) for l in ls],
            lw=lw,
            ls=lst,
            color="green",
            alpha=0.9,
        )
        if alpha < 0.995:
            ax2.plot(
                ls,
                [8 / l + (512 / 44) * (1 / alpha - 1) for l in ls],
                lw=lw,
                ls=lst,
                color="orange",
                alpha=0.9,
            )

    # Add dots for the fast configuration
    ms = 11
    l = 3.0
    a = 0.99
    y = df[(df.alpha == a) & (df.bucketfn == "Linear") & (df["lambda"] == l)].c
    ax.plot(l, y, "o", ms=ms, color="blue")
    y = 8 / l + 32 * (1 / a - 1)
    ax2.plot(l, y, "o", ms=ms, color="blue", mec="blue")

    # Add dots for the default configuration
    l = 3.5
    a = 0.99
    y = df[(df.alpha == a) & (df.bucketfn == "CubicEps") & (df["lambda"] == l)].c
    ax.plot(l, y, "o", ms=ms, color="black")
    y = 8 / l + (512 / 44) * (1 / a - 1)
    # Red outline to marker
    ax2.plot(l, y, "o", ms=ms, color="black", mec="black")

    # Add dots for the compact configuration
    l = 4.0
    a = 0.99
    y = df[(df.alpha == a) & (df.bucketfn == "CubicEps") & (df["lambda"] == l)].c
    ax.plot(l, y, "o", ms=ms, color="red")
    y = 8 / l + (512 / 44) * (1 / a - 1)
    # Red outline to marker
    ax2.plot(l, y, "o", ms=ms, color="red", mec="red")

    ax.set_xlim(2.7, 4.20)
    ax.set_ylim(0, 140)
    ax2.set_ylim(0, 3.5)
    ax2.grid(axis="y", lw=0.5)
    ax.grid(axis="y", lw=0.5)
    ax.grid(axis="x", lw=0.5)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.set_yticks([0, 2.0, 2.5, 3.0], minor=False)
    ax2.set_yticks([2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9], minor=True)
    ax2.grid(axis="y", lw=0.5, which="minor", alpha=0.4)

    # Build legend
    llinear = mpatches.Patch(color="blue", label="Linear $\\gamma$")
    lcubic = mpatches.Patch(color="red", label="Cubic $\\gamma$")
    lpilots = Line2D([0], [0], label="Pilots space", lw=2, color="black")
    lfill = Line2D([], [], label="", lw=0)
    lvec = Line2D([0], [0], label="Total space, Vec<u32>", lw=2, color="green")
    lclef = Line2D([0], [0], label="Total space, CLEF", lw=2, color="orange")
    la = []
    for a in sorted(df.alpha.unique(), reverse=True):
        lw = alpha_size[a]
        ls = alpha_style[a]
        l = Line2D([0], [0], label=f"$\\alpha = {a}$", lw=lw, ls=ls, color="black")
        la.append(l)
    plt.legend(
        handles=[llinear, lcubic, lfill, lpilots, lvec, lclef] + la,
        loc="lower right",
        ncols=5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.savefig(out, bbox_inches="tight")
    plt.show()
    plt.close()


def remap(f):
    with open(f) as f:
        data = json.load(f)
    # Convert json to dataframe
    df = pd.DataFrame(data)

    # For for-loop variants, take min over with/without black_box() to prevent SIMD.
    df["q1_phf"] = df[["q1_phf", "q1_phf_bb"]].min(axis=1)
    df["q1_mphf"] = df[["q1_mphf", "q1_mphf_bb"]].min(axis=1)

    # Print dataframe with two digits precision.
    df = df[
        [
            "alpha",
            "lambda",
            "bucketfn",
            "pilots",
            "remap_type",
            "remap",
            "q1_phf",
            "q32_phf",
            "q1_mphf",
            "q32_mphf",
        ]
    ]

    print(tabulate.tabulate(df, headers=df.columns, tablefmt="orgtbl", floatfmt=".2f"))


def query_batching(f, out):
    with open(f) as f:
        data = json.load(f)
    # Convert json to dataframe
    df = pd.DataFrame(data)

    # A plot with lambda on the x-axis, and two y-axes with build time and size.
    # Create 6 lines, for 3 different alpha values and 2 different bucket functions.
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 4), layout="constrained")

    for ax in axs:
        ax.set_xlabel("Batch or lookahead size")
    axs[0].set_ylabel("Query throughput (ns/key)")

    df.loc[df["mode"] == "loop_bb", "mode"] = "loop"
    groups = df.groupby(["n", "alpha", "lambda", "bucketfn", "mode"])
    for k, g in groups:
        n, alpha, lmbda, bucketfn, mode = k
        ls = "solid"
        lw = 1
        a = 0.8
        ax = axs[1]
        if n < 10**9:
            # a = 0.5
            # lw += 1
            ax = axs[0]
        if lmbda == 3.0:
            color = "blue"
        elif lmbda == 3.5:
            color = "black"
        else:
            color = "red"
        if mode == "stream":
            ls = "dashed"
        if mode == "batch":
            # Batch2 has more consistent performance.
            ls = "dashdot"
            continue
        if mode == "batch2":
            ls = "dotted"
        ax.plot(g["batch_size"], g["q_phf"], ls=ls, color=color, lw=lw, alpha=a)

        if mode == "loop":
            # Horizontal line at height g['q_phf']
            # Min over loop and loop_bb
            assert len(g) == 2, len(g)
            ax.axhline(y=g["q_phf"].values.min(), color=color, lw=lw, ls=ls)
        if n == 10**9:
            ax.set_title(f"$n = 10^9$")
        if n == 20 * 10**6:
            ax.set_title(f"$n = 20$ million")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
        ax.set_xticks([1, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112], minor=True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
        ax.xaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: None))
        ax.set_xlim(1, 64)
        ax.grid(axis="y", lw=0.5)
        ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    axs[0].set_ylim(0, 21)
    axs[1].set_ylim(0, 21)
    axs[1].axhline(y=7.4, color="black", lw=0.5)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Build legend
    lcompact = mpatches.Patch(color="red", label="Cubic, Default/Compact")
    lsimple = mpatches.Patch(color="blue", label="Linear, Fast")
    # ldefault = mpatches.Patch(color="black", label="Default")
    lloop = Line2D([0], [0], label="Loop", lw=2, color="black", ls="solid")
    lbatch = Line2D([0], [0], label="Batch", lw=2, color="black", ls="dotted")
    lstream = Line2D([0], [0], label="Stream", lw=2, color="black", ls="dashed")
    lfill = Line2D([], [], label="", lw=0)
    axs[0].legend(
        handles=[lcompact, lsimple, lfill, lloop, lbatch, lstream],
        loc="upper left",
        ncols=2,
    )

    lopt1 = Line2D(
        [0], [0], label="Single-core RAM throughput", lw=0.5, color="black", ls="solid"
    )
    axs[1].legend(
        handles=[lopt1],
        loc="lower right",
    )

    plt.savefig(out, bbox_inches="tight")
    plt.show()
    plt.close()


class MulticolorPatch(object):
    def __init__(self, label, styles):
        self.styles = styles
        self.label = label

    def get_label(self):
        return self.label


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, style in enumerate(orig_handle.styles):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.styles) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.styles),
                    height,
                    edgecolor="none",
                    linewidth=0,
                    **style,
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def query_throughput(f, out):
    with open(f) as f:
        data = json.load(f)
    # Convert json to dataframe
    df = pd.DataFrame(data)

    # A plot with lambda on the x-axis, and two y-axes with build time and size.
    # Create 6 lines, for 3 different alpha values and 2 different bucket functions.
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 4), layout="constrained")

    print("Fastest: ", df["q_phf"].min())

    aa = 0.4

    for ax in axs:
        ax.set_xlabel("#threads")
    axs[0].set_ylabel("Query throughput (ns/key)")

    df.loc[df["mode"] == "loop_bb", "mode"] = "loop"
    groups = df.groupby(["n", "alpha", "lambda", "bucketfn", "mode", "remap_type"])
    for k, g in groups:
        n, alpha, lmbda, bucketfn, mode, remap_type = k
        ls = "solid"
        lw = 2
        ax = axs[int(n == 10**9)]

        if bucketfn == "CubicEps" and remap_type != "CacheLineEF":
            continue

        # if n < 10**9:
        #     continue
        #     ax = axs[0]
        if lmbda == 3.0:
            c1 = "blue"
        elif lmbda == 3.5:
            c1 = "black"
        else:
            c1 = "red"
        if mode == "stream":
            ls = "dashed"
        if mode == "batch":
            ls = "dotted"
        if mode == "loop_bb":
            ls = "dashdot"

        # Take min over loop and loop_bb
        g = g.groupby(["threads"]).min().reset_index()

        ax.plot(g["threads"], g["q_phf"], ls=ls, color=c1, lw=2, alpha=aa)
        ax.plot(g["threads"], g["q_mphf"], ls=ls, color=c1, lw=1.5, alpha=1.0)
        if n == 10**9:
            ax.set_title(f"$n = 10^9$")
        if n == 20 * 10**6:
            ax.set_title(f"$n = 20$ million")

    xs = [1, 2, 3, 4, 5, 6, 7]
    for n, ax in zip(sorted(df.n.unique()), axs):
        g = df[(df["n"] == n) & (df["threads"] == 1)]
        g = list(g["q_phf"]) + list(g["q_mphf"])
        # Fill light grey between the two above plots.
        # ax.fill_between(
        #     xs,
        #     [min(g) / x for x in xs],
        #     [max(g) / x for x in xs],
        #     color="grey",
        #     alpha=0.1,
        #     edgecolor="none",
        #     lw=0,
        # )

    for ax in axs:
        ax.grid(axis="y", lw=aa)
        ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        ax.set_ylim(0, 13.9)
        ax.set_xlim(0.8, 6.2)

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[1].axhline(y=2.5, color="black", lw=1, ls="dotted", zorder=-1)
    axs[1].plot(
        xs,
        [7.4 / x for x in xs],
        color="black",
        ls="solid",
        lw=0.5,
    )

    # Build legend
    # TODO: Legend for default black params
    lcompact = MulticolorPatch(
        "Cubic, Default/Compact",
        [{"color": "red", "alpha": aa}, {"color": "red", "alpha": 1.0}],
    )
    lsimple = MulticolorPatch(
        "Linear, Fast",
        [{"color": "blue", "alpha": aa}, {"color": "blue", "alpha": 1.0}],
    )
    lphf = MulticolorPatch(
        "PHF", [{"color": "red", "alpha": aa}, {"color": "blue", "alpha": aa}]
    )
    lmphf = MulticolorPatch(
        "MPHF", [{"color": "red", "alpha": 1.0}, {"color": "blue", "alpha": 1.0}]
    )
    lloop = Line2D([0], [0], label="Loop", lw=1.5, color="black", ls="solid")
    lstream = Line2D([0], [0], label="Stream", lw=2, color="black", ls="dashed")
    # lscaling = mpatches.Patch(color="black", label="Perfect scaling", alpha=0.1, lw=0)
    axs[0].legend(
        handles=[lcompact, lphf, lloop, lsimple, lmphf, lstream],
        loc="upper right",
        ncols=2,
        handler_map={MulticolorPatch: MulticolorPatchHandler()},
    )

    lopt = Line2D(
        [0], [0], label="Total RAM throughput", lw=1, color="black", ls="dotted"
    )
    lopt1 = Line2D(
        [0], [0], label="Single-core RAM throughput", lw=0.5, color="black", ls="solid"
    )
    axs[1].legend(
        handles=[lopt1, lopt],
        loc="upper right",
    )

    plt.savefig(out, bbox_inches="tight")
    plt.show()


def string_queries(f):
    with open(f) as f:
        data = json.load(f)
    # Convert json to dataframe
    df = pd.DataFrame(data)

    df = df[df.bucketfn == "CubicEps"]

    df = df.pivot_table(
        "q_mphf", ["hash", "input_type", "bucketfn"], "mode", sort=False
    )
    # df["loop"] = df[["loop_bb"]].min(axis=1)
    df["loop"] = df[["loop", "loop_bb"]].min(axis=1)
    df = df.reset_index()
    df = df.pivot_table(["loop", "stream"], "input_type", "hash", sort=False)
    # df = df.sort_index(axis=1, level=1)

    print(df)
    print()
    df = df.reset_index()
    print(tabulate.tabulate(df, headers=df.columns, tablefmt="orgtbl", floatfmt=".1f"))


def comparison(f):
    with open(f) as f:
        text = f.read()
    # Get all lines starting with 'RESULT'
    prefix = "RESULT "
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    # Remove 'RESULT ' from the start of each line
    lines = [line[len(prefix) :] for line in lines]

    # Split by space, then split by = to build a dictionary.
    def mapline(line):
        x = line.find("bitsPerElement")
        head = line[len("name=") : x - 1]
        d = dict([x.split("=") for x in line[x:].split()])
        d["name"] = head
        return d

    data = [mapline(line) for line in lines]
    # Convert to dataframe
    df = pd.DataFrame(data)

    assert df.loadFactor.unique() == ["1"]
    assert df.queryThreads.unique() == ["1"]
    assert df.N.nunique() == 1
    n = int(df.N.unique()[0])
    # convert to float
    df["build"] = df["constructionTimeMilliseconds"].astype(float) * 1000000 / n

    assert df.numQueriesTotal.nunique() == 1
    m = int(df.numQueriesTotal.unique()[0])
    df["query"] = df["queryTimeMilliseconds"].astype(float) * 1000000 / m
    df["size"] = df["bitsPerElement"].astype(float)

    # 'threads'
    df = df[["name", "size", "build", "query"]]
    df["prod"] = df["build"] * df["query"]
    df = df.sort_values("prod")

    print(df)
    print(
        tabulate.tabulate(
            df,
            headers=df.columns,
            tablefmt="orgtbl",
            floatfmt=[None, None, ".2f", ".0f", ".0f", ".0f"],
        )
    )


plt.close("all")

# 3.4
# bucket_fn_plots()

# 4.1.1
# build_stats()

# 4.1.2
space("data/size.json", "plots/size.svg")

# 4.1.2
# remap("data/remap.json")

# 4.2.1
# query_batching("data/query_batching.json", "plots/query_batching.svg")

# 4.2.2
# query_throughput("data/query_throughput.json", "plots/query_throughput.svg")

# appendix
# string_queries("data/string_queries.json")

# 4.3
# comparison("data/comparison_3e8.txt")
