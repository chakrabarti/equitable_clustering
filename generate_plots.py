import argparse
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("--k", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=str,
        default=["adult", "bank", "creditcard", "diabetes", "census1990"],
    )
    parser.add_argument("--output_directory", default="output")

    args = parser.parse_args()
    for dataset in args.dataset:
        for k in args.k:
            filename = (
                f"{args.output_directory}/{dataset}/{dataset}_{k}_experiment_output.pkl"
            )
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    d = pickle.load(f)

                analyses = []
                max_radius_file = (
                    f"{args.output_directory}/{dataset}/{dataset}_max_radius_file"
                )
                max_ag_file = f"{args.output_directory}/{dataset}/{dataset}_max_ag_file"
                max_pp_file = f"{args.output_directory}/{dataset}/{dataset}_max_pp_file"
                ag_hist_file = (
                    f"{args.output_directory}/{dataset}/{dataset}_ag_hist_file"
                )
                pp_hist_file = (
                    f"{args.output_directory}/{dataset}/{dataset}_pp_hist_file"
                )
                k_file = f"{args.output_directory}/{dataset}/{dataset}_k_comparison"

                scalar_files = [
                    max_radius_file,
                    max_ag_file,
                    max_pp_file,
                ]

                for file in scalar_files:
                    if k == 2:
                        with open(file, "w") as f:
                            f.write("k,analysis,value\n")

                hist_files = [ag_hist_file, pp_hist_file]

                for file in hist_files:
                    if k == 2:
                        with open(file, "w") as f:
                            f.write("k,analysis,bin,value\n")

                if k == 2:
                    with open(k_file, "w") as f:
                        f.write("k,k_ratio\n")

                analyses.append("Alg-AG_analysis")
                analyses.append("Alg-PP_analysis")

                analyses.append("gonzalez_analysis")
                analyses.append("hs_analysis")
                analyses.append("Pseudo-PoF-Alg_analysis")

                for analysis in analyses:
                    with open(max_radius_file, "a") as f:
                        f.write(
                            f"{k},{analysis.split('_analysis')[0]},{d[analysis]['Max radius']}\n"
                        )
                    with open(max_ag_file, "a") as f:
                        f.write(
                            f"{k},{analysis.split('_analysis')[0]},{d[analysis]['Max f1 ratio']}\n",
                        )
                    with open(max_pp_file, "a") as f:
                        f.write(
                            f"{k},{analysis.split('_analysis')[0]},{d[analysis]['Max f2 ratio']}\n",
                        )
                    with open(ag_hist_file, "a") as f:
                        for bin, val in d[analysis]["f1 histogram"].items():
                            f.write(
                                f"{k},{analysis.split('_analysis')[0]},{bin.replace(',', ';')},{val}\n"
                            )
                    with open(pp_hist_file, "a") as f:
                        for bin, val in d[analysis]["f2 histogram"].items():
                            f.write(
                                f"{k},{analysis.split('_analysis')[0]},{bin.replace(',', ';')},{val}\n"
                            )
                    with open(k_file, "a") as f:
                        val = d["Pseudo-PoF-Alg_analysis"]["Num centers"]
                        f.write(f"{k},{val/k:0.4f}\n")

            else:
                print(f"{filename} doesn't exist")

    name_dict = {
        "gonzalez": "Gonzalez",
        "hs": "Hochbaum-Shmoys",
    }

    mpl.rcParams["lines.linewidth"] = 8
    mpl.rcParams["font.size"] = 30
    mpl.rc("font", family="Times New Roman")

    def plot_helper(
        filename,
    ):
        with open(filename, "r") as f:
            df = pd.read_csv(f)
            df = df.drop_duplicates(ignore_index=True)
            df["analysis"] = df["analysis"].replace(name_dict)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.ticklabel_format(axis="y", style="", scilimits=(-2, 4))
            default_cycler = cycler(color=["r", "g", "b", "y", "m"]) + cycler(
                linestyle=["-", "--", ":", "-.", (0, (1, 1))]
            )

            ax.set_prop_cycle(default_cycler)

        return df, fig, ax

    for dataset in args.dataset:
        max_radius_file = f"{args.output_directory}/{dataset}/{dataset}_max_radius_file"
        max_ag_file = f"{args.output_directory}/{dataset}/{dataset}_max_ag_file"
        max_pp_file = f"{args.output_directory}/{dataset}/{dataset}_max_pp_file"
        ag_hist_file = f"{args.output_directory}/{dataset}/{dataset}_ag_hist_file"
        pp_hist_file = f"{args.output_directory}/{dataset}/{dataset}_pp_hist_file"

        k_file = f"{args.output_directory}/{dataset}/{dataset}_k_comparison"

        # figure 1 in paper
        max_radius, fig, ax = plot_helper(max_radius_file)
        for name, group in max_radius.groupby("analysis"):
            group.plot(
                x="k",
                y="value",
                ax=ax,
                label=name,
            )

        ax.set_ylabel("Maximum assignment distance")
        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_1.png")
        plt.close(fig)

        # figure 2c in paper
        max_ag, fig, ax = plot_helper(max_ag_file)

        ax.set_ylabel("$max_j f^{AG}_j$")
        for name, group in max_ag.groupby("analysis"):
            if name == "Alg-PP":
                continue
            group.plot(
                x="k",
                y="value",
                ax=ax,
                label=name,
            )

        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_2c.png")
        plt.close(fig)

        # figure 2a in paper
        max_pp, fig, ax = plot_helper(max_pp_file)

        ax.set_ylabel("$max_j f^{PP}_j$")
        plt.axhline(y=2, color="black", linestyle="-")
        for name, group in max_pp.groupby("analysis"):
            if not (name == "Alg-PP" or name == "Pseudo-PoF-Alg"):
                continue
            group.plot(
                x="k",
                y="value",
                ax=ax,
                label=name,
            )

        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_2a.png")
        plt.close(fig)

        # figure 2b in paper
        max_pp, fig, ax = plot_helper(max_pp_file)

        ax.set_yscale("log")
        ax.set_yticks([2, 10, 100, 500])
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.get_yaxis().set_major_formatter(formatter)
        ax.set_ylabel("$max_j f^{PP}_j$")
        plt.axhline(y=2, color="black", linestyle="-")
        for name, group in max_pp.groupby("analysis"):
            if not (name == "Gonzalez" or name == "Hochbaum-Shmoys"):
                continue
            group.plot(
                x="k",
                y="value",
                ax=ax,
                label=name,
            )

        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_2b.png")
        plt.close(fig)

        # figure 3b in paper
        ag_hist, fig, ax = plot_helper(ag_hist_file)

        def helper(x):
            return (x.str.split(";").str[0].str[1:]).astype(float)

        grouped_ag_hist = ag_hist.groupby(["k", "analysis"])
        grouped_two_ag_hist = ag_hist[helper(ag_hist["bin"]) >= 2].groupby(
            ["k", "analysis"]
        )

        total = grouped_ag_hist.agg({"value": "sum"})
        subtotal = grouped_two_ag_hist.agg({"value": "sum"})

        ratio_greater_two = (100 * subtotal / total).fillna(0).reset_index()

        ax.set_ylabel("Percentage of points with $f^{AG}_j > 2$")
        for name, group in ratio_greater_two.groupby("analysis"):
            if name != "Gonzalez" and name != "Hochbaum-Shmoys":
                continue
            group.plot(
                x="k",
                y="value",
                ax=ax,
                label=name,
            )

        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_3b.png")
        plt.close(fig)

        # figure 3a in paper
        pp_hist, fig, ax = plot_helper(pp_hist_file)

        def helper(x):
            return (x.str.split(";").str[0].str[1:]).astype(float)

        grouped_pp_hist = pp_hist.groupby(["k", "analysis"])
        grouped_two_pp_hist = pp_hist[helper(pp_hist["bin"]) >= 2].groupby(
            ["k", "analysis"]
        )

        total = grouped_pp_hist.agg({"value": "sum"})
        subtotal = grouped_two_pp_hist.agg({"value": "sum"})

        ratio_greater_two = (100 * subtotal / total).fillna(0).reset_index()

        ax.set_ylabel("Percentage of points with $f^{PP}_j > 2$")
        for name, group in ratio_greater_two.groupby("analysis"):
            if name != "Gonzalez" and name != "Hochbaum-Shmoys":
                continue
            group.plot(
                x="k",
                y="value",
                ax=ax,
                label=name,
            )

        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_3a.png")
        plt.close(fig)

        # figure 3c in paper
        with open(k_file, "r") as f:
            k_ratio = pd.read_csv(f)

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.ticklabel_format(axis="y", style="", scilimits=(-2, 4))

        k_ratio.plot(
            x="k",
            y="k_ratio",
            ax=ax,
            label="Pseudo-PoF-Alg",
        )
        ax.set_ylabel("# centers used / k")
        plt.savefig(f"{args.output_directory}/{dataset}/{dataset}_3c.png")
        plt.close(fig)
