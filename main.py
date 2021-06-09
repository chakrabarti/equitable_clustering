import argparse
import os.path
import pickle

from equitable_clustering import *
from scipy.spatial import distance_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sample_file",
        dest="sample_file",
        type=str,
        default="data/processed/bank/bank_1.pkl",
    )
    parser.add_argument(
        "--output_directory", dest="output_directory", type=str, default="output"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    random.seed(args.seed)

    dataset_name = args.sample_file.split("/")[-1].split("_")[0]

    dist_matrix_file = (
        f"{args.output_directory}/{dataset_name}/{dataset_name}_dist_matrix.pkl"
    )

    output_file = f"{args.output_directory}/{dataset_name}/{dataset_name}_{args.k}_experiment_output.pkl"

    Timer.TimerClassReset()
    if os.path.isfile(dist_matrix_file):
        print("Loading distance matrix...")
        with open(dist_matrix_file, "rb") as f:
            original_points, dist_matrix = pickle.load(f)
        print("Loaded distance matrix!")
    else:
        print("Loading data file...")
        with open(args.sample_file, "rb") as f:
            original_points, _ = pickle.load(f)
            print("Loaded data file!")
            print("Generating distance matrix...")
            dist_matrix = distance_matrix(original_points, original_points, p=2)
            dist_matrix /= dist_matrix.max()
            print("Generated distance matrix!")
        print("Dumping distance matrix...")
        with open(dist_matrix_file, "wb") as f:
            pickle.dump((original_points, dist_matrix), f)
        print("Dumped distance matrix!")

    print("Creating clustering problem...")
    clustering_problem = ClusteringProblem(
        args.k,
        dist_matrix,
    )
    print("Created clustering problem!")

    hs_file = f"{args.output_directory}/{dataset_name}/{dataset_name}_{args.k}_hs.pkl"
    gonz_file = (
        f"{args.output_directory}/{dataset_name}/{dataset_name}_{args.k}_gonz.pkl"
    )

    if os.path.isfile(hs_file):
        print("Loading HS...")
        with open(hs_file, "rb") as f:
            R_f, R_f_index, hs_phi = pickle.load(f)
        print("Loaded HS!")

        clustering_problem.set_R_f(R_f, R_f_index, hs_phi)
    else:
        print("Running HS...")
        hs_timer = Timer("hs")
        clustering_problem.find_R_f()
        hs_timer.Accumulate()
        print("Ran HS!")
        print("Dumping HS...")
        dump_hs_timer = Timer("dump hs")
        with open(hs_file, "wb") as f:
            pickle.dump(
                (
                    clustering_problem.R_f,
                    clustering_problem.R_f_index,
                    clustering_problem.hs_phi,
                ),
                f,
            )
        dump_hs_timer.Accumulate()
        print("Dumped HS!")

    print("Generating similarity sets...")
    clustering_problem.construct_similarity_sets()
    print("Generated similarity sets!")

    R_f_hs = None
    hs_phi = None
    gonzalez_assn = None

    print("Running Alg-AG...")
    algorithm_ag_timer = Timer("algorithm_ag")
    (
        R_f_ag,
        (phi_ag, isolated_map_ag, non_isolated_map_ag),
        R_f_hs,
        hs_phi,
    ) = find_R_f(clustering_problem, algorithm=Algorithm.AG, hs=False)
    algorithm_ag_timer.Accumulate()
    print("Ran Alg-AG!")

    print("Running Alg-PP...")
    algorithm_pp_timer = Timer("algorithm_pp")
    (
        R_f_pp,
        (phi_pp, isolated_map_pp, non_isolated_map_pp),
        _,
        _,
    ) = find_R_f(clustering_problem, algorithm=Algorithm.PP, hs=False)
    algorithm_pp_timer.Accumulate()
    print("Ran Alg-PP PP!")

    print("Running Alg-AG analysis...")
    ag_analysis_timer = Timer("ag_analysis")
    ag_analysis = run_analysis(clustering_problem, phi_ag)
    ag_analysis_timer.Accumulate()
    print("Ran Alg-AG analysis!")

    print("Running algorithm PP analysis...")
    pp_analysis_timer = Timer("pp_analysis")
    pp_analysis = run_analysis(clustering_problem, phi_pp)
    pp_analysis_timer.Accumulate()
    print("Ran Alg-PP analysis!")

    print("Running Pseudo-PoF-Alg...")
    algorithm_pseudo_pof_timer = Timer("algorithm_pseudo_pof")
    (
        R_f_pseudo_pof,
        (phi_pseudo_pof, isolated_map_pseudo_pof, non_isolated_map_pseudo_pof),
        R_f_phi,
        hs_phi,
    ) = find_R_f(clustering_problem, algorithm=Algorithm.Pseudo_PoF, hs=False)
    algorithm_pseudo_pof_timer.Accumulate()

    print("Running Pseudo-PoF-Alg analysis...")
    pseudo_pof_analysis_timer = Timer("pseudo_pof_analysis")
    pseudo_pof_analysis = run_analysis(clustering_problem, phi_pseudo_pof)
    pseudo_pof_analysis_timer.Accumulate()
    print("Ran Pseudo-PoF-Alg analysis!")

    if os.path.isfile(gonz_file):
        print("Loading Gonzalez assn...")
        with open(gonz_file, "rb") as f:
            gonzalez_assn = pickle.load(f)
        print("Loaded Gonzalez assn!")
    else:
        print("Running Gonzalez...")
        gonzalez_assn_timer = Timer("gonzalez_assn")
        gonzalez_assn = Gonzalez(clustering_problem.dist_matrix, args.k)
        print("Ran Gonzalez!")
        gonzalez_assn_timer.Accumulate()

        print("Dumping Gonzalez...")
        with open(gonz_file, "wb") as f:
            pickle.dump(gonzalez_assn, f)
        print("Dumped Gonzalez!")

    print("Running Gonzalez analysis...")
    gonzalez_analysis_timer = Timer("gonzalez_analysis")
    gonzalez_analysis = run_analysis(clustering_problem, gonzalez_assn)
    gonzalez_analysis_timer.Accumulate()
    print("Ran Gonzalez analysis!")

    print("Running HS analysis...")
    hs_analysis_timer = Timer("hs_analysis")
    hs_analysis = run_analysis(clustering_problem, hs_phi)
    hs_analysis_timer.Accumulate()
    print("Ran HS analysis!")

    timer_str = Timer.PrintAccumulated()

    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "similarity_set_radii": clustering_problem.R_j_array,
                "R_f_ag": R_f_ag,
                "phi_ag": phi_ag,
                "non_isolated_map_ag": non_isolated_map_ag,
                "isolated_map_ag": isolated_map_ag,
                "R_f_pp": R_f_pp,
                "phi_pp": phi_pp,
                "non_isolated_pp": non_isolated_map_pp,
                "isolated_map_pp": isolated_map_pp,
                "R_f_pseudo_pof": R_f_pseudo_pof,
                "phi_pseudo_pof": phi_pseudo_pof,
                "non_isolated_map_pseudo_pof": non_isolated_map_pseudo_pof,
                "isolated_map_pseudo_pof": isolated_map_pseudo_pof,
                "R_f_hs": R_f_hs,
                "phi_hs": hs_phi,
                "phi_gonz": gonzalez_assn,
                "Alg-AG_analysis": ag_analysis,
                "Alg-PP_analysis": pp_analysis,
                "Pseudo-PoF-Alg_analysis": pseudo_pof_analysis,
                "gonzalez_analysis": gonzalez_analysis,
                "hs_analysis": hs_analysis,
                "timer_str": timer_str,
            },
            f,
        )
