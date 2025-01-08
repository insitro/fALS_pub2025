from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from redun import File
from scipy.stats import pearsonr, ranksums, spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, MultiTaskLasso, MultiTaskLassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def run_power_analysis(
    labels_sub: pd.DataFrame,
    pairs: list[tuple[str]],
    reversion_fraction: float,
    N_boot: int,
    n_cells: np.ndarray,
    group: str,
    feat: str,
) -> pd.DataFrame:
    """
    Run power analysis for a given set of pairs of disease and healthy cell linesß
    """

    for healthy_label, disease_label in pairs:
        # Anchor model
        df = labels_sub.query('model == "TDP-43 mislocalization ratio"')

        df_temp_anchor = bootstrap_sample_simulations(
            df[df[group] == disease_label][feat],
            df[df[group] == healthy_label][feat],
            1.0 - reversion_fraction,
            N_boot,
            n_cells,
        )
        df_temp_anchor["model"] = "TDP-43 mislocalization ratio"

        # DINO model
        df = labels_sub.query('model == "iDINOv3 | DAPI + TDP43 + STMN2 +TUJ1"')

        df_temp_dino = bootstrap_sample_simulations(
            df[df[group] == disease_label][feat],
            df[df[group] == healthy_label][feat],
            1.0 - reversion_fraction,
            N_boot,
            n_cells,
        )
        df_temp_dino["model"] = "iDINOv3 | DAPI + TDP43 + STMN2 +TUJ1"

    df_power_analysis = pd.concat([df_temp_anchor, df_temp_dino], axis=0)
    return df_power_analysis


def prepare_RNA_data(anndata_file, groups2use, rna_reps2use, dino_feature2use):
    """Prepare RNA data for the given groups"""
    anndata_file_obj = File(anndata_file)
    if anndata_file_obj.filesystem.name == "s3":
        staged = File(f"/tmp/{anndata_file_obj.basename()}")
        anndata_file_obj.copy_to(staged)
        anndata_file_obj = staged

    ann = sc.read_h5ad(anndata_file_obj.path)

    # Load average expression per data point
    rna_features = {}
    for x in rna_reps2use:
        if "_z" == x[-2:]:
            df = ann.uns[x[:-2]].transpose()
            rna_features[x] = (df - df.mean()) / df.std()
        else:
            rna_features[x] = ann.uns[x].transpose()

    all_genes = list(rna_features[rna_reps2use[0]].columns)

    # Load average DINO embeddings per data point
    dino_features = {}
    for x in dino_feature2use:
        dino_features[x] = ann.uns[x].transpose()

    # Subset data to the group of interest
    samples2use = [
        x for x in dino_features[dino_feature2use[0]].index if x.split("_")[0] in groups2use
    ]

    for x in rna_reps2use:
        rna_features[x] = rna_features[x].loc[samples2use]
    for x in dino_feature2use:
        dino_features[x] = dino_features[x].loc[samples2use]

    # PCA on DINO
    all_pcas = {}
    for x in dino_feature2use:
        if x not in ["alive_in_well"]:
            pca = PCA(n_components=50)
            dino_features[x] = pca.fit_transform(dino_features[x])
            all_pcas[x] = pca
        else:
            dino_features[x] = dino_features[x].values

    # ALS genes
    genes_dict = {
        "tdp43_targets": ["STMN2", "UNC13A", "UNC13B", "ELAVL2", "ELAVL3"],
        "markers": ["MNX1", "ISL1", "ISL2", "FOXP1", "ACHE", "SLC18A3", "NEFH", "CHAT"],
        "als_genes": [
            "TARDBP",
            "C9orf72",
            "SOD1",
            "FUS",
            "SARM1",
            "VCP",
            "ALS2",
            "ATXN2",
            "HNRNPA1",
        ],
        "others": ["GRIA1", "GRIN1"],
    }

    genes_of_interest = []
    for k, v in genes_dict.items():
        for x in v:
            if x in rna_features[rna_reps2use[0]].columns:
                genes_of_interest.append(x)

    return rna_features, dino_features, genes_of_interest, all_genes


def train_test(
    hold_out_targets,
    rna_features,
    dino_features,
    model_type="LassoCV",
    genes2use=None,
    max_iter=100000,
    n_alphas=100,
):
    """Train and test the model for the given features"""

    result = pd.DataFrame()
    result_ungrouped = pd.DataFrame()
    all_models = {}

    for dino_feature in dino_features.keys():
        for rna_rep in rna_features.keys():
            print(dino_feature, rna_rep)

            X = dino_features[dino_feature]
            if genes2use is None:
                genes2use = rna_features[rna_rep].columns
            y = rna_features[rna_rep][genes2use].values

            all_metrics_ungrouped, model = l1lr_hold_out(
                X, y, hold_out_targets, max_iter=max_iter, model_type=model_type, n_alphas=n_alphas
            )
            all_metrics = (
                all_metrics_ungrouped[["target_idx", "mse", "r2", "pearson.cor", "spearman.cor"]]
                .groupby("target_idx")
                .median()
            )

            all_metrics["target_idx"] = all_metrics.index
            all_metrics["rna_rep"] = rna_rep
            all_metrics["dino_variants"] = dino_feature
            all_metrics["model"] = dino_feature + "-" + rna_rep

            # all_metrics_ungrouped['target_idx'] = all_metrics_ungrouped.index
            all_metrics_ungrouped["rna_rep"] = rna_rep
            all_metrics_ungrouped["dino_variants"] = dino_feature
            all_metrics_ungrouped["model"] = dino_feature + "-" + rna_rep

            result = pd.concat((result, all_metrics), axis=0)
            result_ungrouped = pd.concat((result_ungrouped, all_metrics_ungrouped), axis=0)
            all_models[".".join([dino_feature, rna_rep])] = model

    result["gene"] = [genes2use[x] for x in result["target_idx"]]
    result_ungrouped["gene"] = [genes2use[x] for x in result_ungrouped["target_idx"]]
    return result, result_ungrouped, all_models


def l1lr_hold_out(
    X, y, hold_out_targets, max_iter=1000, model_type="LassoCV", random_state=10, n_alphas=100
):
    all_metrics = []

    for hold_out_target in np.unique(hold_out_targets):
        train_index = hold_out_targets != hold_out_target
        test_index = hold_out_targets == hold_out_target
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]

        assert model_type in ["LassoCV", "Lasso"]
        if model_type == "LassoCV":
            model = MultiTaskLassoCV(
                max_iter=max_iter, n_jobs=-1, random_state=random_state, n_alphas=n_alphas
            )
        else:
            model = MultiTaskLasso(max_iter=max_iter, random_state=random_state)

        model.fit(train_X, train_y)
        pred = model.predict(test_X).reshape(len(test_X), -1)

        test_y = test_y.reshape(len(test_y), -1)

        for idx in range(test_y.shape[1]):
            if len(np.unique(test_y[:, idx])) > 1 and len(np.unique(pred[:, idx])) > 1:
                test = pearsonr(pred[:, idx], test_y[:, idx])
                pearsonr_val, pearsonr_p = test[0], test[1]

                test = spearmanr(pred[:, idx], test_y[:, idx])
                spearman_val, spearman_p = test[0], test[1]
            else:
                pearsonr_val, pearsonr_p = np.nan, np.nan
                spearman_val, spearman_p = np.nan, np.nan

            if pearsonr_p and pearsonr_p > 0:
                pearsonr_logp = -np.log10(pearsonr_p)
            else:
                pearsonr_logp = np.nan

            if spearman_p and spearman_p > 0:
                spearmanr_logp = -np.log10(spearman_p)
            else:
                spearmanr_logp = np.nan

            if len(pred) > 3:
                r2score = r2_score(pred[:, idx], test_y[:, idx])
            else:
                r2score = np.nan
            metrics = [
                hold_out_target,
                idx,
                mean_squared_error(pred[:, idx], test_y[:, idx]),
                r2score,
                pearsonr_val,
                pearsonr_logp,
                spearman_val,
                spearmanr_logp,
            ]

            all_metrics.append(metrics)

    all_metrics = pd.DataFrame(
        all_metrics,
        columns=[
            "hold_out",
            "target_idx",
            "mse",
            "r2",
            "pearson.cor",
            "pearson.logp",
            "spearman.cor",
            "spearman.logp",
        ],
    )

    if model_type == "LassoCV":
        model = MultiTaskLassoCV(max_iter=max_iter, n_jobs=-1)
    else:
        model = MultiTaskLasso(max_iter=max_iter)

    model.fit(X, y)

    return all_metrics, model


def analysis(all_result, genes_of_interest):
    """Perform analysis on imaging the results"""

    for model in all_result["model"].unique():
        result = all_result[all_result["model"] == model]
        print(model)
        sns.displot(result.groupby("gene").median()["pearson.cor"])
        plt.show()

        als_result = result.loc[result["gene"].isin(genes_of_interest)]

        df = {"all_genes": result["pearson.cor"], "als_genes": als_result["pearson.cor"]}

        sns.displot(
            df,
            common_norm=False,
            kind="kde",
            rug=False,
            fill=False,
        )
        plt.xlabel("pearson corr")
        plt.show()

        print(ranksums(als_result["pearson.cor"], result["pearson.cor"], alternative="greater"))

        myorder = als_result.groupby("gene").median().sort_values("pearson.cor").index
        sns.barplot(data=als_result, y="gene", x="pearson.cor", order=myorder)
        plt.show()


def filter_data(all_dfs, stressor, mutant, wt):
    """Filter the DataFrame for a given stressor and mutant pair."""
    all_df = all_dfs[all_dfs["stressor"] == stressor]
    df2use = all_df[all_df["mutant"].isin([mutant, wt])].copy()

    if mutant == "C9ORF72":
        df2use = df2use[df2use["donor"] != "Dins390"]
    elif mutant == "VCP":
        df2use = df2use[df2use["donor"] != "Dins023"]

    return df2use


def calculate_expression_diff(df2use, mutant, wt, perform_dict, stressor):
    """Calculate expression differences and relevant metrics."""
    df = df2use.copy()
    df = df.pivot_table(columns="mutant", index=["gene", "donor", "source"], values="expression")
    df["mut_diff"] = df[mutant] - df[wt]
    df = df.groupby(["gene", "source"]).median()
    df = df.pivot_table(index="gene", columns="source", values="mut_diff")
    df["gene"] = df.index
    df["DINO_pred_performance"] = np.clip(
        [perform_dict.get(f"{gene}-{stressor}", 0) for gene in df.index], 0, 1
    )

    return df


def transform_and_concatenate(all_pred_df, all_y):
    """
    Transform prediction DataFrames and concatenate imputed and observed data.
    """

    def transform_data(df, source_label):
        df = df.copy()
        df["sample"] = df.index
        df = df.melt(
            id_vars="sample",
            value_vars=all_pred_df.columns,
            var_name="gene",
            value_name="expression",
        )
        df["stressor"] = [
            "_".join(x.split("_")[-3:]) if "DMSO" not in x else x.split("_")[-1]
            for x in df["sample"]
        ]
        df["donor"] = [x.split("_")[1] for x in df["sample"]]
        df["mutant"] = [
            "_".join(x.split("_")[2:-1]) if "DMSO" in x else "_".join(x.split("_")[2:-3])
            for x in df["sample"]
        ]
        df["source"] = source_label
        return df

    df_imputed = transform_data(all_pred_df, "DINO-imputed")
    df_observed = transform_data(all_y, "observed")

    all_df_ori = pd.concat((df_imputed, df_observed), axis=0)
    return all_df_ori


def perform_multitask_lasso_cv(
    focal_stressors,
    stressors,
    hold_out_targets,
    dino_features,
    rna_features,
    focal_dino,
    focal_rna,
):
    """
    Perform MultiTaskLasso cross-validation for the given stressors and datasets.
    """

    all_pred_df = pd.DataFrame()
    all_y = pd.DataFrame()

    for stressor in focal_stressors:
        pick = stressors == stressor

        hold_out_targets2use = hold_out_targets[pick]

        if len(np.unique(hold_out_targets2use)) < 2:
            print("Not enough data to perform cross-validation")
            continue

        dino_features2use = {k: v[pick] for k, v in dino_features.items()}
        rna_features2use = {}

        for k, v in rna_features.items():
            rna_features2use[k] = v[pick]
            model = MultiTaskLasso()
            model.fit(dino_features2use["alive_in_well"], rna_features2use[k])
            rna_features2use[k] -= model.predict(dino_features2use["alive_in_well"])

        X = dino_features2use[focal_dino]
        y = rna_features2use[focal_rna]

        pred_df = pd.DataFrame()
        y_df = pd.DataFrame()

        for hold_out_target in np.unique(hold_out_targets2use):
            train_index = hold_out_targets2use != hold_out_target
            test_index = hold_out_targets2use == hold_out_target
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]

            model = MultiTaskLasso(max_iter=100000, random_state=10)
            model.fit(train_X, train_y)
            pred = model.predict(test_X).reshape(len(test_X), -1)

            pred_df = pd.concat(
                (pred_df, pd.DataFrame(pred, columns=test_y.columns, index=test_y.index))
            )
            y_df = pd.concat(
                (y_df, pd.DataFrame(test_y, columns=test_y.columns, index=test_y.index))
            )

        assert len(pred_df) == len(y)
        all_pred_df = pd.concat((all_pred_df, pred_df))
        all_y = pd.concat((all_y, y_df))

    return transform_and_concatenate(all_pred_df, all_y)


def bootstrap_sample_simulations(
    trait_population_1: pd.Series,
    trait_population_2: pd.Series,
    population_1_fraction: float,
    N_boot: int,
    sample_sizes: Iterable[int],
    seed: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Bootstrap sample simulations to compare the power of t-tests for two populations.

    The provided series should be scalars. The two populations need not be the same size, but
    the simulated samples are equal sizes. The trait might be a direct measurement,
    or the output of a classifier, such as the logit or a class label. If the trait is a binary
    classifier, then the t-test is performed between the average accuracy of the classifier.

    Parameters
    ----------
    trait_population_1: pd.Series
        Trait values for population 1. Samples will be drawn from this series with replacement.
    trait_population_2: pd.Series
        Trait values for population 2. Samples will be drawn from this series with replacement.
    population_1_fraction: float
        Fraction of population 1 in simulated dataset. The rest are drawn from sample 2.
    N_boot: int
        Number of simulations at each size
    sample_sizes: Iterable[int]
        The sample sizes to run; each will be run `N_boot` times.
    seed: Optional[Any]
        Anything `np.random.default_rng` can accept in the `seed` argument.

    Returns
    -------
    pd.DataFrame
        Contains the outputs from all simulations.
    """

    generator = np.random.default_rng(seed=seed)

    results = []
    for sample_size in sample_sizes:
        for i in range(N_boot):
            # Simulate sick and target datasets
            p1 = trait_population_1.sample(int(sample_size), replace=True, random_state=generator)
            p2 = trait_population_2.sample(int(sample_size), replace=True, random_state=generator)
            p1_mixture = trait_population_1.sample(
                int(sample_size * population_1_fraction), replace=True, random_state=generator
            )
            p2_mixture = trait_population_2.sample(
                int(sample_size) - int(1.0 - population_1_fraction),
                replace=True,
                random_state=generator,
            )
            mixture = pd.concat([p1_mixture, p2_mixture], axis=0)

            result = {
                "sample_size": sample_size,
                "population_1_fraction": population_1_fraction,
                "p1_mean": p1.mean(),
                "p2_mean": p2.mean(),
                "mixture_mean": mixture.mean(),
            }
            mixture_vs_p1 = ttest_ind(mixture, p1)
            result["mixture_vs_p1_t_statistic"] = mixture_vs_p1.statistic
            result["mixture_vs_p1_p_value"] = mixture_vs_p1.pvalue
            mixture_vs_p2 = ttest_ind(mixture, p2)
            result["mixture_vs_p2_t_statistic"] = mixture_vs_p2.statistic
            result["mixture_vs_p2_p_value"] = mixture_vs_p2.pvalue

            results.append(result)

    return pd.DataFrame(results)


def run_TDP43_imputation(df):
    """Perform TDP43 imputation from DAPI+TUJI DINO embeddings using linear regression"""

    df_train, df_test = train_test_split(df, train_size=0.80, random_state=2712)

    X_train = np.array(list(df_train['iDINO_DAPI+TUJ1'].values))
    y_train = df_train.log_median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity
    X_test = np.array(list(df_test['iDINO_DAPI+TUJ1'].values))
    y_test = df_test.log_median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity

    clf = LinearRegression()
    _ = clf.fit(X_train, y_train)
    metrics = spearmanr(y_test, clf.predict(X_test))

    return y_test, clf.predict(X_test), metrics.correlation
