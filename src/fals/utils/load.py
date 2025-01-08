import os
from typing import Optional

import boto3
import duckdb
import numpy as np
import pandas as pd

from fals.config import ROOT_DATA_PATH
from fals.utils.stats import perform_multitask_lasso_cv, prepare_RNA_data, run_power_analysis, run_TDP43_imputation


# relations to use in analysis
def wells_relation(
    con: duckdb.DuckDBPyConnection, root_path: str, group_ids: Optional[list[int]] = None
) -> duckdb.DuckDBPyRelation:
    """
    Fetch a DuckDB Python Relation for the wells table
    Optionally filtered by group ids
    """
    wells = con.from_parquet(os.path.join(root_path, "wells", "*.pq"), union_by_name=True)
    if group_ids:
        group_id_str = ", ".join(str(i) for i in group_ids)
        wells = wells.filter(f"group_id in ({group_id_str})")
    return wells


def features_relation(con: duckdb.DuckDBPyConnection, root_path: str) -> duckdb.DuckDBPyRelation:
    """
    Fetch a DuckDB Python Relation for the features table
    Optionally filtered by group ids
    """
    feautures = con.from_parquet(os.path.join(root_path, "features", "*.pq"))
    return feautures


def workflow_relation(con: duckdb.DuckDBPyConnection, root_path: str) -> duckdb.DuckDBPyRelation:
    """
    note: we need to pull in workflow information for cell line instances.
    this can be regenerated running on researchdb and saving to csv/parquet:
    select
        p.file_registry_id$ as parent_instance_registry_id
        , p.name$ as parent_instance_name
        , p.notes as parent_instance_notes
        , ci.file_registry_id$ as cell_line_instnce_registry_id
        , case
            when p.notes like '%Super Scale%' then 'super_scale'
            when p.notes like '%WF1%' then 'wf_1'
            when p.notes like '%WF2%' then 'wf_2'
            -- fall back on wf_2 if null
            when p.notes like '%SS_%' then 'super_scale'
            when p.notes is null then 'wf_2'
            else 'other'
        end as workflow
        , ci.name$ as cell_line_instance_name
    from cell_line_instance ci
    join cell_line_instance p
        on ci.parent_batch = p.id
    """
    workflow = con.from_parquet(os.path.join(root_path, "workflow.pq"))
    return workflow


def get_con() -> duckdb.DuckDBPyConnection:
    """Instantiate an in-memory duckdb connection"""
    con = duckdb.connect()
    con.execute("install httpfs")
    con.execute("load httpfs")
    s = boto3.session.Session()
    creds = s.get_credentials().get_frozen_credentials()
    cred_map = {
        "s3_access_key_id": creds.access_key,
        "s3_secret_access_key": creds.secret_key,
        "s3_session_token": creds.token,
        "s3_region": "us-west-2",
    }
    for k, v in cred_map.items():
        con.execute(f"SET {k} = '{v}'")
    return con


# Figure specific loading methods


def load_genes(rna_features, focal_rna, genes_dict):
    """Load genes and stressors of interest"""

    genes_of_interest = []
    for _, genes in genes_dict.items():
        for gene in genes:
            if gene in rna_features[focal_rna].columns:
                genes_of_interest.append(gene)
            else:
                print(gene, "not found")

    all_genes = rna_features[focal_rna].columns

    return genes_of_interest, all_genes


def load_stressors(rna_features, rna_reps2use):
    """Load stressors and hold-out targets"""
    stressors = np.asarray(
        [
            "_".join(x.split("_")[-3:]) if "DMSO" not in x else "DMSO"
            for x in rna_features[rna_reps2use[0]].index
        ]
    )
    hold_out_targets = np.asarray([x.split("_")[1] for x in rna_features[rna_reps2use[0]].index])

    return stressors, hold_out_targets


def load_performance(focal_stressors, performance_topdir, focal_dino, focal_rna):
    """Load performance data dataset"""

    perform_dict = {}

    for stressor in focal_stressors:
        file_path = os.path.join(
            performance_topdir,
            f"Lasso.pca50.regress_out_alive_in_well.no_pred_for_density.hold_out_by_donor_per_{stressor}.avg_across_holdouts.csv",
        )
        df = pd.read_csv(file_path, index_col=0)

        # Filter by model and focal_dino and focal_rna
        df = df[df["model"] == f"{focal_dino}-{focal_rna}"]
        for _, row in df.iterrows():
            perform_dict[f"{row['gene']}-{stressor}"] = row["pearson.cor"]

    return perform_dict


def load_data_fig_4A() -> pd.DataFrame:
    """Load data for Figure 4A"""

    base_path = os.path.join(ROOT_DATA_PATH, "ml/tables/features")
    df = pd.read_parquet(os.path.join(base_path, "tardbp_features.pq"))

    return run_TDP43_imputation(df)


def load_data_fig_4A_supp() -> pd.DataFrame:
    """Load data for Figure 4A supplementary"""

    base_path = os.path.join(ROOT_DATA_PATH, "ml/tables/features")
    return pd.read_parquet(os.path.join(base_path, "tardbp_full_features.pq"))


def load_data_fig_4D_E(data_dir):
    """Load RNA-seq imputation dataset"""

    rna_reps2use = [
        "CPM_z",
    ]
    dino_feature2use = ["alive_in_well", "iDINO_v3_4channel_DAPI_TDP43_STMN2_TUJ1_224crop"]
    groups2use = ["val32rep1", "val32rep2", "valgroup2"]

    focal_stressors = ["DMSO"]
    focal_dino = "iDINO_v3_4channel_DAPI_TDP43_STMN2_TUJ1_224crop"
    focal_rna = "CPM_z"

    genes_dict = {
        "tdp43_targets": ["STMN2", "UNC13A", "UNC13B", "ELAVL2", "ELAVL3"],
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

    rna_features, dino_features, genes_of_interest, all_genes = prepare_RNA_data(
        os.path.join(data_dir, "data_concat.h5ad"),
        groups2use,
        rna_reps2use,
        dino_feature2use,
    )

    genes_of_interest, all_genes = load_genes(rna_features, focal_rna, genes_dict)
    stressors, hold_out_targets = load_stressors(
        rna_features=rna_features, rna_reps2use=rna_reps2use
    )

    all_df_ori = perform_multitask_lasso_cv(
        focal_stressors=focal_stressors,
        stressors=stressors,
        hold_out_targets=hold_out_targets,
        dino_features=dino_features,
        rna_features=rna_features,
        focal_dino=focal_dino,
        focal_rna=focal_rna,
    )

    return all_df_ori, genes_of_interest, all_genes


def load_data_fig_5A() -> pd.DataFrame:
    """Load in-donor classifier data for Figure 5A"""

    base_path = os.path.join(ROOT_DATA_PATH, "ml", "results")
    files = [
        "results_Well_iDINO_v3_1channel_DAPI_224crop.pq",
        "results_Well_iDINO_v3_2channel_DAPI_TUJ1_224crop.pq",
        "results_Cell_iDINO_v3_4channel_DAPI_TDP43_STMN2_TUJ1_224crop.pq",
    ]

    renames = {
        "iDINO_v3_4channel_DAPI_TDP43_STMN2_TUJ1_224crop": "iDINOv3 | DAPI + TDP43 + STMN2 +TUJ1",
        "iDINO_v3_2channel_DAPI_TUJ1_224crop": "iDINOv3 | DAPI + TUJ1",
        "iDINO_v3_1channel_DAPI_224crop": "iDINOv3 | DAPI",
    }

    results = pd.concat([pd.read_parquet(os.path.join(base_path, f)) for f in files])
    results["test_and_edit"] = results.apply(
        lambda x: f"{x.gene: <18} {x.intended_test_donor: >7} ", axis=1
    )
    results = results.sort_values(by=["test_and_edit"])

    results.feature_name = results.feature_name.replace(
        renames.keys(),
        renames.values(),
    )
    results["feat_hold"] = results.apply(
        lambda x: f"{x.feature_name: <7} {x.hold_out_columns: >20}", axis=1
    )
    results["feature_name"] = results.apply(
        lambda x: x.feature_name.replace("idino_v", "iDINOv").replace(
            "alive_in_well", "Number of cells alive in well"
        ),
        axis=1,
    )

    # Remove the VCP 023 test donor since it was a mislabelled sample
    results = results[results.test_and_edit != "kiVCP              Dins023 "]

    return results


def load_data_fig_5C() -> pd.DataFrame:
    """Load cross-donor classifier data for Figure 5C"""

    def _pretty_cont_vars(cont_vars):
        if "alive_in_well" in cont_vars:
            return "Number of cells alive in well"
        if "iDINO_v3_1channel_DAPI_224crop" in cont_vars:
            return "iDINOv3 | DAPI"
        if "iDINO_v3_2channel_DAPI_TUJ1_224crop" in cont_vars:
            return "iDINOv3 | DAPI + TUJ1"
        if "iDINO_v3_4channel_DAPI_TDP43_STMN2_TUJ1_224crop" in cont_vars:
            return "iDINOv3 | DAPI + TDP43 + STMN2 + TUJ1"
        return None

    base_path = os.path.join(ROOT_DATA_PATH, "ml", "results")
    files = ["1chan_fixed_VCP_023.pq", "2chan_fixed_VCP_023.pq", "4chan_fixed_VCP_023.pq"]
    dfs = [pd.read_parquet(f"{base_path}/{file}") for file in files]

    for df in dfs[1:]:
        df = df[df["cont_vars"] != "alive_in_well"]

    results = pd.concat(dfs)
    results["cont_vars_pretty"] = results.cont_vars.apply(_pretty_cont_vars)

    return results.reset_index(drop=True)


def load_data_fig_5E() -> pd.DataFrame:
    """Load cross-donor classifiers TDP43 vs iDINO data for Figure 5E"""

    base_path = os.path.join(ROOT_DATA_PATH, "ml", "results")
    files = ["model_results_anchor.pq", "model_results_dino.pq"]
    results = pd.concat([pd.read_parquet(os.path.join(base_path, f)) for f in files])

    # Remove A5Vhom from kiSOD1 due to outlier status
    results = results[["A5Vhom" not in x for x in results.test_cell_lines]]
    results["var_type"] = results.apply(lambda x: f"{x.cont_vars.split(':')[1].strip()}", axis=1)
    results["model"] = results["var_type"].map(
        {
            "'median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity'>]": "TDP-43 mislocalization ratio",
            "'dino_embedding/iDINO_v3_4channel_DAPI_TDP43_STMN2_TUJ1_224crop'>]": "iDINOv3 | DAPI + TDP43 + STMN2 +TUJ1",
        }
    )
    return results


def load_data_fig_5F() -> pd.DataFrame:
    """Load and process power analysis data for Figure 5F"""

    base_path = os.path.join(ROOT_DATA_PATH, "ml", "predictions")

    labels_anchor = pd.read_parquet(os.path.join(base_path, "model_predicted_labels_anchor.pq"))
    labels_dino = pd.read_parquet(os.path.join(base_path, "model_predicted_labels_dino.pq"))

    # Preprocess data for power analysis (use an example cell line)
    labels_anchor["model"] = "TDP-43 mislocalization ratio"
    labels_dino["model"] = "iDINOv3 | DAPI + TDP43 + STMN2 +TUJ1"
    label_cols = ["predicted_label", "cell_line_edit_description", "model"]
    labels = pd.concat([labels_anchor[label_cols], labels_dino[label_cols]], axis=0)
    labels_sub = labels[labels.cell_line_edit_description.isin(["corr_C9ORF72_het", "C9ORF72"])]
    labels_sub["predicted_label_binary"] = labels_sub["predicted_label"].map(
        {"non-disease": 0, "disease": 1}
    )

    # Run power analysis
    df_power_analysis = run_power_analysis(
        labels_sub=labels_sub,
        pairs=[["corr_C9ORF72_het", "C9ORF72"]],
        reversion_fraction=0.9,
        N_boot=100,
        n_cells=np.arange(10, 1500, 50),
        group="cell_line_edit_description",
        feat="predicted_label_binary",
    )

    return df_power_analysis
