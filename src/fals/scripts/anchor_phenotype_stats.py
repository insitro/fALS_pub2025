import os

from typing import Any

import duckdb
import numpy as np
import pandas as pd
import redun
import statsmodels.formula.api as smf

from scipy.stats import ttest_ind

from fals import analysis
from fals.config import ROOT_DATA_PATH


redun_namespace = "fals.scripts.anchor_stats"


"""
CONFIG AND ALIASES
"""
# Aliases
tdp43_ratio = "median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity"
well_tdp43_ratio = "well_TDP43_ratio"
stmn2 = "median_cell_mask_STMN2_pixel_intensity"
well_stmn2 = "well_STMN2_cell_intensity"

# for this analysis, we're also looking in the neurites
neurite_stmn2 = "mean_neurite_STMN2_pixel_intensity"
well_neurite_stmn2 = "well_STMN2_neurite_intensity"

# normalized well level stats
norm_tdp43_ratio = "well_TDP43_ratio_norm"
norm_stmn2 = "well_STMN2_cell_intensity_norm"
norm_neurite_stmn2 = "well_STMN2_neurite_intensity_norm"

dmso = "Dimethyl sulfoxide [MilliporeSigma]"
bortezomib = "Bortezomib [Tocris]"
puromycin = "Puromycin dihydrochloride [MilliporeSigma]"
thapsigargin = "Thapsigargin [SelleckChem]"
tunicamycin = "Tunicamycin [MilliporeSigma]"

drug_compounds = [bortezomib, puromycin, thapsigargin, tunicamycin]
all_compound = [dmso] + drug_compounds

# target doses
dose_pairs = [(dmso, 0), (bortezomib, 0.0078), (puromycin, 3.4965), (thapsigargin, 0.0137), (tunicamycin, 0.4996)]

# shorthand
drug_labels = {
    dmso: "0 DMSO",
    bortezomib: "Bort",
    puromycin: "Puro",
    thapsigargin: "Thaps",
    tunicamycin: "Tunica"
}

# experiments
grp2exp = {
    1: "Group1-4",
    2: "Group1-4",
    3: "Group1-4",
    4: "Group1-4",
    5: "Validation 16",
    6: "Validation 32 part 1",
    7: "Validation 32 part 1",
    8: "Validation 32 part 2",
    9: "Validation 32 part 2",
    10: "Validation 48line",
    11: "Validation 48line2",
}
exp2grps = {
    "Exp1": "(1, 2, 3, 4)",
    "Exp2": "(5, 6, 7)",
    "Exp3": "(8, 9)",
    "Exp4": "(10)",
    "Exp5": "(11)",
}

mutations_to_score = [
    'C9ORF72',
    'kiDCTN1-R1101Khom',  # no validation
    'kiDCTN1-R1101het',  # no validation
    'kiFUS-R521Chet',
    'kiSOD1-A5Vhet',
    'kiSOD1-A5Vhom',
    'kiTARDBP-G295Shet',
    'kiTARDBP-M337Vhet',
    'kiVCP-R155Chet',
    'koTBK1-hom',  # no validation
]

fals_output_path = os.path.join(ROOT_DATA_PATH, "anchor", "stats")

# exclude "dead" plates
table_path = os.path.join(ROOT_DATA_PATH, "anchor", "tables")
well_filters = "plate_barcode not in ('PF2887', 'PF3060', 'PF3158')"


def density_compare(rel, mutation, drug, dose, features):
    """
    Generate a duckdb relation that has density matched WT and mutant pairs for a given feature
    Columns:
        alive_in_well
        group_id
        plate_barcode
        category ('wt' or 'mut')
        donor
        condition ('WT-<type>' or mutation)
        feature
    """
    wt_filter = analysis.wt_filter
    featurestr = ", ".join(features)
    mut_query = rel.filter(
        f"cell_line_edit_description = '{mutation}' and compound_name='{drug}' and concentration_uM={dose}"
    ).set_alias("mut")
    mut_pairs = mut_query.project("donor_registry_id, group_id").distinct().set_alias("mut_pairs")
    wt_paired = rel.filter(
        f"disease_category in ({wt_filter}) and compound_name='{drug}' and concentration_uM={dose}"
    ).set_alias("wt").join(
        mut_pairs,
        how="inner",
        condition="wt.donor_registry_id=mut_pairs.donor_registry_id and wt.group_id=mut_pairs.group_id"
    ).set_alias("wt_paired")
    if wt_paired.count('*').fetchone()[0] < 1:
        return None
    combined_pairs = wt_paired.project(
        f"alive_in_well, group_id, plate_barcode, 'wt' as category, donor_registry_id as donor, 'WT-'|| wt_paired.cell_line_edit_description as condition, {featurestr}"
    ).union(mut_query.project(f"alive_in_well, group_id, plate_barcode, 'mut' as category, donor_registry_id as donor, '{mutation}' as condition, {featurestr}"))
    # get bounds base on alive_in_well for each donor
    donor_min_max = combined_pairs.aggregate("""
    donor,
    quantile(case when category = 'mut' then alive_in_well end, 0.1) as mut_min,
    quantile(case when category = 'wt' then alive_in_well end, 0.1) as wt_min,
    quantile(case when category = 'mut' then alive_in_well end, 0.9) as mut_max,
    quantile(case when category = 'wt' then alive_in_well end, 0.9) as wt_max
    """).set_alias("donor_min_max")
    donor_bounds = donor_min_max.project(
        "donor as bound_donor, greatest(mut_min, wt_min) as lower_bound, least(mut_max, wt_max) as upper_bound"
    )
    return combined_pairs.join(
        donor_bounds,
        how="inner",
        condition="donor=bound_donor and alive_in_well >= lower_bound and alive_in_well <= upper_bound"
    )


def compare(rel, mutation, drug, dose, features):
    """
    Generate a duckdb relation that has WT and mutant pairs for a given feature
    Columns:
        alive_in_well
        group_id
        plate_barcode
        category ('wt' or 'mut')
        donor
        condition ('WT-<type>' or mutation)
        feature
    """
    wt_filter = analysis.wt_filter
    featurestr = ", ".join(features)
    mut_query = rel.filter(
                f"cell_line_edit_description = '{mutation}' and compound_name='{drug}' and concentration_uM={dose}"
            ).set_alias("mut")
    mut_pairs = mut_query.project("donor_registry_id, group_id").distinct().set_alias("mut_pairs")
    wt_paired = rel.filter(
        f"disease_category in ({wt_filter}) and compound_name='{drug}' and concentration_uM={dose}"
    ).set_alias("wt").join(
        mut_pairs,
        how="inner",
        condition="""
        wt.donor_registry_id=mut_pairs.donor_registry_id and wt.group_id=mut_pairs.group_id
        """
    ).set_alias("wt_paired")
    combined_pairs = wt_paired.project(
        f"""
        plate_barcode,
        alive_in_well,
        group_id,
        donor_registry_id as donor,
        'wt' as category,
        'WT' as condition,
        {featurestr},
        """
    ).union(mut_query.project(f"""
        plate_barcode,
        alive_in_well,
        group_id,
        donor_registry_id as donor,
        'mut' as category,
        '{mutation}' as condition,
        {featurestr},
        """))
    return combined_pairs


def parse_mutant_level_model(model, threshold: float = 0.05):
    hypothesis = f"(C(category)[mut]) = (C(category)[wt])"
    ttest = model.t_test(hypothesis)
    res = dict()
    res["donor_interact_pval"] = ttest.pvalue
    res["donor_interact_coeff"] = ttest.effect[0]
    res["significant"] = ttest.effect[0] < threshold
    return res


def parse_model(model, keys: dict[str, str], threshold: float = 0.05):
    interact_rows = []
    donors = set([x.split("[")[-1].strip("]") for x in model.params.index if "C(donor)" in x])
    for donor in donors:
        interact_row = keys.copy()
        hypothesis = f"(C(category)[mut]:C(donor)[{donor}]) = (C(category)[wt]:C(donor)[{donor}])"
        try:
            ttest = model.t_test(hypothesis)
        except Exception as e:
            print(model.summary())
            raise e
        interact_row["donor_interact"] = donor
        interact_row["donor_interact_pval"] = ttest.pvalue
        interact_row["donor_interact_coeff"] = ttest.effect[0]
        interact_row["significant"] = ttest.effect[0] < threshold
        interact_row["lower_ci"] = ttest.conf_int()[0][0]
        interact_row["upper_ci"] = ttest.conf_int()[0][1]
        interact_rows.append(interact_row)
    return interact_rows


# model tasks
def donor_ttest(
    donordata: pd.DataFrame,
    feature: str,
    metadata: dict[str, Any]
):
    wt = donordata[donordata.category == "wt"][feature].values
    mut = donordata[donordata.category == "mut"][feature].values
    tstat, pval = ttest_ind(wt, mut)
    ttrow = {
        "model": f"ttest-{feature}",
        "donor_interact_coeff": np.mean(mut) - np.mean(wt),
        "donor_interact_pval": pval,
        "significant": pval < 0.05,
        "lower_ci": 0,
        "upper_ci": 0,
    }
    ttrow.update(metadata)
    return ttrow


def matched_density(
    matched_df: pd.DataFrame, feature: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    try:
        model_matched_fixed = (
            smf.ols(f"{feature} ~ C(category) + C(group_id) + C(donor) - 1",
                    data=matched_df).fit()
        )
        density_fixed_row = parse_mutant_level_model(model_matched_fixed)
        density_fixed_row.update(metadata)
        density_fixed_row["model"] = f"paired-fixed-{feature}"
    except Exception as e:
        print(f"failed to run paired model for {metadata}. {e}")
        density_fixed_row = metadata
    return density_fixed_row


def donor_level_fixed(
    df: pd.DataFrame, feature: str, metadata=dict[str, Any]
) -> list[dict[str, Any]]:
    try:
        model = (
            smf.ols(f"{feature} ~ C(category):C(donor) + alive_in_well - 1",
                    data=df).fit()
        )
        parsed = parse_model(model, metadata)
    except Exception as e:
        print(f"failed to run donor level linear model for {metadata}. {e}")
        parsed = []
    return parsed


def mutant_level_fixed(df: pd.DataFrame, feature: str, metadata=dict[str, Any]) -> dict[str, Any]:
    try:
        # do fixed effect
        model_fixed = (
            smf.ols(f"{feature} ~ C(category) + C(group_id) + C(donor) + alive_in_well - 1",
                    data=df).fit()
        )
        fixed_row = parse_mutant_level_model(model_fixed)
        fixed_row.update(metadata)
        fixed_row["model"] = f"linear-fixed-{feature}"
    except Exception as e:
        print(f"failed to run mutant level linear model for {metadata}. {e}")
        fixed_row = metadata
    return fixed_row


@redun.task()
def save_stats(
    output_path: str,
    donor_level: list[dict[str, Any]],
    mutant_level: list[dict[str, Any]],
    ttests: list[dict[str, Any]],
    all_wells: pd.DataFrame,
    passing_wells: pd.DataFrame
) -> list[redun.File]:
    # save to output path
    def _todf_save(stats: list[str, Any], basename: str):
        this_df = pd.DataFrame(stats)
        this_file = redun.File(os.path.join(output_path, basename))
        this_df.to_csv(this_file.path)
        return this_file
    stats = [donor_level, mutant_level, ttests]
    basenames = [
        "donor_level_model_stats.csv",
        "mutant_level_model_stats.csv",
        "paired_density_ttests.csv"
    ]
    files = [_todf_save(l, b) for l, b in zip(stats, basenames)]
    all_wells_file = redun.File(os.path.join(output_path, "all_wells.csv"))
    all_wells.to_csv(all_wells_file.path)
    passing_wells_file = redun.File(os.path.join(output_path, "passing_wells.csv"))
    passing_wells.to_csv(passing_wells_file.path)
    files.extend([all_wells_file, passing_wells_file])
    return files


@redun.task()
def main(output_path: str = fals_output_path):
    """
    Run stats for all pairwise mut/wt sets
    Features are STMN intensity and TDP-43 C/N ratio

    Models:
    * Density fixed effect
    * Matched density
    * Posthoc ttest for fixed effect model

    All stats are uncorrected for multiple comparisons
    Correction performed in notebooks before reporting and plotting
    """

    # build an Analysis object: sets controls, filters, etc
    all_grps = analysis.Analysis(
        groups=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        table_path=table_path,
        additional_well_filters=well_filters
    )

    # size and donor line filter
    # this is pretty small so let's pull in memory then recast to duckdb rel for easier manipulation
    passing_wells_df = all_grps.target_wells.filter(
        "(alive_in_well >= 80.0) and donor_registry_id not in ('Dins603')"
    ).to_df()
    passing_wells_df.loc[:, "experiment"] = passing_wells_df.group_id.map(grp2exp)
    passing_wells = duckdb.from_df(passing_wells_df)

    # we exclude these C9 lines as they were added in later experiments
    skip_c9_donors = "('Dins522','Dins523','Dins527','Dins530','Dins532','Dins533')"
    # response variables
    features = [norm_tdp43_ratio, norm_stmn2, norm_neurite_stmn2, "well_neurite_area"]
    # build up outputs
    mutant_level = list()
    donor_level = list()
    ttests = list()
    for exp_grp, grp_filter in exp2grps.items():
        passing_wells_expgrp = passing_wells.filter(
            f"group_id in {grp_filter} and donor_registry_id not in {skip_c9_donors} and plate_barcode !='PF3846_2'")
        for mutation in mutations_to_score:
            for drug, dose in dose_pairs:
                rel = compare(passing_wells_expgrp, mutation, drug, dose, features)
                den_rel = density_compare(passing_wells_expgrp, mutation, drug, dose, features)
                # linear models
                for feature in features:
                    row = {
                        "experiment": exp_grp,
                        "compound": drug_labels[drug],
                        "model": f"linear-{feature}",
                        "mutation": mutation,
                    }
                    # fixed effect density models
                    rel_df = rel.order("category").df()
                    donor_level.extend(donor_level_fixed(df=rel_df, feature=feature, metadata=row))
                    mutant_level.append(
                        mutant_level_fixed(df=rel_df, feature=feature, metadata=row)
                    )
                    # check to see if there is data to run paired
                    if not den_rel or den_rel.count('*').fetchone()[0] < 1:
                        continue
                    # do ttest on density matched across donors
                    matched_df = den_rel.order("category").df()
                    mutant_level.append(
                        matched_density(matched_df=matched_df, feature=feature, metadata=row)
                    )
                    # do ttest between donors
                    for donor, donordata in matched_df.groupby("donor"):
                        other_metadata = {
                            "experiment": exp_grp,
                            "mutation": mutation,
                            "compound": drug_labels[drug],
                            "donor_interact": donor,
                        }
                        ttests.append(
                            donor_ttest(
                                donordata=donordata,
                                feature=feature,
                                metadata=other_metadata
                            )
                        )

    return save_stats(
        output_path=output_path,
        donor_level=donor_level,
        mutant_level=mutant_level,
        ttests=ttests,
        all_wells=all_grps.target_wells.df(),
        passing_wells=passing_wells_df
    )
