from typing import Optional, Sequence, Tuple

from duckdb import DuckDBPyRelation
import numpy as np
import pandas as pd


WELL_AGGS = {
    "well_TDP43_ratio": pd.NamedAgg(
        "median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity", "median"
    ),  # cyto/nucleus - original phenotype
    # integrated cyto / integrated cell
    "well_TDP43_cyto_fraction": pd.NamedAgg("TDP43_cyto_fraction", "median"),
    "well_TDP43_nucleus_intensity": pd.NamedAgg(
        "median_nucleus_mask_TDP43_pixel_intensity", "median"
    ),
    "well_STMN2_cell_intensity": pd.NamedAgg("median_cell_mask_STMN2_pixel_intensity", "median"),
    "well_STMN2_neurite_intensity": pd.NamedAgg("mean_neurite_STMN2_pixel_intensity", "median"),
    "well_nucleus_mask_area": pd.NamedAgg("nucleus_mask_area_pixel_sq", "median"),
    "well_cell_mask_area": pd.NamedAgg("cell_mask_area_pixel_sq", "median"),
    "well_cyto_mask_area": pd.NamedAgg("cyto_mask_area_pixel_sq", "median"),
    "well_neurite_area": pd.NamedAgg("neurite_area_pixel_sq", "median"),
}


def parse_ID_col(df: pd.DataFrame) -> pd.DataFrame:
    """Extract metadata from ID column
    example: 'PF123_A03_2_123'
    """

    # add "cell_" so that we don't have to alias when joining
    try:
        df["cell_plate_barcode"], df["cell_well_position"], df["cell_field"], df["cell_id"] = zip(
            *df["ID"].apply(lambda r: r.split("_"))
        )
    except ValueError:
        # the plate has an underscore
        (
            df["cell_plate_barcode_prefix"],
            df["cell_plate_barcode_suffix"],
            df["cell_well_position"],
            df["cell_field"],
            df["cell_id"],
        ) = zip(*df["ID"].apply(lambda r: r.split("_")))
        df["cell_plate_barcode"] = (
            df["cell_plate_barcode_prefix"] + "_" + df["cell_plate_barcode_suffix"]
        )
    # join id
    df.loc[:, "cell_well_id"] = df["cell_plate_barcode"] + ":" + df["cell_well_position"]
    return df


def add_cell_state_cols(
    df_feats: pd.DataFrame, dead_threshold: float, clump_threshold: float
) -> pd.DataFrame:
    """Calculates the number and percentage of cells in each well that are either
    dead, alive, clumped, or apoptotic.

    Parameters
    ----------
    df_feats : pandas.DataFrame
        A DataFrame containing the features of each cell.
    dead_threshold : float
        The maximum nucleus mask area (in pixels^2) for a cell to be considered dead.
    clump_threshold : float
        The minimum nucleus mask area (in pixels^2) for a cell to be considered clumped.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with additional columns for the number and percentage of cells in each well
        that are either dead, alive, clumped, or apoptotic.
    """

    # Calculate whether each cell is dead, clumped, alive, or apoptotic
    df_feats["dead"] = df_feats["nucleus_mask_area_pixel_sq"] < dead_threshold
    df_feats["clump"] = df_feats["nucleus_mask_area_pixel_sq"] > clump_threshold
    df_feats["alive"] = df_feats["nucleus_mask_area_pixel_sq"].between(
        dead_threshold, clump_threshold
    )
    df_feats["apoptotic"] = df_feats["alive"] & (
        df_feats.cell_mask_area_pixel_sq < df_feats.nucleus_mask_area_pixel_sq
    )
    return df_feats


def cell_state_summary(df_feats: pd.DataFrame) -> pd.DataFrame:
    """Calculate stats of cells in each well that are either dead, clumped, alive, or apoptotic

    Returns a dataframe with summary statistics by plate+well
    """
    # Total cells in each state
    cell_states = ["alive", "dead", "clump", "apoptotic"]
    well_stats: pd.DataFrame = df_feats.groupby("cell_well_id")[cell_states].aggregate("sum")
    well_stats = well_stats.reset_index()
    well_stats.rename(columns={c: f"{c}_in_well" for c in cell_states}, inplace=True)
    well_stats["cells_in_well"] = (
        well_stats["alive_in_well"] + well_stats["dead_in_well"] + well_stats["clump_in_well"]
    )

    # Percentage of cells in each well that are either dead, alive, or clumped
    for feat in ["alive", "dead", "clump"]:
        well_stats[f"pct_{feat}_in_well"] = (
            well_stats[f"{feat}_in_well"] / well_stats["cells_in_well"]
        )

    # Percentage of cells in each well that are apoptotic
    well_stats["pct_apoptotic_in_well"] = (
        well_stats["apoptotic_in_well"] / well_stats["alive_in_well"]
    )
    # rename to well_id so we don't add back in a cell_well_id column on the well df
    well_stats.rename(columns={"cell_well_id": "well_id"}, inplace=True)
    return well_stats


def feature_summary(df_feats: pd.DataFrame) -> pd.DataFrame:
    """Calculates the well level median of a set of features

    Filters first on alive, non-apoptotic cells
    """
    alive_cells = df_feats[df_feats["alive"] & ~df_feats["apoptotic"]]
    well_stats: pd.DataFrame = alive_cells.groupby("cell_well_id").aggregate(**WELL_AGGS)
    well_stats = well_stats.reset_index()
    well_stats.rename(columns={"cell_well_id": "well_id"}, inplace=True)
    return well_stats


def normalize(
    well_df: pd.DataFrame,
    control_compound: str = "Dimethyl sulfoxide [MilliporeSigma]",
    control_line: str = "Cins1013",
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Use batch control in DMSO to normalize well level features"""
    baseline_df = well_df[
        (well_df["compound_name"] == control_compound)
        & (well_df["cell_line_registry_id"] == control_line)
    ]
    if not features:
        features = list(WELL_AGGS.keys())
    for feature in features:
        if baseline_df[feature].empty:
            # handle plates w/o DMSO
            baseline_median = 0
        else:
            baseline_median = baseline_df[feature].median()
        well_df.loc[:, f"{feature}_norm"] = well_df[feature] - baseline_median
    # Fraction of wells that are alive out of the total wells plated
    # do this here since we need metadata and summary data
    well_df.loc[:, "pct_alive_of_plated"] = well_df["alive_in_well"] / well_df["total_cells_well"]
    return well_df


def add_TDP43_feats(
    df_feats: pd.DataFrame, quantile_cutoff: float, quantile_cutoff_feat: str
) -> pd.DataFrame:
    """compute TDP43 feats, add them to the dataframe, and remove outliers"""
    for metric in ["mean", "median", "integrated"]:
        df_feats[f"{metric}_cyto_nucleus_mask_ratio_TDP43_pixel_intensity"] = (
            df_feats[f"{metric}_cyto_mask_TDP43_pixel_intensity"]
            / df_feats[f"{metric}_nucleus_mask_TDP43_pixel_intensity"]
        )
        df_feats[f"{metric}_cell_nucleus_mask_ratio_TDP43_pixel_intensity"] = (
            df_feats[f"{metric}_cell_mask_TDP43_pixel_intensity"]
            / df_feats[f"{metric}_nucleus_mask_TDP43_pixel_intensity"]
        )
        df_feats[f"{metric}_neurite_cell_mask_ratio_TDP43_pixel_intensity"] = (
            df_feats[f"{metric}_neurite_TDP43_pixel_intensity"]
            / df_feats[f"{metric}_cell_mask_TDP43_pixel_intensity"]
        )
    # normalize median ratio by area ratio of cyto/nucleus
    df_feats["median_area_norm_cyto_nucleus_mask_ratio_TDP43_pixel_intensity"] = df_feats[
        "median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity"
    ] * (df_feats["cyto_mask_area_pixel_sq"] / df_feats["nucleus_mask_area_pixel_sq"])
    # cyto fraction
    df_feats["TDP43_cyto_fraction"] = (
        df_feats["integrated_cyto_mask_TDP43_pixel_intensity"]
        / df_feats["integrated_cell_mask_TDP43_pixel_intensity"]
    )
    df_feats = df_feats[
        df_feats[quantile_cutoff_feat]
        <= df_feats[quantile_cutoff_feat].quantile(q=quantile_cutoff)
    ]
    return df_feats


def extract_technical_controls(
    well_rel: DuckDBPyRelation,
    control_cins: str,
    random_state: int = 2712,
    control_compound: str = "Dimethyl sulfoxide [MilliporeSigma]",
    target_compound: str = "Bortezomib [Tocris]",
    density_target: float = 4920.0,
    concentrations: Sequence[float] = (
        0.0007,
        0.0011,
        0.0017,
        0.0029,
        0.0049,
        0.0078,
        0.0117,
        0.0254,
        0.039,
    ),
) -> Tuple[list[str], list[str], list[str]]:
    """
    Extracts technical controls as a list of well ids

    Technical controls per each plate:
    Technical control 1 is 6 wells of DMSO and 6 wells of max Bortezomib concentration
    Technical control 2 is 2x 9 increasing concentrations of Bortezomib.

    Parameters
    ----------
    well_rel : DuckDBPyRelation
        DuckDBPyRelation of well level metadata
    control_cins: str
        Control cell_line_registry_id line.
    random_state: int
        Sampler seed.
    control_compound: str = "Dimethyl sulfoxide [MilliporeSigma]"
        Name of the compound used as a control
    target_compound: str = "Bortezomib [Tocris]"
        Name of the compound used as a target
    density_target: float = 4920.0
        Restrict to wells with total_cells_well
    concentrations: Sequence[float] = (
        0.0007, 0.0011, 0.0017, 0.0029, 0.0049, 0.0078, 0.0117, 0.0254, 0.039
        )
        Set of concentrations to include in technical control 2

    Returns
    -------
    Tuple[list[str], list[str], list[str]]
        A tuple with lists of well ids for tech controls 1, 2 and targets
    """
    rel_controls = well_rel.filter(f"""
    parent_instance_name = '{control_cins}'
    AND total_cells_well = {density_target}
    """)

    # technical control 1: 6 random wells from control and max concentration of target
    max_target = (
        rel_controls.filter(f"compound_name = '{target_compound}'").max("concentration_uM")
    ).fetchone()[0]
    assert max_target in concentrations
    ctr_df = (
        rel_controls.filter(f"compound_name = '{control_compound}'")
        .project("plate_barcode, well_id")
        .order("well_id")  # otherwise random seed is pointless
        .distinct()  # just to be safe, though this should be unique on well_id
    ).df()
    target_df = (
        rel_controls.filter(f"compound_name = '{target_compound}'")
        .project("plate_barcode, well_id, concentration_uM")
        .order("well_id")
        .distinct()
    ).df()
    tech_1 = []
    tech_2 = []
    picked = set()
    np.random.seed(random_state)
    for plate, ctr_plate in ctr_df.groupby("plate_barcode"):
        candidates = set(ctr_plate["well_id"].to_list())
        candidates -= picked
        replace = len(candidates) < 6
        chosen = list(np.random.choice(list(candidates), 6, replace=replace))
        picked.update(chosen)
        tech_1.extend(chosen)
    for (plate, conc), target_plate in target_df.groupby(["plate_barcode", "concentration_uM"]):
        if conc not in concentrations:
            continue
        candidates = set(target_plate["well_id"].to_list())
        candidates -= picked
        if conc == max_target:
            replace = len(candidates) < 6
            chosen = list(np.random.choice(list(candidates), 6, replace=replace))
            picked.update(chosen)
            tech_1.extend(chosen)
            candidates -= set(chosen)
        replace = len(candidates) < 2
        chosen = list(np.random.choice(list(candidates), 2, replace=replace))
        picked.update(chosen)
        tech_2.extend(chosen)
    assert set(tech_1) & set(tech_2) == set(), "Found overlap between technical controls"
    # subtract technical controls from all valid as target
    target = set(well_rel.project("well_id").distinct().df()["well_id"].to_list())
    target -= set(tech_1 + tech_2)
    return tech_1, tech_2, list(target)
