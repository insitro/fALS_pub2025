from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from fals.utils import process, viz
from fals.utils import load
from fals.utils.datasets import get_line_pairs


DMSO = "Dimethyl sulfoxide [MilliporeSigma]"
TARGET_CONCENTRATIONS = {
    "Bortezomib [Tocris]": [0.0012, 0.0032, 0.0078, 0.039],
    "Puromycin dihydrochloride [MilliporeSigma]": [1.7483, 3.4965, 7.992, 9.99],
}

wt_categories = ["Wild-Type", "Engineered Familial Control", "Corrected Familial Mutation"]
wt_filter = ",".join(f"'{c}'" for c in wt_categories)
disease_categories = ["Familial Patient", "Engineered Familial", "Over-Expression Familial"]
disease_filter = ",".join(f"'{c}'" for c in disease_categories)


class Analysis:
    """
    Object to hold config, data and methods
    """

    def __init__(
        self,
        groups: list[int],
        table_path: str = "./tables/",
        # default args
        target_feat: str = "median_cyto_nucleus_mask_ratio_TDP43_pixel_intensity",
        control_cins: str = "Cins1013 | hNIL - 113",  # parent
        additional_well_filters: Optional[str] = None,
        additional_feature_filters: Optional[str] = None,
    ):
        self.groups = groups
        self.table_path = table_path
        self.con = load.get_con()
        # set config
        self.target_feat = target_feat
        self.control_cins = control_cins  # this is actually the parent instance
        self.additional_well_filters = additional_well_filters
        self.additional_feature_filters = additional_feature_filters

        # set the controls and target
        self.set_pairs()

    @property
    def base_wells(self):
        # we always exclude row B
        wells = load.wells_relation(self.con, self.table_path, self.groups).filter(
            "well_position[1] != 'B'"
        )
        if self.additional_well_filters:
            wells = wells.filter(self.additional_well_filters)
        # add workflow
        workflow = load.workflow_relation(self.con, self.table_path)
        wells = wells.join(
            workflow,
            how="left",
            condition="cell_line_instance_registry_id=instance_registry_id_cell_line",
        )
        return wells

    @property
    def base_features(self):
        features = load.features_relation(self.con, self.table_path)
        if self.additional_feature_filters:
            features = features.filter(self.additional_feature_filters)
        return features.join(self.base_wells, how="inner", condition="cell_well_id = well_id")

    @property
    def compounds(self):
        compounds = self.base_wells.project("compound_name").distinct().df()
        return sorted(compounds["compound_name"].to_list())

    @property
    def drug_compounds(self):
        return [drug for drug in self.compounds if drug != DMSO]

    @property
    def cell_lines(self):
        lines = self.base_wells.project("cell_line").distinct().df()
        return sorted(lines["cell_line"].to_list())

    @property
    def mutation_categories(self):
        muts = self.base_wells.project("cell_line_edit_description").distinct().df()
        return sorted(muts["cell_line_edit_description"].to_list())

    def plot_nucleus_mask_distributions(
        self,
        dead_threshold: float = 650,
        clump_threshold: float = 1500,
    ):
        """
        For each plate, plot the distribution of nucleus mask areas (
        """
        grid = viz.sns.displot(
            data=self.base_features.filter("nucleus_mask_area_pixel_sq < 3000")
            .project("plate_barcode,nucleus_mask_area_pixel_sq")
            .df(),
            x="nucleus_mask_area_pixel_sq",
            col="plate_barcode",
            col_wrap=4,
            kind="kde",
        )
        for ax in grid.axes.flat:
            ax.axvline(dead_threshold, ls="--")
            ax.axvline(clump_threshold, ls="--")
        return grid.figure

    def plot_target_feat_distributions(self):
        """
        Plot distribution of target feature broken out by alive
        """
        return viz.sns.displot(
            data=self.base_features.project(f"{self.target_feat}, alive").df(),
            x=self.target_feat,
            hue="alive",
        ).figure

    def set_pairs(self):
        """
        Assign control and target wells based on cell line and treatments
        """
        alive_cells = self.base_features.filter("alive = true and apoptotic = false")
        t1, t2, target = process.extract_technical_controls(
            well_rel=self.base_wells, control_cins=self.control_cins
        )
        tech_1_wells = self.con.from_df(pd.DataFrame.from_dict({"well_id_1": np.array(t1)}))
        tech_2_wells = self.con.from_df(pd.DataFrame.from_dict({"well_id_2": np.array(t2)}))
        target_wells = self.con.from_df(pd.DataFrame.from_dict({"well_id_t": np.array(target)}))
        # set well and features filtered by selected wells
        self.ctrl1 = alive_cells.join(tech_1_wells, how="inner", condition="well_id_1 = well_id")
        self.ctrl2 = alive_cells.join(tech_2_wells, how="inner", condition="well_id_2 = well_id")
        self.target = alive_cells.join(target_wells, how="inner", condition="well_id_t = well_id")
        self.ctrl1_wells = self.base_wells.join(
            tech_1_wells, how="inner", condition="well_id_1 = well_id"
        )
        self.ctrl2_wells = self.base_wells.join(
            tech_2_wells, how="inner", condition="well_id_2 = well_id"
        )
        self.target_wells = self.base_wells.join(
            target_wells, how="inner", condition="well_id_t = well_id"
        )
        # create a set of the wt/mut pairs
        self.line_pairs = set()
        for group in self.groups:
            self.line_pairs.update(get_line_pairs(group))

    def plot_ctrl1(self, feature: Optional[str] = None):
        """
        Box plots of target feature for control and max concentration broken out by plate
        """
        if not feature:
            feature = self.target_feat
        if feature in self.ctrl1_wells.columns:
            rel = self.ctrl1_wells
        else:
            rel = self.ctrl1
        fig = viz.plt.figure(figsize=(8, 4))
        viz.sns.boxplot(
            data=rel.project(f"plate_barcode, compound_name, {feature}")
            .order("plate_barcode")
            .df(),
            hue="compound_name",
            y=feature,
            x="plate_barcode",
            showfliers=False,
        )
        viz.plt.xticks(rotation=90)
        viz.plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        return fig

    def plot_ctrl2(self, feature: Optional[str] = None):
        """
        Box plots at target feature at each target concentration broken out by plate
        This is very busy with multiple groups, so one plot per group
        """
        if not feature:
            feature = self.target_feat
        if feature in self.ctrl2_wells.columns:
            rel = self.ctrl2_wells
        else:
            rel = self.ctrl2
        fig, ax = viz.plt.subplots(1, len(self.groups), figsize=(40, 5), sharey="row")
        for spi, group in enumerate(self.groups):
            if len(self.groups) > 1:
                this_ax = ax[spi]
            else:
                this_ax = ax
            viz.sns.boxplot(
                data=rel.filter(f"group_id = {group}")
                .project(f"plate_barcode, concentration_uM, {feature}")
                .order("plate_barcode")
                .df(),
                x="concentration_uM",
                y=feature,
                hue="plate_barcode",
                showfliers=False,
                ax=this_ax,
            )
            this_ax.title.set_text(f"Group {group}")
        return fig

    def plot_cell_line_dose_grid(
        self,
        feature: str,
        compound: str,
        hue: str,
        cell_lines: Optional[list[str]] = None,
    ):
        """
        Box plot grid of feature with cell line on X, hue, and column per concentration
        """
        concentrations = TARGET_CONCENTRATIONS.get(compound)  # None ok, we use all
        required_cols = ["cell_line", "compound_name", "concentration_uM"]
        required_cols += [feature, hue]
        if feature in self.target_wells.columns:
            rel = self.target_wells
        else:
            rel = self.target

        if cell_lines:
            cell_line_filter = ", ".join(f"'{c}'" for c in cell_lines)
            rel = rel.filter(f"cell_line in ({cell_line_filter})")

        return viz.boxplot_data(
            df_data=rel.project(", ".join(required_cols)).df(),
            feat=feature,
            compound=compound,
            concentrations=concentrations,
            hue=hue,
        )

    def plot_paired_dose_response(
        self,
        feature: str,
        compound: str,
        line_pairs: Optional[list[tuple[str, str]]] = None,
    ):
        """
        Dose response line plots for each selected line pair or all
        """
        concentrations = TARGET_CONCENTRATIONS.get(compound)  # None ok, we use all
        required_cols = ["cell_line", "compound_name", "concentration_uM"]
        required_cols += [feature]
        if feature in self.target_wells.columns:
            rel = self.target_wells
        else:
            rel = self.target
        if not line_pairs:
            line_pairs = list(self.line_pairs)
        return viz.plot_line_pairs(
            df=rel.project(", ".join(required_cols)).df(),
            line_pairs=line_pairs,
            feat=feature,
            compound=compound,
            concentrations=concentrations,
        )

    def plot_dose_response_by(
        self,
        feature: str,
        compound: str,
        cell_lines: list[str],
        by: str = "cell_line",
        palette: Optional[list[str]] = None,
    ):
        """
        Dose response line plots for selected lines
        """
        concentrations = TARGET_CONCENTRATIONS.get(compound)  # None ok, we use all
        required_cols = ["cell_line", "compound_name", "concentration_uM"]
        required_cols += [feature, by]
        required_cols = list(set(required_cols))
        if feature in self.target_wells.columns:
            rel = self.target_wells
        else:
            rel = self.target
        cell_line_filter = ", ".join(f"'{c}'" for c in cell_lines)
        rel = rel.filter(f"cell_line in ({cell_line_filter})")
        return viz.plot_dose_response(
            df=rel.project(", ".join(required_cols)).df().sort_values(by),
            feat=feature,
            compound=compound,
            concentrations=concentrations,
            by=by,
            palette=palette,
        )

    def plot_normalized_dose_response(self, mutation: str, feature: str, method: str = "plate"):
        fig, ax = viz.plt.subplots(1, 4, figsize=(24, 4), sharey="row")
        if method == "plate_matched":
            normed = normalize_to_within_plate_wt(
                self.target_wells.filter("alive_in_well >= 80.0"), feature
            )

        elif method == "density_matched":
            normed = normalize_density_matched(
                self.target_wells.filter("alive_in_well >= 80.0"), feature
            )
        elif method == "workflow_matched":
            normed = normalize_to_workflow_matched_wt(
                self.target_wells.filter("alive_in_well >= 80.0"), feature
            )
        else:
            raise ValueError(f"{method} is not a valid normalization method")

        for i, drug in enumerate(self.drug_compounds):
            rel = normed.filter(
                f"cell_line_edit_description = '{mutation}' and compound_name in ('{drug}', '{DMSO}')"
            ).project(
                f"""
                    cell_line,
                    concentration_uM,
                    relative_{feature}
                    """
            )
            graph = viz.sns.lineplot(
                data=rel.order("cell_line").df(),
                x="concentration_uM",
                y=f"relative_{feature}",
                hue="cell_line",
                palette=viz.insitro_alternating,
                ax=ax[i],
            )
            ax[i].plot(
                rel.project("concentration_uM").distinct().df()["concentration_uM"].to_list(),
                [0] * len(rel.project("concentration_uM").distinct().df()),
                "k--",
            )
            graph.set(xscale="symlog")
            graph.title.set_text(f"{drug}: {mutation}")
            handles, labels = ax[i].get_legend_handles_labels()
            ax[i].get_legend().remove()
        fig.legend(handles, labels, bbox_to_anchor=(1, 0.90))
        return fig


# static methods
def normalize_to_within_plate_wt(
    rel: duckdb.DuckDBPyRelation, feature: str
) -> duckdb.DuckDBPyRelation:
    """
    Normalize by computing the median for all 'control' wells of the same donor
    """
    norm_rel = (
        rel.filter(f"disease_category in ({wt_filter})").aggregate(f"""
        donor_registry_id as donor_norm,
        compound_name as compound_norm,
        concentration_uM as conc_norm,
        plate_barcode as plate_norm,
        median({feature}) as feature_wt_median,
        stddev({feature}) as feature_wt_std
        """)
    ).set_alias("norm")
    join_cond = """
    donor_registry_id = donor_norm and
    compound_name = compound_norm and
    concentration_uM = conc_norm and
    plate_barcode = plate_norm
    """
    return rel.join(norm_rel, how="inner", condition=join_cond).project(f"""
        *,
        {feature} - feature_wt_median as relative_{feature},
        ({feature} - feature_wt_median) / feature_wt_std as zscore_{feature}
        """)


def normalize_density_matched(
    rel: duckdb.DuckDBPyRelation, feature: str
) -> duckdb.DuckDBPyRelation:
    """
    Look across plates for wells of WT lines of a given donor background
    that have comparable (within 2std) alive_in_well values
    """
    # for each line+cond get alive_in_well median+std
    alive_rel = (
        rel.aggregate("""
        cell_line as cell_line_all,
        donor_registry_id as donor_all,
        compound_name as compound_all,
        concentration_uM as conc_all,
        median(alive_in_well) as alive_in_well_median_all,
        stddev(alive_in_well) as alive_in_well_std_all
        """)
    ).set_alias("alive_summary")
    # filter down wells to WT that have the same donor+cond and similar alive in well
    # aggregate median+std on line+cond
    join_cond = """
    donor_registry_id = donor_all and
    compound_name = compound_all and
    concentration_uM = conc_all and
    abs(alive_in_well - alive_in_well_median_all) <= (2 * alive_in_well_std_all)
    """
    norm_rel = (
        rel.filter(f"disease_category in ({wt_filter})")
        .join(alive_rel, how="inner", condition=join_cond)
        .aggregate(f"""
            cell_line_all,
            compound_name as compound_norm,
            concentration_uM as conc_norm,
            median({feature}) as feature_wt_median,
            stddev({feature}) as feature_wt_std
            """)
    ).set_alias("norm")
    # now that we've aggregated the subset of WT wells for each line+cond, we join back
    final_join_cond = """
    cell_line = cell_line_all and
    compound_name = compound_norm and
    concentration_uM = conc_norm
    """
    return rel.join(norm_rel, how="inner", condition=final_join_cond).project(f"""
        *,
        {feature} - feature_wt_median as relative_{feature},
        ({feature} - feature_wt_median) / feature_wt_std as zscore_{feature},
        """)


def normalize_to_workflow_matched_wt(
    rel: duckdb.DuckDBPyRelation, feature: str
) -> duckdb.DuckDBPyRelation:
    """
    Normalize by computing the median for all 'control' wells of the same donor
    within the same workflow
    """
    norm_rel = (
        rel.filter(f"disease_category in ({wt_filter})").aggregate(f"""
        donor_registry_id as donor_norm,
        compound_name as compound_norm,
        concentration_uM as conc_norm,
        workflow as workflow_norm,
        median({feature}) as feature_wt_median,
        stddev({feature}) as feature_wt_std
        """)
    ).set_alias("norm")
    join_cond = """
    donor_registry_id = donor_norm and
    compound_name = compound_norm and
    concentration_uM = conc_norm and
    workflow = workflow_norm
    """
    return rel.join(norm_rel, how="inner", condition=join_cond).project(f"""
        *,
        {feature} - feature_wt_median as relative_{feature},
        ({feature} - feature_wt_median) / feature_wt_std as zscore_{feature}
        """)
