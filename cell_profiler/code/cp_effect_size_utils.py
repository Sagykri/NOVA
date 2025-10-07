# ################### #
# Packages
# ################### #

import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

# ################### #
# Global variables
# ################### #


CP_OUTPUTS_FOLDER = os.path.join(os.getenv("NOVA_HOME"), 'cell_profiler', 'outputs', 'cell_profiler_RUNS', 'Final_cp_analysis')
REQUIRED_FILES = ['Image.csv', 'Pbodies.csv', 'Cytoplasm.csv']


Pbodies_measures = [
    "AreaShape_Area",
    "AreaShape_Compactness",
    "AreaShape_Eccentricity",
    "AreaShape_EquivalentDiameter",
    "AreaShape_FormFactor",
    "AreaShape_HuMoment_0",
    "AreaShape_HuMoment_1",
    "AreaShape_HuMoment_2",
    "AreaShape_HuMoment_3",
    "AreaShape_HuMoment_4",
    "AreaShape_HuMoment_5",
    "AreaShape_HuMoment_6",
    "AreaShape_MeanRadius",
    "AreaShape_MedianRadius",
    "AreaShape_NormalizedMoment_0_2",
    "AreaShape_NormalizedMoment_0_3",
    "AreaShape_NormalizedMoment_1_1",
    "AreaShape_NormalizedMoment_1_2",
    "AreaShape_NormalizedMoment_1_3",
    "AreaShape_NormalizedMoment_2_0",
    "AreaShape_NormalizedMoment_2_1",
    "AreaShape_NormalizedMoment_2_2",
    "AreaShape_NormalizedMoment_2_3",
    "AreaShape_NormalizedMoment_3_0",
    "AreaShape_NormalizedMoment_3_1",
    "AreaShape_NormalizedMoment_3_2",
    "AreaShape_NormalizedMoment_3_3",
    
]
PB_in_cyto_measures = [
    "Math_DCP1A_PB_over_cyto",
    "Math_Texture_Contrast_DCP1A_pb_only_15",
    "Math_Texture_Contrast_DCP1A_pb_only_3",
    "Math_Texture_Contrast_DCP1A_pb_only_5",
    "Math_Texture_Contrast_DCP1A_pb_only_9",
    "Math_Texture_Entropy_DCP1A_pb_only_15",
    "Math_Texture_Entropy_DCP1A_pb_only_3",
    "Math_Texture_Entropy_DCP1A_pb_only_5",
    "Math_Texture_Entropy_DCP1A_pb_only_9",
    "Math_Texture_Homogeneity_DCP1A_pb_only_15",
    "Math_Texture_Homogeneity_DCP1A_pb_only_3",
    "Math_Texture_Homogeneity_DCP1A_pb_only_5",
    "Math_Texture_Homogeneity_DCP1A_pb_only_9",

    
]

measures_to_plot = [
    'num_pb', 
    'mean_AreaShape_Area', 
    'mean_AreaShape_Compactness',
    'mean_AreaShape_Eccentricity', 
    'mean_AreaShape_EquivalentDiameter',
    'mean_AreaShape_FormFactor', 
    'mean_AreaShape_MeanRadius',
    'mean_AreaShape_MedianRadius', 
    'mean_AreaShape_NormalizedMoment_1_1',

    'mean_Math_DCP1A_PB_over_cyto', 'mean_Math_LSM14A_PB_over_cyto',
    'mean_Math_Texture_Contrast_DCP1A_pb_only_3', 'mean_Math_Texture_Contrast_LSM14A_pb_only_3',
    'mean_Math_Texture_Entropy_DCP1A_pb_only_3', 'mean_Math_Texture_Entropy_LSM14A_pb_only_3',
    'mean_Math_Texture_Homogeneity_DCP1A_pb_only_3', 'mean_Math_Texture_Homogeneity_LSM14A_pb_only_3',




]

# ################### #
# Functions
# ################### #

def validate_cp_files(cp_files, marker_path):
    for f in REQUIRED_FILES: assert f in cp_files, f"File {f} is missing in {marker_path}"

def extract_path_parts(path):
    path = path.replace("file:///V:/", "")
    #print(path)
    parts = os.path.normpath(path).split(os.sep)
    return pd.Series({
        'batch': parts[-6],
        'cell_line': parts[-5],
        'condition': parts[-3],  
        'rep': parts[-2]  
    })

def assign_plate_Coyne_new(cell_line):

    # Subjects mapping to plates
    plate3 = ['Ctrl-EDi037', 'C9-CS8RFT', 'SALSPositive-CS7TN6', 'SALSNegative-CS6ZU8']
    plate2 = ['Ctrl-EDi029', 'C9-CS7VCZ', 'SALSPositive-CS4ZCD', 'SALSNegative-CS0JPP']
    plate1 = ['Ctrl-EDi022', 'C9-CS2YNL', 'SALSPositive-CS2FN3', 'SALSNegative-CS0ANK']

    if cell_line in plate1:
        return 'plate1'
    elif cell_line in plate2:
        return 'plate2'
    elif cell_line in plate3:
        return 'plate3'
    else:
        print(f"Subject not found in plates: {cell_line}")
        return None

def assign_gene_group_Coyne_new(cell_line):
    
    if 'C9' in cell_line:
        return 'C9'
    elif 'Ctrl' in cell_line:
        return 'Ctrl'
    elif 'SALSNegative' in cell_line:
        return 'SALSNegative'
    elif 'SALSPositive' in cell_line:
        return 'SALSPositive'
    else:
        print(f"Subject gene group not found: {cell_line}")
        return None
    
def assign_gene_group_Coyne_old(cell_line):
    
    if 'c9orf72ALSPatients' in cell_line:
        return 'C9'
    elif 'Controls' in cell_line:
        return 'Ctrl'
    elif 'sALSNegativeCytoTDP43' in cell_line:
        return 'SALSNegative'
    elif 'sALSPositiveCytoTDP43' in cell_line:
        return 'SALSPositive'
    else:
        print(f"Subject gene group not found: {cell_line}")
        return None
    
    
def merge_on_group(df1, df2, group_by_columns, how='outer'):
    #print(df1.head(), df2.head())
    return pd.merge(df1, df2, on=group_by_columns, how=how)

def collect_cp_results_by_cell_line(analysis_type, include_condition=False, validate=True):        
    # holds the paths to marker folders for each cell line (key=cell line, value=paths)
    paths_by_cell_line = defaultdict(list)
    cp_files_by_cell_line = defaultdict(list)

    # pattern to find marker folders
    pattern = os.path.join(CP_OUTPUTS_FOLDER, analysis_type, '*', '*', '*', '*', '*', '*')
    # store marker folders by cell line
    for marker_path in glob.glob(pattern):
        
        marker = os.path.basename(marker_path)    
        cell_line = Path(marker_path).resolve().parents[3].name
        if include_condition: condition = Path(marker_path).resolve().parents[1].name
        
        # make sure all required files exist in marker_path
        if validate: validate_cp_files(os.listdir(marker_path), marker_path)
        
        # store paths to marker folders
        if include_condition: 
            paths_by_cell_line[cell_line+"_"+condition].append(marker_path)
        else:
            paths_by_cell_line[cell_line].append(marker_path)
            
    return paths_by_cell_line

def load_cp_results(paths_by_cell_line, REQUIRED_FILES):
    
    cp_data = {}


    for cell_line, marker_paths in paths_by_cell_line.items():
        print(f"number of subjects from cell line {cell_line}: {len(marker_paths)}")
        cp_data[cell_line] = {}
        for cp_file_name in REQUIRED_FILES:
            try:
                dfs = []
                for marker_path in marker_paths:
                    # Load CP results files
                    #print(marker_path, cp_file_name)
                    df = pd.read_csv(os.path.join(marker_path, cp_file_name))
                    
                    # Add batch, cell line, condition, rep
                    df['batch'], df['cell_line'], df['condition'], df['rep'] = extract_path_parts(marker_path)
                    dfs.append(df)

                ##DEBUG for d in dfs: print(cell_line, cp_file_name, d.shape)

                cp_data[cell_line][cp_file_name] = pd.concat(dfs, ignore_index=True)
                ##print(cell_line, file_name, cp_data[cell_line][file_name].shape)
            except FileNotFoundError as e:
                print(e)
                pass
    return cp_data

def filter_zero_nuclei(image_df, nuc_count_col, cell_line):
    """Remove site images with 0 nuclei and print how many were removed."""
    zero_nuc_mask = image_df[nuc_count_col] == 0
    num_removed = zero_nuc_mask.sum()
    total_rows = len(image_df)

    if num_removed > 0:
        print(f"⚠️ {cell_line}: Removed {num_removed} of {total_rows} site images with 0 nuclei.")

    return image_df[~zero_nuc_mask].copy()

def get_features_per_image(cp_data, cell_line):

    # ------------------ #
    #'Image.csv'
    # Each row in this file is summation (e.g., in Alyssa’s data: 5 rows for 5 images per rep1, in one patient)
    # ------------------ #
    
    # Get the relevant per-site measurements 
    image_df = cp_data[cell_line]['Image.csv']
    
    # Choose correct nucleus count column
    nuc_count_col = 'Count_nucleus' if 'Count_nucleus' in image_df.columns else 'Count_DAPI'
    
    # Filter out 0-nuclei images
    image_df = filter_zero_nuclei(image_df, nuc_count_col, cell_line)
    
    # Compute num_pb safely
    image_df = image_df.assign(
        num_pb=np.where(
            image_df[nuc_count_col] == 0,
            0,
            image_df['Count_Pbodies'] / image_df[nuc_count_col]
        )
    )
    return image_df


def get_aggregated_features_per_image(cp_data, cell_line, group_by_columns, cp_file_name, measurement_cols, agg_function="mean"):
    """
    Aggregates features per image (site-level), e.g., computing mean values across object-level rows.
    """
    # Determine label prefix
    agg_label = agg_function if isinstance(agg_function, str) else agg_function.__name__
    agg_label += "_"

    # Get relevant measurements
    df = cp_data[cell_line][cp_file_name][measurement_cols + group_by_columns]
    
    # Replace inf value with None
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Prepare for aggregation
    df_grouped = df[measurement_cols + group_by_columns].groupby(group_by_columns)

    # Pass string instead of function to avoid warning
    agg_to_use = agg_function if isinstance(agg_function, str) else agg_function.__name__

    # Aggregate and rename
    df_site_agg = df_grouped.agg(agg_to_use).reset_index()
    df_site_agg = df_site_agg.rename(columns={
        col: agg_label + col for col in measurement_cols
    })
    return df_site_agg



def collect_all_features(cp_data, group_by_columns, PB_in_cyto_measures=PB_in_cyto_measures):

    cp_data = cp_data.copy()
    
    dfs = []
    for cell_line in cp_data:
        
        # Features from 'Image.csv'
        image_df = get_features_per_image(cp_data, cell_line)
        # keep only relevant cols
        image_df = image_df[group_by_columns + ['num_pb']]

        #  'Pbodies.csv'
        pb_site_means = get_aggregated_features_per_image(
            cp_file_name='Pbodies.csv', 
            measurement_cols=Pbodies_measures, 
            cp_data=cp_data, 
            cell_line=cell_line, 
            group_by_columns=group_by_columns, 
            agg_function=np.mean
         )

        # 'Cytoplasm.csv'
        pb_cyto_site_means = get_aggregated_features_per_image(
            cp_file_name='Cytoplasm.csv', 
            measurement_cols=PB_in_cyto_measures, 
            cp_data=cp_data, 
            cell_line=cell_line, 
            group_by_columns=group_by_columns, 
            agg_function=np.mean
         )

        # merge the three dataframes into one
        print(cell_line, image_df.shape, pb_site_means.shape, pb_cyto_site_means.shape)
        merged = merge_on_group(image_df, pb_site_means, group_by_columns)
        print(merged.shape)
        merged = merge_on_group(merged, pb_cyto_site_means, group_by_columns)
        print(merged.shape)

        #print(merged.shape)
        dfs.append(merged)

    cp_measurements = pd.concat(dfs, ignore_index=True)
    print("Shape after merging is:", cp_measurements.shape)        
    return cp_measurements


def run_analysis_generate_report(df, feature_columns, group_col, batch_col, output_dir="mixedlm_report", cov_type=None):
    """
        Mixed model: learns a variance of patient-level effects → generalizable to new patients.
        Fixed model: fits a separate intercept for each patient → not generalizable, just adjusts for each patient's bias.
                     Fixed captures batch effects (e.g., individual differences between patients), 
                     but does not treat patients as a population-level random effect.
                     so the effect are means *adjusted* for per-patient effects.
    """
    
    # NOTE!! this function is doing mixedlm.fit(reml=False, method='lbfgs', disp=False)

    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm, ols
    from statsmodels.stats.multitest import multipletests
    
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    report_results = []

    for feature in feature_columns:
        print(f"\n\n\nAnalysing CP feature: {feature}")
        sub_df = df[[feature, group_col, batch_col]].dropna().copy()
        sub_df = sub_df.rename(columns={feature: "feature_value"})
        
        convergence_flag = "mixedlm"
        fallback_used = False
        
        try:
            if cov_type=="cluster":
                print(f"⚠️ Fallback to fixed-effects model for feature: {feature} with cov_type=cluster")
    
                convergence_flag = "fallback_ols_fixed_batch_cov_type+cluster"
                fallback_used = True
            
                result = ols(f"feature_value ~ C({group_col})", 
                             data=sub_df).fit(cov_type="cluster", cov_kwds={"groups": sub_df[batch_col]})
                print(result.summary())
            else:



                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")  # Catch all warnings

                    # --- Mixed model: random intercept for batch ---
                    # Treat each patient as a random intercept
                    # Estimate group-level effects (fixed effects of gene group)
                    # Account for intra-patient correlation across site images
                    model = mixedlm(f"feature_value ~ C({group_col})", data=sub_df, groups=sub_df[batch_col])
                    result = model.fit(reml=False, method='lbfgs', disp=False)
                    #print(result.summary())

                    # Print any warnings to the screen
                    for w in wlist:
                        warning_type = type(w.message).__name__
                        print(f"⚠️ Warning during model fit for feature {feature}: {warning_type}: {str(w.message)}")

                    # If random-effect variance ~ 0, force fixed-batch fallback
                    if hasattr(result, "cov_re") and result.cov_re.iloc[0, 0] < 1e-6:
                        raise ValueError("Random effect variance is near zero.")
                    else:
                        # show results of mixed effect model
                        print(result.summary())

        
        except Exception as e:
            print(f"❌ {e} — Unable to fit random intercept (e.g., low variance or convergence issue)")
            print(f"⚠️ Fallback to fixed-effects model for feature: {feature}")
    
            convergence_flag = "fallback_ols_fixed_batch"
            fallback_used = True

            # --- Fallback: Fixed-effects linear model (OLS) with batch as FIXED effect  ---
            # We fit "feature_value ~ C(gene_group) + C(patient_id)" so the model estimates:
            ### # An Intercept: The baseline value for a reference patient in the reference group (WT).
            ### # Group-level fixed effects: How each gene group differs from WT.
            ### # Patient-level fixed effects: How each individual patient differs from the reference patient.
            # Fixed OLS includes dummy variables for both gene_group and patient_id. 
            # C(patient_id) is a control variable for technical variation.
            # we report only the gene group coefficients and ignore the patient-specific coefficients.
            # captures batch effects (e.g., individual differences between patients), but does not treat patients as a population-level random effect.
            
            result = ols(f"feature_value ~ C({group_col}) + C({batch_col})", data=sub_df).fit()
            print(result.summary())
            
            
        # Print interpretation
        print_mixedlm_conclusions(result)

        
        
        # --- Collect outputs (only for GROUP effects) ---
        report_results = collect_results(result, report_results, group_col, feature, fallback_used, convergence_flag)
        
    # FDR correction and save to all results to single CSV
    results_df = pd.DataFrame(report_results)

    results_df_wo_intercept = results_df[results_df['comparison'] != "Intercept"] # Sagy 5.10.25 - since we don't want to correct for intercept

    # FDR across all tests
    if not results_df_wo_intercept.empty:
        _, results_df_wo_intercept["fdr_pval_global"], _, _ = multipletests(results_df_wo_intercept["pval"], method="fdr_bh")
    else:
        results_df_wo_intercept["fdr_pval_global"] = []

    results_df = pd.concat([results_df[results_df['comparison'] == "Intercept"], results_df_wo_intercept], ignore_index=True)

    if output_dir is not None:
        save_path = os.path.join(output_dir, "mixedlm_results.csv")
        print(f"\n\nSaving all results to: {save_path}")
        results_df.to_csv(save_path, index=False)

    return results_df


def collect_results(result, report_results, group_col, feature, fallback_used, convergence_flag):
    
    # If random-effect variance ~ 0, force fixed-batch fallback
    if fallback_used:
        group_var = np.nan  # no random variance in OLS
    else:
        group_var = float(result.cov_re.iloc[0, 0])

    fe_params = result.params
    pvals = result.pvalues
    ci = result.conf_int()
    resid_var = result.mse_resid if hasattr(result, 'mse_resid') else result.scale
    model_stats = {
                    "aic": result.aic if hasattr(result, 'aic') else np.nan,
                    "bic": result.bic if hasattr(result, 'bic') else np.nan,
                    "loglik": result.llf if hasattr(result, 'llf') else np.nan,
                    "r_squared": result.rsquared if hasattr(result, 'rsquared') else np.nan  # Only for OLS
                }

    # Keep only coefficients for C(group_col), exclude Intercept and any C(batch_col) terms
    group_prefix = f"C({group_col})"
    effect_names = [name for name in fe_params.index
                          if name.startswith(group_prefix)]

    sig_map = lambda p: "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    pvals_dict = {}


    # Save the Intercept as a separate row in the results
    report_results.append({
        "feature": feature,
        "comparison": "Intercept",
        "effect_size": fe_params["Intercept"],
        "pval": float(pvals["Intercept"]),
        "ci_lower": float(ci.loc["Intercept"][0]),
        "ci_upper": float(ci.loc["Intercept"][1]),
        "group_var": group_var,
        "residual_var": float(resid_var),
        "significance": sig_map(pvals["Intercept"]),
        "fit_status": convergence_flag,
        "used_fixed_model": fallback_used,
        "aic": model_stats["aic"],
        "bic": model_stats["bic"],
        "loglik": model_stats["loglik"],
        "r_squared": model_stats["r_squared"]
    })

    for name in effect_names:
        coef = fe_params[name]
        pval = pvals[name]
        ci_low, ci_high = ci.loc[name]

        # Extract the treatment level from "C(group_col)[T.<level>]"
        m = re.search(r"\[T\.(.*)\]", name)
        comparison_label = m.group(1) if m else name  # fallback to raw name

        pvals_dict[comparison_label] = pval
        report_results.append({
            "feature": feature,
            "comparison": comparison_label,
            "effect_size": coef,
            "pval": float(pval),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
            "group_var": group_var,
            "residual_var": float(resid_var),
            "significance": sig_map(pval),
            "fit_status": convergence_flag,
            "used_fixed_model": fallback_used,
            "aic": model_stats["aic"],
            "bic": model_stats["bic"],
            "loglik": model_stats["loglik"],
            "r_squared": model_stats["r_squared"]
        })


    return report_results
        

def print_mixedlm_conclusions(result, reference_group="Ctrl", alpha=0.05):
    
    
    # Extract fixed effect estimates
    coefs = result.params
    conf_int = result.conf_int()
    stderr = result.bse
    pvals = result.pvalues

    print("\n### Fixed Effects: Gene Group Differences from", reference_group)
    for param in coefs.index:

        if param == "Intercept":
            print(f"- {reference_group}: estimated average = {coefs[param]:.2f}")
        elif 'group' in param:
            
            # Extract group name safely
            m = re.search(r"\[T\.(.*?)\]", param)
            group = m.group(1) if m else param

            est = coefs[param]
            se = stderr[param]
            p = pvals[param]
            ci_low, ci_high = conf_int.loc[param]
            sig = "✔️ Significant" if p < alpha else "✖️ Not significant"
            print(f"- {group}: {est:+.5f} units vs {reference_group} "
                  f"(p = {p:.5f}, 95% CI: [{ci_low:.5f}, {ci_high:.5f}]) → {sig}")

    print("\n### Random Effects (Batch-level / Intra-patient variability)")
    # Check if result includes random effects (i.e., it's a MixedLMResults object)
    if hasattr(result, "cov_re") and isinstance(result.cov_re, pd.DataFrame):
        try:
            group_var = result.cov_re.iloc[0, 0]
            print(f"- Estimated variance across batches: {group_var:.5f}")
            if group_var > 1e-4:
                print("- ✅ Indicates batch-level variability was successfully modeled.")
            else:
                print("- ⚠️ Very low variance — batch-level differences may be minimal or incorrectly specified.")
        except Exception as e:
            print(f"- ⚠️ Could not extract random effects variance: {e}")
    else:
        print("- ℹ️ Model did not include random effects (fixed-effects fallback used).")
        
        


def plot_cp_feature_grouped_by_gene(cp_measurements, cp_feature_col, group_col="gene_group", patient_col="patient_id", color_mapping=None, model_results_df=None):
    
    df = cp_measurements.copy()

    # Define fixed color and label mapping for both group_col and patient_col
    _palette = {}
    groups = df[group_col].astype(str).unique().tolist() + df[patient_col].astype(str).unique().tolist()
    for g in groups:
        if g in color_mapping: _palette[color_mapping[g]['alias']] = color_mapping[g]['color']
    if patient_col=='batch':
        for b in df[patient_col].astype(str).unique().tolist():
            _palette[b]='gray'

    
    # Rename groups to aliases
    label_mapping = {k: v["alias"] for k, v in color_mapping.items() if k in groups}
    df[group_col] = df[group_col].cat.rename_categories(label_mapping)
    df[patient_col] = df[patient_col].astype("category").cat.rename_categories(label_mapping)
   
    sns.set(style="whitegrid", font_scale=1.0)
    plt.figure(figsize=(8, 8))

    # Boxplot
    sns.boxplot(
        data=df,
        x=group_col,
        hue=group_col, #patient_col,
        y=cp_feature_col,
        palette=_palette,
        width=0.3,
        showfliers=False,
        #legend=False
    )

    # Stripplot
    sns.stripplot(
        data=df,
        x=group_col,
        y=cp_feature_col,
        hue=patient_col,
        dodge=True,
        #jitter=True,
        palette=_palette,
        marker='o',
        alpha=0.3,
        legend=True
    )
    # Then move legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(
        unique.values(),
        unique.keys(),
        title=patient_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    # Overlay effect sizes + p-values if model results are available
    if model_results_df is not None:
        # Filter for this feature
        res_df = model_results_df[model_results_df["feature"] == cp_feature_col].copy()
        
        # Rename CSV values
        res_df['comparison'] = pd.Categorical(res_df['comparison'])
        res_df['comparison'] = res_df['comparison'].cat.rename_categories(label_mapping)

        ax = plt.gca()
        xtick_positions = ax.get_xticks()
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]

        for xtick_pos, group_label in zip(xtick_positions, xtick_labels):
            if group_label == xtick_labels[0]:
                continue  # Skip Intercept (reference)

            res_row = res_df[res_df["comparison"] == group_label].copy()
            if not res_row.empty:
                est = res_row["effect_size"].values[0]
                pval = res_row["pval"].values[0]
                annotation = f"Δ={est:+.2f}, p={pval:.3f}"

                ax.text(
                    xtick_pos,
                    df[cp_feature_col].min() * 1.05,
                    annotation,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color="black"
                )
    # Format
    plt.title(f"{cp_feature_col} grouped by {group_col}")
    plt.xlabel("Gene Group")
    plt.ylabel(cp_feature_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # add top space if needed
    plt.show()
