import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import sys

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from multiprocessing import Pool, cpu_count

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import pyarrow.parquet as pq

from matplotlib.backends.backend_pdf import PdfPages 

import gc
gc.collect()

# 30 sec

sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
from ExternalFunctions import nice_string_output, add_text_to_ax
setMplParam()


# Now import from Enum
sys.path.append('/groups/icecube/cyan/factory/DOMification')
from Enum.Flavour import Flavour
from Enum.EnergyRange import EnergyRange
from EventPeek.PseudoNormaliser import PseudoNormaliser

def get_normalised_dom_features(pmt_event_df: pd.DataFrame, Q_cut: float = -1) -> pd.DataFrame:
    selected_columns = ["dom_x", "dom_y", "dom_z", "Qtotal", "t1"]
    event_np = pmt_event_df[selected_columns].to_numpy()
    normaliser = PseudoNormaliser()
    
    normalised_np = normaliser(event_np, column_names=selected_columns)
    normalised_df = pd.DataFrame(normalised_np, columns=selected_columns)
    normalised_df[["dom_x", "dom_y", "dom_z"]] = pmt_event_df[["dom_x", "dom_y", "dom_z"]].to_numpy()
    normalised_df = normalised_df[normalised_df["Qtotal"] > Q_cut] if Q_cut is not None else normalised_df

    return normalised_df


def add_string_column_to_event_df(event_df: pd.DataFrame, ref_position_df: pd.DataFrame, tolerance=2.0):
    event_df = event_df.copy()
    event_df['string'] = np.nan

    x_diff = np.square(ref_position_df['dom_x'].values[:, None] - event_df['dom_x'].values)
    y_diff = np.square(ref_position_df['dom_y'].values[:, None] - event_df['dom_y'].values)
    
    distances = np.sqrt(x_diff + y_diff)
    min_indices = np.argmin(distances, axis=0)

    within_tolerance = np.min(distances, axis=0) <= tolerance
    event_df.loc[within_tolerance, 'string'] = ref_position_df.iloc[min_indices[within_tolerance]]['string'].values

    return event_df

def get_string_df(event_df: pd.DataFrame):
    string_df = event_df.groupby("string").agg({"dom_x": "mean", "dom_y": "mean", "Qtotal": "sum"})
    string_df.reset_index(inplace=True)
    return string_df

# Optimised function: Reduce dtype and ensure stability
def calculate_horizontal_boundary(pmt_event_df: pd.DataFrame):
    xy_points = pmt_event_df[['dom_x', 'dom_y']].drop_duplicates().values.astype(np.float32, copy=False)

    if xy_points.shape[0] < 3:
        return xy_points  # Return raw points if not enough for ConvexHull
    
    try:
        hull = ConvexHull(xy_points)
        return xy_points[hull.vertices]
    except Exception:
        print("ConvexHull failed: Points are nearly collinear.")
        return xy_points

def calculate_horizontal_PCA(boundary_points: np.ndarray):
    if boundary_points.shape[0] < 2:
        print("⚠️ Warning: Not enough points for PCA. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan  # ✅ Safe return to prevent crashing

    if np.all(boundary_points == boundary_points[0]):  # All points identical
        print("⚠️ Warning: All points are identical. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan

    centre = np.mean(boundary_points, axis=0)
    pca = PCA(n_components=2)

    try:
        pca.fit(boundary_points)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        major_axis_length, minor_axis_length = np.sqrt(eigenvalues)
        return major_axis_length, minor_axis_length, eigenvectors, centre
    except Exception as e:
        print(f"⚠️ PCA failed: {e}. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan  # ✅ Return safe values


def find_gmm_centres(string_df: pd.DataFrame, n_components=2):
    if string_df.shape[0] < n_components:
        return None, None  # Not enough points for clustering
    points = string_df[['dom_x', 'dom_y']].values
    weights = string_df['Qtotal'].to_numpy()
    weights = np.clip((weights / (weights.max() + 1e-7)) * 10, 1, 10).astype(int)
    weighted_points = np.repeat(points, weights, axis=0)
    # weighted_points = np.concatenate([points[i].reshape(1, -1).repeat(w, axis=0) for i, w in enumerate(weights)])
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(weighted_points)  # No `sample_weight` needed
    centres = gmm.means_
    cluster_dist = np.linalg.norm(centres[0] - centres[1])  # Distance between clusters
    std_dev = np.std(points, axis=0).mean()
    separation_score = cluster_dist / (std_dev + 1e-7)
    return centres, separation_score


def find_kmeans_centres(string_df: pd.DataFrame, n_clusters=2):
    if string_df.shape[0] < n_clusters:
        return None, None  # Not enough points for clustering
    points = string_df[['dom_x', 'dom_y']].values
    weights = string_df['Qtotal'].to_numpy()
    weights = np.clip((weights / (weights.max() + 1e-7)) * 10, 1, 10).astype(int)
    weighted_points = np.repeat(points, weights, axis=0)
    # weighted_points = np.concatenate([points[i].reshape(1, -1).repeat(w, axis=0) for i, w in enumerate(weights)])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(weighted_points)
    centres = kmeans.cluster_centers_
    cluster_dist = np.linalg.norm(centres[0] - centres[1])
    std_dev = np.std(points, axis=0).mean()
    separation_score = cluster_dist / (std_dev + 1e-7)
    return centres, separation_score


def compute_dispersion(string_df: pd.DataFrame, string_df_original: pd.DataFrame):
    if string_df.shape[0] < 3:
        return np.nan  # Not enough points
    eps = 1e-8  # Avoid division by zero
    points = string_df[['dom_x', 'dom_y']].values
    string_df_original = string_df_original.reindex(string_df["string"])  # Reindex based on `string_df`
    weights = string_df_original['Qtotal'].to_numpy()

    if np.all(np.isnan(weights)) or np.all(weights == 0):
        print("⚠️ Warning: All weights are zero or NaN. Returning NaN.")
        return np.nan
    centroid = np.average(points, axis=0, weights=weights)

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # 2D area
    except Exception as e:
        print(f"⚠️ ConvexHull failed: {e}. Returning NaN.")
        return np.nan
    total_weighted_distance = np.sum(weights * np.linalg.norm(points - centroid, axis=1))
    dispersion = total_weighted_distance / (hull_area + eps)

    return dispersion


def compute_outer_mass_fraction(string_df: pd.DataFrame, num_bins=10):
    if string_df.shape[0] < 3:
        return np.nan  # Not enough points
    eps = 1e-8
    points = string_df[['dom_x', 'dom_y']].to_numpy()
    weights = string_df['Qtotal'].to_numpy()
    
    if np.all(np.isnan(weights)) or np.all(weights == 0):
        print("⚠️ Warning: All weights are zero or NaN. Returning NaN.")
        return np.nan
    
    centroid = np.average(points, axis=0, weights=weights)
    distances = np.linalg.norm(points - centroid, axis=1)
    
    if np.all(distances == 0):  # All points are identical
        print("⚠️ Warning: All points are at the same position. Returning NaN.")
        return np.nan
    
    max_dist = np.nanmax(distances)
    bins = np.linspace(0, max_dist, num_bins + 1)

    mass_distribution, _ = np.histogram(distances, bins=bins, weights=weights)
    
    outer_mass = np.sum(mass_distribution[num_bins // 2:])  # Sum outer half
    total_mass = np.sum(mass_distribution)
    
    if total_mass == 0:
        print("⚠️ Warning: Total mass is zero. Returning NaN.")
        return np.nan
    
    outer_mass_fraction = outer_mass / (total_mass + eps)
    
    return outer_mass_fraction


# XY features
def compute_max_extent(boundary_points: np.ndarray) -> tuple:
    dist_matrix = squareform(pdist(boundary_points, metric="euclidean"))
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)  # Find indices of max distance
    point1, point2 = boundary_points[i], boundary_points[j]
    
    return point1, point2, dist_matrix[i, j]

# Z features
def calculate_vertical_stretch(pmt_event_df: pd.DataFrame):
    stretch_per_string = pmt_event_df.groupby("string")["dom_z"].agg(lambda x: x.max() - x.min())
    if stretch_per_string.empty:
        print("⚠️ Warning: No valid `dom_z` data for vertical stretch calculation. Returning NaNs.")
        return np.nan, np.nan, [np.nan, np.nan], np.nan
    stretch_mean = stretch_per_string.mean()
    stretch_max = stretch_per_string.max()
    top_one_sigma = stretch_per_string.quantile(0.843)
    bottom_one_sigma = stretch_per_string.quantile(0.157)
    half_interquartile_range = (top_one_sigma - bottom_one_sigma) / 2
    if stretch_per_string.empty or np.isnan(stretch_mean) or np.isnan(stretch_max):
        print("⚠️ Warning: No valid `dom_z` stretch data. Returning NaNs.")
        return np.nan, np.nan, [np.nan, np.nan], np.nan
    if stretch_per_string.isna().all():
        print("⚠️ Warning: All stretch values are NaN. Returning NaNs.")
        return np.nan, np.nan, [np.nan, np.nan], np.nan
    try:
        max_string = stretch_per_string.idxmax()
        max_z_positions = pmt_event_df.loc[pmt_event_df["string"] == max_string, "dom_z"].agg(["min", "max"]).values
    except Exception:
        print("⚠️ Warning: Could not determine max stretch string. Returning NaNs.")
        max_string, max_z_positions = np.nan, [np.nan, np.nan]

    return stretch_mean, stretch_max, max_z_positions, half_interquartile_range



## ** NEW FACES ASSEMBLE! ** ##
def rookies_assemble(pmt_event_df: pd.DataFrame, ref_position_df: pd.DataFrame, Q_cut: int):
    # Normalise the event DataFrame
    pseudo_normalised_df = get_normalised_dom_features(pmt_event_df, Q_cut)
    pseudo_normalised_df = add_string_column_to_event_df(pseudo_normalised_df, ref_position_df)
    pmt_event_df = add_string_column_to_event_df(pmt_event_df, ref_position_df)

    border_xy = calculate_horizontal_boundary(pmt_event_df)

    xy_end1, xy_end2, extent_max = compute_max_extent(border_xy)
    major_PCA, minor_PCA, eigenvectors, centre = calculate_horizontal_PCA(border_xy)
    eccentricity_PCA = np.sqrt(1 - (minor_PCA / major_PCA)**2)
    aspect_contrast_PCA = (major_PCA - minor_PCA) / (major_PCA + minor_PCA)

    stretch_mean, stretch_max, max_z_positions, stretch_hiqr = calculate_vertical_stretch(pseudo_normalised_df)

    string_df = get_string_df(pseudo_normalised_df)
    gmm_centres, gmm_score = find_gmm_centres(string_df)
    kmeans_centres, kmeans_score = find_kmeans_centres(string_df)
    dispersion = compute_dispersion(string_df, get_string_df(pmt_event_df))
    outer_mass_fraction = compute_outer_mass_fraction(string_df)

    # 🛠 **Reintroducing Missing Calculations**
    max_extent_stretch = max(extent_max, stretch_max)
    product = extent_max * stretch_max
    hypotenuse = np.sqrt(extent_max**2 + stretch_max**2)

    result = (
        extent_max, stretch_max, stretch_mean, stretch_hiqr, 
        max_extent_stretch, product, hypotenuse, 
        major_PCA, minor_PCA, eccentricity_PCA, aspect_contrast_PCA,
        gmm_score, kmeans_score, 
        dispersion, outer_mass_fraction
    )

    return result


def plot_distribution(data_dict, title, xlabel, ylabel, binwidth, isLog=False):
    """
    Generalised function to plot distributions for different neutrino flavours.
    
    Parameters:
        data_dict (dict): Keys are Flavour enums, values are numpy arrays of data.
        title (str): Title of the plot.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        binwidth (float): Bin width for the histogram.
        isLog (bool): Whether to use a logarithmic scale.
    
    Returns:
        fig, ax: The figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(13, 9))

    # 🚀 **Fix: Ensure sample_data is a NumPy array**
    sample_data = np.array(next(iter(data_dict.values()), []))  # Convert to array

    # 🚨 **Handle Empty or NaN-Only Data**
    if sample_data.size == 0 or np.all(np.isnan(sample_data.astype(float))):
        print(f"⚠️ Warning: All values are NaN or empty for {title}. Skipping plot.")
        return None, None

    try:
        # Compute histogram parameters (only if valid data exists)
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(
            sample_data, binwidth=binwidth, isLog=isLog, isDensity=False
        )
    except ValueError as e:
        print(f"⚠️ Histogram binning error in {title}: {e}. Skipping plot.")
        return None, None

    # Iterate over neutrino flavours
    for flavour, data in data_dict.items():
        clean_data = np.array(data, dtype=float)  # 🚀 **Ensure valid NumPy array**
        clean_data = clean_data[~np.isnan(clean_data)]  # Remove NaNs for histogram plotting
        
        # Choose correct colour index (2 for e, 0 for mu, 1 for tau)
        colour_index = {Flavour.E: 2, Flavour.MU: 0, Flavour.TAU: 1}.get(flavour, 3)
        hatch_index = {Flavour.E: '/', Flavour.MU: '\\', Flavour.TAU: '-'}

        ax.hist(clean_data, bins=bins, color=getColour(colour_index), 
                histtype='step', label=fr'$\nu_{{{flavour.alias}}}$', linewidth=3, hatch=hatch_index.get(flavour, None))

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Summary statistics
    stats_summary = {f"N events": sum(len(d) for d in data_dict.values()),
                     "binwidth": f"{binwidth:.1f}",
                     "Nbins": Nbins}

    for flavour, data in data_dict.items():
        clean_data = np.array(data, dtype=float)  # Ensure valid NumPy array
        nan_count = np.isnan(clean_data).sum()  # Count NaN values
        valid_data = clean_data[~np.isnan(clean_data)]  # Remove NaNs for calculations

        flavour_stats = {
            f"$\\nu_{{{flavour.alias}}}$": '',
            "N_events": len(data),
            "N_NaN": nan_count,  # Add NaN count
            "max": f"{np.nanmax(valid_data):.2f}" if valid_data.size > 0 else "N/A",
            "min": f"{np.nanmin(valid_data):.2f}" if valid_data.size > 0 else "N/A",
            "mean": f"{np.nanmean(valid_data):.2f}" if valid_data.size > 0 else "N/A",
            "median": f"{np.nanmedian(valid_data):.2f}" if valid_data.size > 0 else "N/A",
            "std": f"{np.nanstd(valid_data):.2f}" if valid_data.size > 0 else "N/A",
        }

        # 🚨 **Fixing Colour Mapping for Text**
        colour_offset = {Flavour.E: 0.85, Flavour.MU: 0.65, Flavour.TAU: 0.45}
        text_position = colour_offset.get(flavour, 0.85)

        add_text_to_ax(0.80, text_position, nice_string_output(flavour_stats), 
                       ax, fontsize=10, color='black')

    add_text_to_ax(0.70, 0.97, nice_string_output(stats_summary), ax, fontsize=10, color='black')
    
    return fig, ax


def process_event(event_df_ref_Qcut):
    """Wrapper function for parallel event processing"""
    event_df, ref_position_df, Q_cut = event_df_ref_Qcut
    return rookies_assemble(event_df, ref_position_df, Q_cut)


def collect_extent_stretch_part_from_different_flavours(root_before_subdir: str, 
                                                         er: EnergyRange, 
                                                         part: int, 
                                                         Q_cut: int,
                                                         num_shards_dict: dict):
    """
    Collects and processes extent, stretch, and other geometric features for different neutrino flavours.
    """

    # Define feature dictionary for all flavours
    feature_data = {
        "extent_max": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "stretch_max": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "stretch_mean": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "stretch_hiqr": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "max_extent_stretch": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "product": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "hypotenuse": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "major_PCA": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "minor_PCA": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "eccentricity_PCA": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "aspect_contrast_PCA": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "gmm_score": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "kmeans_score": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "dispersion": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "outer_mass_fraction": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    }

    for flavour in [Flavour.E, Flavour.MU, Flavour.TAU]:
        subdir = os.path.join(root_before_subdir, EnergyRange.get_subdir(er, flavour))
        num_shards = num_shards_dict.get(flavour, 10)  # Default 10 shards
        ref_position_file = "/groups/icecube/cyan/factory/DOMification/dom_ref_pos/unique_string_dom_completed.csv"
        ref_position_df = pd.read_csv(ref_position_file)

        # num_workers = max(1, cpu_count() - 2)  # Use all but 2 cores
        num_workers = min(40, cpu_count() - 8)  # Use at most 40 cores
        for shard in range(1, num_shards + 1):  
            pmt_file = os.path.join(subdir, str(part), f"PMTfied_{shard}.parquet")
            if not os.path.exists(pmt_file):
                print(f"⚠️ Warning: Shard {shard} for {flavour.alias} not found. Skipping.")
                continue
            
            print(f"📂 Processing {pmt_file} for {flavour.alias}...")
            columns_needed = ["event_no", "dom_x", "dom_y", "dom_z", "Qtotal", "t1"]
            pmt_df = pq.read_table(pmt_file, columns=columns_needed).to_pandas()
            pmt_df = pmt_df

            pmt_df_grouped = list(pmt_df.groupby("event_no"))

            with Pool(num_workers) as pool:
                results = list(tqdm(pool.imap(process_event, 
                                            [(df, ref_position_df, Q_cut) for _, df in pmt_df_grouped]),
                                    total=len(pmt_df_grouped),
                                    desc=f"Processing {flavour.alias} events (Shard {shard})",
                                    miniters=max(1000, len(pmt_df_grouped) // 10),
                                    mininterval=30))

            for values in results:
                if values is None:
                    continue
                for i, key in enumerate(feature_data.keys()):
                    feature_data[key][flavour].append(values[i])

    # Convert lists to numpy arrays
    for key in feature_data.keys():
        for flavour in [Flavour.E, Flavour.MU, Flavour.TAU]:
            feature_data[key][flavour] = np.array(feature_data[key][flavour])

    # Generate plots
    figs = []
    plot_params = [
        ("extent_max", "Max Extent(XY) [m]", "Counts", 50),
        ("stretch_max", "Max Stretch(Z) [m]", "Counts", 25),
        ("stretch_mean", "Mean Stretch(Z) [m]", "Counts", 10),
        ("stretch_hiqr", "Half Interquartile Range(Z) [m]", "Counts", 10),
        ("max_extent_stretch", "Max Extent & Stretch(XY or Z) [m]", "Counts", 50),
        ("product", "(Max Extent) x (Max Stretch)(XYZ) [m²]", "Counts", 25000),
        ("hypotenuse", "Hypotenuse(XYZ) [m]", "Counts", 50),
        ("major_PCA", "Major PCA(XY) [m]", "Counts", 50),
        ("minor_PCA", "Minor PCA(XY) [m]", "Counts", 50),
        ("eccentricity_PCA", "Eccentricity PCA(XY)", "Counts", 0.1),
        ("aspect_contrast_PCA", "Aspect Contrast PCA(XY)", "Counts", 0.1),
        ("gmm_score", "GMM Separation Score(XY)", "Counts", 0.1),
        ("kmeans_score", "KMeans Separation Score(XY)", "Counts", 0.1),
        ("dispersion", "Dispersion(XY)", "Counts", 0.1),
        ("outer_mass_fraction", "Outer Mass Fraction(XY)", "Counts", 0.1)
    ]

    for key, xlabel, ylabel, binwidth in plot_params:
        fig, _ = plot_distribution(feature_data[key], 
                                   f"{key.replace('_', ' ').title()} Distribution for {er.latex} ($Q_{{\\text{{adjusted}}}}$ > {Q_cut})",
                                   xlabel, ylabel, binwidth)
        if fig:
            figs.append(fig)

    return figs


def save_figs_to_pdf(figs, output_pdf_dir: str, output_pdf_file: str):
    """Save multiple figures into a single PDF file."""
    os.makedirs(output_pdf_dir, exist_ok=True)
    output_pdf_path = os.path.join(output_pdf_dir, output_pdf_file)

    with PdfPages(output_pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)  # Free memory

    print(f"Saved PDF: {output_pdf_path}")


def run():
    root_dir_noCR_CC_IN = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/CC_CRclean_Contained/"
    output_pdf_dir = "/lustre/hpc/icecube/cyan/factory/DOMification/EventPeek/"

    er = EnergyRange.ER_10_TEV_1_PEV
    Q_cuts = [-10, -1, 0]
    part = 1

    # Define different shard counts per flavour
    # num_shards_dict = {
    #     Flavour.E: 4,   # num shards for nu e(4)
    #     Flavour.MU: 9,  # num shards for nu mu(9)
    #     Flavour.TAU: 4   # num shards for nu tau(4)
    # }
    num_shards_dict = {
        Flavour.E: 1,   # num shards for nu e(4)
        Flavour.MU: 1,  # num shards for nu mu(9)
        Flavour.TAU: 1   # num shards for nu tau(4)
    }

    # Add tqdm for tracking different Q_cut values
    for Q_cut in tqdm(Q_cuts, desc="Processing different Q_cuts", unit="cut"):
        print(f"Processing {er.string} energy range with Q_cut={Q_cut}")
        start_time = time.time()
        
        figs = collect_extent_stretch_part_from_different_flavours(root_dir_noCR_CC_IN, er, part, Q_cut, num_shards_dict)

        if not figs:  # Check if figs is empty
            print(f"⚠️ Warning: No figures generated for {er.string} with Q >{Q_cut}. Skipping PDF saving.")
            continue  # Skip saving to avoid empty PDFs

        output_pdf_file = f"ExtentStretch_{er.string}_Qcut{Q_cut}_part{part}.pdf"
        save_figs_to_pdf(figs, output_pdf_dir, output_pdf_file)

        print(f"Finished processing {er.string} with Q_cut={Q_cut} in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
