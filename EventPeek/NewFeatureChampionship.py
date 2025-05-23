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
        # print("ConvexHull failed: Points are nearly collinear.")
        return xy_points

def calculate_horizontal_PCA(boundary_points: np.ndarray):
    if boundary_points.shape[0] < 2:
        # print("⚠️ Warning: Not enough points for PCA. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan  # ✅ Safe return to prevent crashing
    if np.all(boundary_points == boundary_points[0]):  # All points identical
        # print("⚠️ Warning: All points are identical. Returning NaNs.")
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


def compute_area_ratio(boundary_points: np.ndarray):
    if boundary_points.shape[0] < 3:
        return np.nan
    border_xy_IceCube = np.array([
            (-256.1400146484375, -521.0800170898438),
            (-132.8000030517578, -501.45001220703125),
            (-9.13000011444092, -481.739990234375),
            (114.38999938964844, -461.989990234375),
            (237.77999877929688, -442.4200134277344),
            (361.0, -422.8299865722656),
            (405.8299865722656, -306.3800048828125),
            (443.6000061035156, -194.16000366210938),
            (500.42999267578125, -58.45000076293945),
            (544.0700073242188, 55.88999938964844),
            (576.3699951171875, 170.9199981689453),
            (505.2699890136719, 257.8800048828125),
            (429.760009765625, 351.0199890136719),
            (338.44000244140625, 463.7200012207031),
            (224.5800018310547, 432.3500061035156),
            (101.04000091552734, 412.7900085449219),
            (22.11000061035156, 509.5),
            (-101.05999755859375, 490.2200012207031),
            (-224.08999633789062, 470.8599853515625),
            (-347.8800048828125, 451.5199890136719),
            (-392.3800048828125, 334.239990234375),
            (-437.0400085449219, 217.8000030517578),
            (-481.6000061035156, 101.38999938964844),
            (-526.6300048828125, -15.60000038146973),
            (-570.9000244140625, -125.13999938964844),
            (-492.42999267578125, -230.16000366210938),
            (-413.4599914550781, -327.2699890136719),
            (-334.79998779296875, -424.5),])
    IceCube_area = ConvexHull(border_xy_IceCube).volume
    area = ConvexHull(boundary_points).volume
    return area / IceCube_area

def compute_outer_mass_fraction(string_df: pd.DataFrame, num_bins=10):
    if string_df.shape[0] < 3:
        return np.nan  # Not enough points
    eps = 1e-8
    points = string_df[['dom_x', 'dom_y']].to_numpy()
    weights = string_df['Qtotal'].to_numpy()
    if np.all(np.isnan(weights)) or np.all(weights == 0):
        return np.nan
    centroid = np.average(points, axis=0, weights=weights)
    distances = np.linalg.norm(points - centroid, axis=1)
    if np.all(distances == 0):  # All points are identical
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

def compute_normalised_weighted_distance(string_df: pd.DataFrame):
    if string_df.shape[0] < 3:
        return np.nan
    points = string_df[['dom_x', 'dom_y']].to_numpy()
    weights = string_df['Qtotal'].to_numpy()
    
    centroid = np.average(points, axis=0, weights=weights)
    distances = np.linalg.norm(points - centroid, axis=1)
    total_weighted_distance = np.sum(distances * weights)
    
    total_Q_total = np.sum(weights)
    return total_weighted_distance / total_Q_total

# XY features
def compute_max_extent(boundary_points: np.ndarray) -> tuple:
    dist_matrix = squareform(pdist(boundary_points, metric="euclidean"))
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)  # Find indices of max distance
    point1, point2 = boundary_points[i], boundary_points[j]
    return point1, point2, dist_matrix[i, j]

def get_cos_squared_extent_PCA_major(extent_end1: np.ndarray, 
                                    extent_end2: np.ndarray, 
                                    extent_max: float,
                                    eigen_vectors: np.ndarray,
                                    major_PCA: float) -> float:
    if major_PCA == 0 or extent_max == 0:
        return np.nan
    extent_vector = extent_end2 - extent_end1
    extent_unit_vector = extent_vector / np.linalg.norm(extent_vector)
    pca_major_vector = eigen_vectors[0]
    cos_theta_squared = np.clip(np.dot(extent_unit_vector, pca_major_vector) ** 2, 0, 1)
    return cos_theta_squared

# Z features
def calculate_vertical_stretch(pmt_event_df: pd.DataFrame):
    stretch_per_string = pmt_event_df.groupby("string")["dom_z"].agg(lambda x: x.max() - x.min())
    if stretch_per_string.empty:
        # print("⚠️ Warning: No valid `dom_z` data for vertical stretch calculation. Returning NaNs.")
        return np.nan, np.nan, [np.nan, np.nan], np.nan
    stretch_mean = stretch_per_string.mean()
    stretch_max = stretch_per_string.max()
    top_one_sigma = stretch_per_string.quantile(0.843)
    bottom_one_sigma = stretch_per_string.quantile(0.157)
    half_interquartile_range = (top_one_sigma - bottom_one_sigma) / 2
    if stretch_per_string.empty or np.isnan(stretch_mean) or np.isnan(stretch_max):
        # print("⚠️ Warning: No valid `dom_z` stretch data. Returning NaNs.")
        return np.nan, np.nan, [np.nan, np.nan], np.nan
    if stretch_per_string.isna().all():
        # print("⚠️ Warning: All stretch values are NaN. Returning NaNs.")
        return np.nan, np.nan, [np.nan, np.nan], np.nan
    try:
        max_string = stretch_per_string.idxmax()
        max_z_positions = pmt_event_df.loc[pmt_event_df["string"] == max_string, "dom_z"].agg(["min", "max"]).values
    except Exception:
        # print("⚠️ Warning: Could not determine max stretch string. Returning NaNs.")
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
    cos_sq = get_cos_squared_extent_PCA_major(xy_end1, xy_end2, extent_max, eigenvectors, major_PCA)
    eccentricity_PCA = np.sqrt(1 - (minor_PCA / major_PCA)**2)
    aspect_contrast_PCA = (major_PCA - minor_PCA) / (major_PCA + minor_PCA)

    stretch_mean, stretch_max, max_z_positions, stretch_hiqr = calculate_vertical_stretch(pseudo_normalised_df)

    string_df = get_string_df(pseudo_normalised_df)
    gmm_centres, gmm_score = find_gmm_centres(string_df)
    kmeans_centres, kmeans_score = find_kmeans_centres(string_df)
    area_ratio = compute_area_ratio(border_xy)
    outer_mass_fraction = compute_outer_mass_fraction(string_df)
    normalised_weighted_distance = compute_normalised_weighted_distance(string_df)

    # 🛠 **Reintroducing Missing Calculations**
    max_extent_stretch = max(extent_max, stretch_max)
    product = np.sqrt(extent_max * stretch_max)
    hypotenuse = np.sqrt(extent_max**2 + stretch_max**2)

    result = (
        extent_max, stretch_max, stretch_mean, stretch_hiqr, 
        max_extent_stretch, product, hypotenuse, 
        major_PCA, minor_PCA, eccentricity_PCA, aspect_contrast_PCA, cos_sq,
        gmm_score, kmeans_score, 
        area_ratio, outer_mass_fraction, normalised_weighted_distance
    )

    return result


def plot_distribution(data_dict, title, xlabel, ylabel, binwidth, isLog=False, isDicLeft=False):
    fig, ax = plt.subplots(figsize=(17, 11))

    # 🚀 **Fix: Ensure sample_data is a NumPy array**
    sample_data = np.array(next(iter(data_dict.values()), []))  # Convert to array

    # 🚨 **Handle Empty or NaN-Only Data**
    if sample_data.size == 0 or np.all(np.isnan(sample_data.astype(float))):
        # print(f"⚠️ Warning: All values are NaN or empty for {title}. Skipping plot.")
        return None, None

    try:
        # Compute histogram parameters (only if valid data exists)
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(
            sample_data, binwidth=binwidth, isLog=isLog, isDensity=False
        )
    except ValueError as e:
        # print(f"⚠️ Histogram binning error in {title}: {e}. Skipping plot.")
        return None, None

    # Iterate over neutrino flavours
    for flavour, data in data_dict.items():
        clean_data = np.array(data, dtype=float)  # 🚀 **Ensure valid NumPy array**
        clean_data = clean_data[~np.isnan(clean_data)]  # Remove NaNs for histogram plotting
        
        # Choose correct colour index (2 for e, 0 for mu, 1 for tau)
        colour_index = {Flavour.E: 2, Flavour.MU: 0, Flavour.TAU: 1}.get(flavour, 3)
        hatch_index = {Flavour.E: '/', Flavour.MU: '\\', Flavour.TAU: '-'}

        ax.hist(clean_data, bins=bins, color=getColour(colour_index), 
                histtype='step', label=fr"${{{flavour.latex}}}$", linewidth=2, hatch=hatch_index.get(flavour, None))

    ax.legend(fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=22)

    # Summary statistics
    stats_summary = {f"N events": sum(len(d) for d in data_dict.values()),
                     "binwidth": f"{binwidth:.1f}",
                     "Nbins": Nbins}
    text_x_pos = 0.02 if isDicLeft else 0.75
    for flavour, data in data_dict.items():
        clean_data = np.array(data, dtype=float)  # Ensure valid NumPy array
        nan_count = np.isnan(clean_data).sum()  # Count NaN values
        valid_data = clean_data[~np.isnan(clean_data)]  # Remove NaNs for calculations

        flavour_stats = {
            f"${{{flavour.latex}}}$": '',
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

        add_text_to_ax(text_x_pos, text_position, nice_string_output(flavour_stats), 
                       ax, fontsize=22, color='black')

    add_text_to_ax(0.70, 0.97, nice_string_output(stats_summary), ax, fontsize=22, color='black')
    
    return fig, ax


def process_event(event_df_ref_Qcut):
    """Wrapper function for parallel event processing"""
    event_df, ref_position_df, Q_cut = event_df_ref_Qcut
    return rookies_assemble(event_df, ref_position_df, Q_cut)

def collect_event_refs(root_before_subdir, er, part, num_shards_dict):
    event_refs = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    for flavour in event_refs:
        subdir = os.path.join(root_before_subdir, EnergyRange.get_subdir(er, flavour))
        num_shards = num_shards_dict.get(flavour, 10)
        for shard in range(1, num_shards + 1):
            pmt_file = os.path.join(subdir, str(part), f"PMTfied_{shard}.parquet")
            if not os.path.exists(pmt_file):
                continue
            table = pq.read_table(pmt_file, columns=["event_no"])
            event_nos = table.column("event_no").to_numpy()
            event_refs[flavour].extend([(shard, en) for en in event_nos])
    return event_refs

def collect_idols_part_from_different_flavours(root_before_subdir: str, 
                                                        er: EnergyRange, 
                                                        part: int, 
                                                        Q_cut: int,
                                                        num_shards_dict: dict):
    """
    Collects and processes extent, stretch, and other geometric features for different neutrino flavours.
    """
    event_refs = collect_event_refs(root_before_subdir, er, part, num_shards_dict)
    min_events = min(len(v) for v in event_refs.values())

    # Truncate each list to `min_events`
    balanced_event_refs = {fl: refs[:min_events] for fl, refs in event_refs.items()}

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
        "cos_sq_extent_PCA_major": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "gmm_score": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "kmeans_score": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "area_ratio": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "outer_mass_fraction": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []},
        "normalised_weighted_distance": {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
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
            
            # Get allowed event_nos for this shard
            allowed_event_nos = {
                en for (sh, en) in balanced_event_refs[flavour] if sh == shard
            }
            if not allowed_event_nos:
                continue  # Skip shard if no selected events

            # Filter groupby object
            pmt_df_grouped = [group for eid, group in pmt_df.groupby("event_no") if eid in allowed_event_nos]
            # pmt_df_grouped = list(pmt_df.groupby("event_no"))

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
        ("extent_max", "Max Extent(XY) [m]", "Counts", 20),
        ("stretch_max", "Max Stretch(Z) [m]", "Counts", 10),
        ("stretch_mean", "Mean Stretch(Z) [m]", "Counts", 10),
        ("stretch_hiqr", "Half Interquartile Range(Z) [m]", "Counts", 3),
        ("max_extent_stretch", "max(Extent,Stretch)(XY or Z) [m]", "Counts", 20),
        ("product", "sqrt((Max Extent) x (Max Stretch))(XYZ) [m]", "Counts", 20),
        ("hypotenuse", "Hypotenuse(XYZ) [m]", "Counts", 20),
        ("major_PCA", "Major PCA(XY) [m]", "Counts", 20),
        ("minor_PCA", "Minor PCA(XY) [m]", "Counts", 20),
        ("eccentricity_PCA", "Eccentricity PCA(XY)", "Counts", 0.05, False, True),
        ("aspect_contrast_PCA", "Aspect Contrast PCA(XY)", "Counts", 0.05),
        ("cos_sq_extent_PCA_major", "Cos²(Extent↔PCA Major Axis)(XY)", "Counts", 0.01, False, True),
        ("gmm_score", "GMM Separation Score(XY)", "Counts", 0.1, False, True),
        ("kmeans_score", "KMeans Separation Score(XY)", "Counts", 0.1, False, True),
        ("area_ratio", "Area Ratio(XY)", "Counts", 0.05),
        ("outer_mass_fraction", "Outer Mass Fraction(XY)", "Counts", 0.05, False, True),
        ("normalised_weighted_distance", "Normalised Weighted Distance(XY)", "Counts", 3)
    ]
    for params in plot_params:
        key, xlabel, ylabel, binwidth, *extra = params  # Unpack optional values safely
        isLog = extra[0] if len(extra) > 0 else False  # Default False if not provided
        isDicLeft = extra[1] if len(extra) > 1 else False  # Default False if not provided
        fig, _ = plot_distribution(feature_data[key], 
                                   f"{key.replace('_', ' ').title()} Distribution for {er.latex} ($Q_{{\\text{{adjusted}}}}$ > {Q_cut})",
                                   xlabel, ylabel, binwidth, isLog, isDicLeft)
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
    er = EnergyRange.ER_1_PEV_100_PEV
    # Q_cuts = [-10, -1, 0]
    Q_cuts = [0]
    part = 1

    # Define different shard counts per flavour
    # num_shards_dict = {
    #     Flavour.E: 2,   # num shards for nu e(4)
    #     Flavour.MU: 4,  # num shards for nu mu(9)
    #     Flavour.TAU: 2   # num shards for nu tau(4)
    # }
    num_shards_dict = {
        Flavour.E: 1,   # num shards for nu e(4)
        Flavour.MU: 2,  # num shards for nu mu(9)
        Flavour.TAU: 1   # num shards for nu tau(4)
    }

    # Add tqdm for tracking different Q_cut values
    for Q_cut in tqdm(Q_cuts, desc="Processing different Q_cuts", unit="cut"):
        print(f"Processing {er.string} energy range with Q_cut={Q_cut}")
        start_time = time.time()
        
        figs = collect_idols_part_from_different_flavours(root_dir_noCR_CC_IN, er, part, Q_cut, num_shards_dict)

        if not figs:  # Check if figs is empty
            print(f"⚠️ Warning: No figures generated for {er.string} with Q >{Q_cut}. Skipping PDF saving.")
            continue  # Skip saving to avoid empty PDFs

        output_pdf_file = f"FeatureContest_{er.string}_Q>{Q_cut}_part{part}.pdf"
        save_figs_to_pdf(figs, output_pdf_dir, output_pdf_file)

        print(f"Finished processing {er.string} with Q>{Q_cut} in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
