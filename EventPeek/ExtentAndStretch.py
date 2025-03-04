import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import matplotlib.colors as mcolors

import sys
import scipy.stats as stats
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.spatial.distance import pdist, squareform


from sklearn.decomposition import PCA
import os

import pyarrow as pa
import pyarrow.parquet as pq

from enum import Enum

from matplotlib.backends.backend_pdf import PdfPages 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


# Optimised function: Reduce dtype and ensure stability
def calculate_horizontal_boundary(pmt_event_df: pd.DataFrame):
    xy_points = pmt_event_df[['dom_x', 'dom_y']].drop_duplicates().to_numpy(dtype=np.float32)

    if xy_points.shape[0] < 3:
        return xy_points  # Return raw points if not enough for ConvexHull
    
    try:
        hull = ConvexHull(xy_points)
        return xy_points[hull.vertices]
    except Exception:
        print("ConvexHull failed: Points are nearly collinear.")
        return xy_points


# Optimised function: Reduce dtype for speedup
def compute_max_extent(boundary_points: np.ndarray) -> tuple:
    dist_matrix = squareform(pdist(boundary_points, metric="euclidean"))
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)  # Find indices of max distance
    point1, point2 = boundary_points[i], boundary_points[j]
    
    return point1, point2, dist_matrix[i, j]



# Optimised function: Use float32 for calculations
def calculate_vertical_stretch(pmt_event_df: pd.DataFrame):
    stretch_per_string = pmt_event_df.groupby("string")["dom_z"].agg(lambda x: x.max() - x.min())

    mean_z_stretch = stretch_per_string.mean()
    max_z_stretch = stretch_per_string.max()

    mean_string = (stretch_per_string - mean_z_stretch).abs().idxmin()
    max_string = stretch_per_string.idxmax()

    mean_z_positions = pmt_event_df.loc[pmt_event_df["string"] == mean_string, "dom_z"].agg(["min", "max"]).values
    max_z_positions = pmt_event_df.loc[pmt_event_df["string"] == max_string, "dom_z"].agg(["min", "max"]).values

    return mean_z_stretch, max_z_stretch, mean_z_positions, max_z_positions



# Optimised function: Reduce dtype and ensure efficiency
def get_extent_stretch_shard(pmt_event_df: pd.DataFrame, Q_cut: int):
    ref_position_file = "/groups/icecube/cyan/factory/DOMification/dom_ref_pos/unique_string_dom_completed.csv"
    ref_position_df = pd.read_csv(ref_position_file)

    pseudo_normalised_df = get_normalised_dom_features(pmt_event_df, Q_cut)
    if pseudo_normalised_df.empty:
        return 0.0, 0.0  

    pseudo_normalised_df = add_string_column_to_event_df(pseudo_normalised_df, ref_position_df)
    border_xy = calculate_horizontal_boundary(pseudo_normalised_df)

    xy_end1, xy_end2, max_extent_2d = compute_max_extent(border_xy)
    mean_z_stretch, max_z_stretch, mean_z_positions, max_z_positions = calculate_vertical_stretch(pseudo_normalised_df)

    return max_extent_2d, max_z_stretch


def plot_dist_extent(extents_e: np.ndarray, 
                     extents_mu: np.ndarray,
                     extents_tau: np.ndarray,
                     title: str):
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(extents_mu, binwidth = 50, isLog=False, isDensity = False)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.hist(extents_e, bins=bins, color=getColour(2), histtype='step', label=r'$\nu_e$', linewidth=3, hatch='/')
    ax.hist(extents_mu, bins=bins, color=getColour(0), histtype='step', label=r'$\nu_\mu$', linewidth=3, hatch='\\')
    ax.hist(extents_tau, bins=bins, color=getColour(1), histtype='step', label=r'$\nu_\tau$', linewidth=2, hatch='-')

    ax.legend()
    ax.set_xlabel(r'Extent [m]')
    ax.set_ylabel('Counts')
    ax.set_title(fr"{title}")
    
    d = {'N events': len(extents_e) + len(extents_mu) + len(extents_tau),
         'binwidth': f"{binwidth:.1f}",
         'Nbins': Nbins}
    
    d_e = {r'$\nu_e$': '',
           'N_events': len(extents_e),
           'max': f"{np.max(extents_e):.2f}",
           'min': f"{np.min(extents_e):.2f}",
            'mean': f"{np.mean(extents_e):.2f}",
            'median': f"{np.median(extents_e):.2f}",
            'std': f"{np.std(extents_e):.2f}"}
    d_mu = {r'$\nu_\mu$': '',
            'N_events': len(extents_mu),
            'max': f"{np.max(extents_mu):.2f}",
            'min': f"{np.min(extents_mu):.2f}",
            'mean': f"{np.mean(extents_mu):.2f}",
            'median': f"{np.median(extents_mu):.2f}",
            'std': f"{np.std(extents_mu):.2f}"}
    d_tau = {r'$\nu_\tau$': '',
            'N_events': len(extents_tau),
            'max': f"{np.max(extents_tau):.2f}",
            'min': f"{np.min(extents_tau):.2f}",
            'mean': f"{np.mean(extents_tau):.2f}",
            'median': f"{np.median(extents_tau):.2f}",
            'std': f"{np.std(extents_tau):.2f}"}

    add_text_to_ax(0.70, 0.97, nice_string_output(d), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.85, nice_string_output(d_e), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.70, nice_string_output(d_mu), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.55, nice_string_output(d_tau), ax, fontsize=10, color='black')
    return fig, ax


def plot_dist_stretch(stretches_e: np.ndarray, 
                      stretches_mu: np.ndarray,
                      stretches_tau: np.ndarray,
                      title: str):
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(stretches_mu, binwidth = 50, isLog=False, isDensity = False)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.hist(stretches_e, bins=bins, color=getColour(2), histtype='step', label=r'$\nu_e$', linewidth=3, hatch='/')
    ax.hist(stretches_mu, bins=bins, color=getColour(0), histtype='step', label=r'$\nu_\mu$', linewidth=3, hatch='\\')
    ax.hist(stretches_tau, bins=bins, color=getColour(1), histtype='step', label=r'$\nu_\tau$', linewidth=2, hatch='-')

    ax.legend()
    ax.set_xlabel(r'Stretch [m]')
    ax.set_ylabel('Counts')
    ax.set_title(fr"{title}")
    
    d = {'N events': len(stretches_e) + len(stretches_mu) + len(stretches_tau),
         'binwidth': f"{binwidth:.1f}",
         'Nbins': Nbins}
    
    d_e = {r'$\nu_e$': '',
            'N_events': len(stretches_e),
            'max': f"{np.max(stretches_e):.2f}",
            'min': f"{np.min(stretches_e):.2f}",
            'mean': f"{np.mean(stretches_e):.2f}",
            'median': f"{np.median(stretches_e):.2f}",
            'std': f"{np.std(stretches_e):.2f}"}
    d_mu = {r'$\nu_\mu$': '',
            'N_events': len(stretches_mu),
            'max': f"{np.max(stretches_mu):.2f}",
            'min': f"{np.min(stretches_mu):.2f}",
            'mean': f"{np.mean(stretches_mu):.2f}",
            'median': f"{np.median(stretches_mu):.2f}",
            'std': f"{np.std(stretches_mu):.2f}"}
    d_tau = {r'$\nu_\tau$': '',
            'N_events': len(stretches_tau),
            'max': f"{np.max(stretches_tau):.2f}",
            'min': f"{np.min(stretches_tau):.2f}",
            'mean': f"{np.mean(stretches_tau):.2f}",
            'median': f"{np.median(stretches_tau):.2f}",
            'std': f"{np.std(stretches_tau):.2f}"}

    add_text_to_ax(0.70, 0.97, nice_string_output(d), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.85, nice_string_output(d_e), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.70, nice_string_output(d_mu), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.55, nice_string_output(d_tau), ax, fontsize=10, color='black')
    return fig, ax


def plot_max_extent_stretch(max_e: np.ndarray,
                            max_mu: np.ndarray,
                            max_tau: np.ndarray,
                            title: str):
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(max_mu, binwidth = 50, isLog=False, isDensity = False)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.hist(max_e, bins=bins, color=getColour(2), histtype='step', label=r'$\nu_e$', linewidth=3, hatch='/')
    ax.hist(max_mu, bins=bins, color=getColour(0), histtype='step', label=r'$\nu_\mu$', linewidth=3, hatch='\\')
    ax.hist(max_tau, bins=bins, color=getColour(1), histtype='step', label=r'$\nu_\tau$', linewidth=2, hatch='-')
    
    ax.legend()
    ax.set_xlabel(r'Extent or Stretch [m]')
    ax.set_ylabel('Counts')
    ax.set_title(fr"{title}")
    
    d = {'N events': len(max_e) + len(max_mu) + len(max_tau),
            'binwidth': f"{binwidth:.1f}",
            'Nbins': Nbins}
    
    d_e = {r'$\nu_e$': '',
              'N_events': len(max_e),
              'max': f"{np.max(max_e):.2f}",
              'min': f"{np.min(max_e):.2f}",
              'mean': f"{np.mean(max_e):.2f}",
            'median': f"{np.median(max_e):.2f}",
            'std': f"{np.std(max_e):.2f}"}
    d_mu = {r'$\nu_\mu$': '',
            'N_events': len(max_mu),
            'max': f"{np.max(max_mu):.2f}",
            'min': f"{np.min(max_mu):.2f}",
            'mean': f"{np.mean(max_mu):.2f}",
            'median': f"{np.median(max_mu):.2f}",
            'std': f"{np.std(max_mu):.2f}"}
    d_tau = {r'$\nu_\tau$': '',
            'N_events': len(max_tau),
            'max': f"{np.max(max_tau):.2f}",
            'min': f"{np.min(max_tau):.2f}",
            'mean': f"{np.mean(max_tau):.2f}",
            'median': f"{np.median(max_tau):.2f}",
            'std': f"{np.std(max_tau):.2f}"}
    
    add_text_to_ax(0.70, 0.97, nice_string_output(d), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.85, nice_string_output(d_e), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.70, nice_string_output(d_mu), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.55, nice_string_output(d_tau), ax, fontsize=10, color='black')
    return fig, ax


def plot_dist_product(products_e: np.ndarray,
                        products_mu: np.ndarray,
                        products_tau: np.ndarray,
                        title: str):
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(products_mu, binwidth = 10000, isLog=False, isDensity = False)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.hist(products_e, bins=bins, color=getColour(2), histtype='step', label=r'$\nu_e$', linewidth=3, hatch='/')
    ax.hist(products_mu, bins=bins, color=getColour(0), histtype='step', label=r'$\nu_\mu$', linewidth=3, hatch='\\')
    ax.hist(products_tau, bins=bins, color=getColour(1), histtype='step', label=r'$\nu_\tau$', linewidth=2, hatch='-')
    ax.legend()
    ax.set_xlabel(r'Product [m$^2$]')
    ax.set_ylabel('Counts')
    ax.set_title(fr"{title}")
    
    d = {'N events': len(products_e) + len(products_mu) + len(products_tau),
            'binwidth': f"{binwidth:.1f}",
            'Nbins': Nbins}
    
    d_e = {r'$\nu_e$': '',
           'N_events': len(products_e),
            'max': f"{np.max(products_e):.2f}",
            'min': f"{np.min(products_e):.2f}",
            'mean': f"{np.mean(products_e):.2f}",
            'median': f"{np.median(products_e):.2f}",
            'std': f"{np.std(products_e):.2f}"}
    
    d_mu = {r'$\nu_\mu$': '',
            'N_events': len(products_mu),
            'max': f"{np.max(products_mu):.2f}",
            'min': f"{np.min(products_mu):.2f}",
            'mean': f"{np.mean(products_mu):.2f}",
            'median': f"{np.median(products_mu):.2f}",
            'std': f"{np.std(products_mu):.2f}"}
    
    d_tau = {r'$\nu_\tau$': '',
            'N_events': len(products_tau),
            'max': f"{np.max(products_tau):.2f}",
            'min': f"{np.min(products_tau):.2f}",
            'mean': f"{np.mean(products_tau):.2f}",
            'median': f"{np.median(products_tau):.2f}",
            'std': f"{np.std(products_tau):.2f}"}
    
    add_text_to_ax(0.70, 0.97, nice_string_output(d), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.85, nice_string_output(d_e), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.70, nice_string_output(d_mu), ax, fontsize=10, color='black')
    add_text_to_ax(0.70, 0.55, nice_string_output(d_tau), ax, fontsize=10, color='black')
    return fig, ax
    

def collect_extent_stretch_shards_from_different_flavours(root_before_subdir: str, er: EnergyRange, part: int, shard: int, Q_cut: int):
    extent_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    stretch_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    max_extent_strecth_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    product_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    
    for flavour in [Flavour.E, Flavour.MU, Flavour.TAU]:
        subdir = os.path.join(root_before_subdir, EnergyRange.get_subdir(er, flavour))
        pmt_file = os.path.join(subdir, str(part), f"PMTfied_{int(shard)}.parquet")
        if not os.path.exists(pmt_file):
            continue
        pmt_df = pq.read_table(pmt_file).to_pandas()
        pmt_df_grouped = list(pmt_df.groupby("event_no"))
        for i, (event_no, event_df) in tqdm(enumerate(pmt_df_grouped), total=len(pmt_df_grouped), desc=f"Processing {flavour.alias} events", unit="event", mininterval=5):
            extent, stretch = get_extent_stretch_shard(event_df, Q_cut=Q_cut)
            max_extent_strecth = max(extent, stretch)
            product = extent * stretch
            extent_data[flavour].append(extent)
            stretch_data[flavour].append(stretch)
            max_extent_strecth_data[flavour].append(max_extent_strecth)
            product_data[flavour].append(product)

    extents_e, extents_mu, extents_tau = map(np.array, [extent_data[Flavour.E], extent_data[Flavour.MU], extent_data[Flavour.TAU]])
    stretches_e, stretches_mu, stretches_tau = map(np.array, [stretch_data[Flavour.E], stretch_data[Flavour.MU], stretch_data[Flavour.TAU]])
    
    return extents_e, extents_mu, extents_tau, stretches_e, stretches_mu, stretches_tau, max_extent_strecth_data, product_data


def collect_extent_stretch_part_from_different_flavours(root_before_subdir: str, 
                                                         er: EnergyRange, 
                                                         part: int, 
                                                         Q_cut: int,
                                                         num_shards: int = 10):  # Default: Process up to 10 shards
    extent_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    stretch_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    max_extent_stretch_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}
    product_data = {Flavour.E: [], Flavour.MU: [], Flavour.TAU: []}  
    figs = []

    for flavour in [Flavour.E, Flavour.MU, Flavour.TAU]:
        subdir = os.path.join(root_before_subdir, EnergyRange.get_subdir(er, flavour))

        # Iterate over multiple shard files
        for shard in range(1, num_shards + 1):  
            pmt_file = os.path.join(subdir, str(part), f"PMTfied_{shard}.parquet")

            if not os.path.exists(pmt_file):
                print(f"⚠️ Warning: Shard {shard} for {flavour.alias} not found. Skipping.")
                continue

            print(f"📂 Processing {pmt_file} for {flavour.alias}...")

            pmt_df = pq.read_table(pmt_file).to_pandas()
            pmt_df_grouped = list(pmt_df.groupby("event_no"))
            # pmt_df_grouped = pmt_df_grouped[:100]  # DEBUG: Limit to 1000 events for faster processing

            for i, (event_no, event_df) in tqdm(enumerate(pmt_df_grouped), total=len(pmt_df_grouped), 
                                                 desc=f"Processing {flavour.alias} events (Shard {shard})", 
                                                 unit="event", mininterval=5):
                extent, stretch = get_extent_stretch_shard(event_df, Q_cut=Q_cut)
                max_extent_stretch = max(extent, stretch)
                product = extent * stretch

                extent_data[flavour].append(extent)
                stretch_data[flavour].append(stretch)
                max_extent_stretch_data[flavour].append(max_extent_stretch)
                product_data[flavour].append(product)

    # Convert lists to numpy arrays
    extents_e, extents_mu, extents_tau = map(np.array, 
                                        [extent_data[Flavour.E], 
                                         extent_data[Flavour.MU], 
                                         extent_data[Flavour.TAU]])
    stretches_e, stretches_mu, stretches_tau = map(np.array, 
                                        [stretch_data[Flavour.E], 
                                         stretch_data[Flavour.MU], 
                                         stretch_data[Flavour.TAU]])
    max_e_s_e, max_e_s_mu, max_e_s_tau = map(np.array, 
                                         [max_extent_stretch_data[Flavour.E], 
                                          max_extent_stretch_data[Flavour.MU], 
                                          max_extent_stretch_data[Flavour.TAU]])

    product_e, product_mu, product_tau = map(np.array, 
                                         [product_data[Flavour.E], 
                                          product_data[Flavour.MU], 
                                          product_data[Flavour.TAU]])

    fig_extent, _ = plot_dist_extent(extents_e, extents_mu, extents_tau, 
                                  f"Max Extent Distribution for {er.latex} ($Q_{{\\text{{adjusted}}}}$ > {Q_cut})")
    fig_stretch, _ = plot_dist_stretch(stretches_e, stretches_mu, stretches_tau, 
                                    f"Max Stretch Distribution for {er.latex} ($Q_{{\\text{{adjusted}}}}$ > {Q_cut})")
    fig_max_extent_stretch, _ = plot_max_extent_stretch(max_e_s_e, max_e_s_mu, max_e_s_tau,
                                                     f"Max (Extent, Stretch) Distribution for {er.latex} ($Q_{{\\text{{adjusted}}}}$ > {Q_cut})")
    fig_product, _ = plot_dist_product(product_e, product_mu, product_tau,
                                      f"(Max Extent)x(Max Stretch) Distribution for {er.latex} ($Q_{{\\text{{adjusted}}}}$ > {Q_cut})")

    figs.extend([fig_extent, fig_stretch, fig_max_extent_stretch, fig_product])
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
    part, shard = 1, 1

    # Add tqdm for tracking different Q_cut values
    for Q_cut in tqdm(Q_cuts, desc="Processing different Q_cuts", unit="cut"):
        print(f"Processing {er.string} energy range with Q_cut={Q_cut}")
        start_time = time.time()
        
        # figs = collect_extent_stretch_shards_from_different_flavours(root_dir_noCR_CC_IN, er, part, shard, Q_cut)
        figs = collect_extent_stretch_part_from_different_flavours (root_dir_noCR_CC_IN, er, part, Q_cut, num_shards = 10)

        if not figs:  # Check if figs is empty
            print(f"⚠️ Warning: No figures generated for {er.string} with Q >{Q_cut}. Skipping PDF saving.")
            continue  # Skip saving to avoid empty PDFs

        output_pdf_file = f"ExtentStretch_{er.string}_Qcut{Q_cut}_part{part}.pdf"
        save_figs_to_pdf(figs, output_pdf_dir, output_pdf_file)

        print(f"Finished processing {er.string} with Q_cut={Q_cut} in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
