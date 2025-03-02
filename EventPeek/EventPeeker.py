import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
import scipy.stats as stats
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import os

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
import time

from matplotlib.backends.backend_pdf import PdfPages 

sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
from ExternalFunctions import nice_string_output, add_text_to_ax
setMplParam()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DOMification')))
# Now import from Enum
from Enum.Flavour import Flavour
from Enum.EnergyRange import EnergyRange
from EventPeek.PseudoNormaliser import PseudoNormaliser

class EventPeeker():
    def __init__(self):
        pass
    def __call__(self, 
                 root_dir: str, 
                 er: EnergyRange, 
                 flavour: Flavour, 
                 output_pdf_dir:str, 
                 output_pdf_file:str, 
                 N_doms_cut:int = 1250, # E: 1250, MU: ? , TAU: 1225
                 Q_adj_cut:float = -1):
        self.save_plots_to_pdf(root_dir, er, flavour, output_pdf_dir, output_pdf_file, N_doms_cut, Q_adj_cut)
        
    def save_plots_to_pdf(self,
                      root_dir: str, 
                      er: EnergyRange, 
                      flavour: Flavour, 
                      output_pdf_dir:str, 
                      output_pdf_file:str, 
                      N_doms_cut:int = 1250,
                      Q_adj_cut:float = -1):
        os.makedirs(output_pdf_dir, exist_ok=True)

        output_pdf_path = os.path.join(output_pdf_dir, output_pdf_file)
        subdir = EnergyRange.get_subdir(er, flavour)
        with PdfPages(output_pdf_path) as pdf:
            # Get all events from this subdir
            subdir_path = os.path.join(root_dir, subdir)
            files = [f for f in os.listdir(subdir_path) if f.endswith(".parquet")]
            
            for file in tqdm(files, desc=f"Processing {flavour.alias} events"):
                part = int(file.split("_")[-1].split(".")[0])
                # Process and generate plots for each part
                interesting_events = self.get_interesting_events(truth_file = os.path.join(subdir_path, file), N_doms_cut=N_doms_cut)
                energy_range = EnergyRange.get_energy_range(subdir).latex
                title_for_energy_spectrum = f"${flavour.latex}$,  {energy_range}   (part {part})"
                
                # Get energy distribution plot
                fig, ax = self.get_energy_distribution_from_truth_file(os.path.join(subdir_path, file), title=title_for_energy_spectrum)
                pdf.savefig(fig)  # Save the energy distribution plot
                plt.close(fig)  # Close the figure to free memory
                
                for event_no, shard_no, energy in interesting_events.values:
                    pmt_shard_df = pq.read_table(os.path.join(subdir_path, str(part), f"PMTfied_{int(shard_no)}.parquet")).to_pandas()
                    pmt_event_df = pmt_shard_df[pmt_shard_df["event_no"] == event_no]
                    
                    # Get DOM heatmap plot
                    fig, ax = self.plot_DOM_heatmap_for_this_event(pmt_event_df, shard_no, energy, Q_cut=Q_adj_cut)
                    pdf.savefig(fig)  # Save the DOM heatmap plot
                    plt.close(fig)  # Close the figure to free memory
                    
    def get_interesting_events(self, truth_file:str, N_doms_cut = 1250):
        truth_df = pq.read_table(truth_file).to_pandas()
        N_doms = truth_df["N_doms"]
        
        interesting_event_no = truth_df[["event_no", "shard_no", "energy"]][N_doms > N_doms_cut]
        return interesting_event_no
    
    def get_energy_distribution_from_truth_file(self, truth_file: str, title: str = None):
        def linear(x, coeff, offset):
            return coeff*x + offset
        truth_df = pq.read_table(truth_file).to_pandas()
        energies = truth_df["energy"]

        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(energies, Nbins=50, isLog=True)

        valid_mask = counts > 0  # Avoid log(0) errors
        log_x = np.log10(bin_centers[valid_mask])
        log_y = np.log10(counts[valid_mask])
        
        # **Poisson errors (standard deviation) for log(y)**
        sigma = 1 / np.sqrt(counts[valid_mask])

        popt, pcov = opt.curve_fit(linear, log_x, log_y, sigma=sigma, absolute_sigma=True)# returns optimal parameters and covariance matrix
        slope, intercept = popt

        log_y_fit = linear(log_x, *popt)
        
        # **Compute χ², DOF, and p-value**
        residuals = (log_y - log_y_fit) / sigma
        chi2_value = np.sum(residuals**2)
        dof = len(log_y) - len(popt)  # Degrees of freedom
        p_value = stats.chi2.sf(chi2_value, dof)  # Compute p-value

        # **Plot Histogram and Fit**
        fig, ax = plt.subplots(figsize=(18, 13))
        ax.hist(energies, bins=bins, histtype='step', lw=2, color=getColour(0), label='Energy distribution')

        # **Plot Fit Line Only for Valid Data**
        y_fit = 10**linear(log_x, *popt)
        ax.plot(10**log_x, y_fit, label=f'Fit: $y = {slope:.2f}x + {intercept:.2f}$', color=getColour(1), linestyle='-')

        ax.legend()
        ax.set_xlabel(r'$\log_{10}(\mathrm{energy})$')
        ax.set_ylabel('Counts')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(fr"{title}")

        # **Summary statistics**
        d = {'N_events': f"{len(energies)}",
            'mean': f"{np.mean(energies):.3e}",
            'std': f"{np.std(energies):.3e}",
            'max': f"{np.max(energies):.3e}",
            'min': f"{np.min(energies):.3e}",
            'slope': f"{slope:.3f}",
            'intercept': f"{intercept:.3f}",
            'χ²': f"{chi2_value:.3f}",
            'DOF': f"{dof}",
            'p-value': f"{p_value:.3f}"}

        add_text_to_ax(0.65, 0.85, nice_string_output(d), ax, fontsize=12)
        return fig, ax

    def plot_DOM_heatmap_for_this_event(self, pmt_event_df: pd.DataFrame, shard_no:int, energy:int, Q_cut:int = -1, elev=40, azim=105):
        position_file = "/groups/icecube/cyan/factory/DOMification/dom_ref_pos/unique_string_dom_completed.csv"
        position_df = pd.read_csv(position_file)

        # Normalise the event DataFrame
        pseudo_normalised_df = self.get_normalised_dom_features(pmt_event_df, Q_cut)
        pseudo_normalised_df = self.add_string_column_to_event_df(pseudo_normalised_df, position_df)
        
        # Create figure and 3D axis
        fig, ax = plt.subplots(figsize=(18, 13), subplot_kw={'projection': '3d'})
        # **Plot reference DOM positions as small grey dots (background)**
        ax.scatter(
            position_df["dom_x"], position_df["dom_y"], position_df["dom_z"],
            color=getColour(1), marker=".", s=10, label="Reference DOMs", zorder=1
        )

        # ✅ Apply Qtotal cutoff **before** computing statistics
        major_axis_length, minor_axis_length, eigenvectors, centre, border_xy = self.calculate_horizontal_PCA(pseudo_normalised_df)
        mean_z_stretch, max_z_stretch, mean_z_positions, max_z_positions = self.calculate_vertical_stretch(pseudo_normalised_df)
        xy_end1, xy_end2, max_extent_2d = self.compute_max_extent(border_xy)
        # Extract relevant columns
        dom_x = pseudo_normalised_df["dom_x"]
        dom_y = pseudo_normalised_df["dom_y"]
        dom_z = pseudo_normalised_df["dom_z"]
        Qtotal = pseudo_normalised_df["Qtotal"]
        t1 = pseudo_normalised_df["t1"]

        # Marker size scaling
        min_marker_size, max_marker_size = 10, 200
        Qmin, Qmax = Qtotal.min(), Qtotal.max()
        marker_sizes = min_marker_size + (Qtotal - Qmin) / (Qmax - Qmin) * (max_marker_size - min_marker_size)

        # Normalise arrival time for colormap
        t_norm = mcolors.Normalize(vmin=t1.min(), vmax=t1.max())

        # **Plot event scatter points (foreground)**
        sc = ax.scatter(dom_x, dom_y, dom_z, c=t1, cmap="cool", norm=t_norm,
                        s=marker_sizes, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=2)

        # Colorbar for arrival time
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.75)
        cbar.set_label("Adjusted Arrival Time (t1)")
        
        # **String labels**
        TEXT_Z = 600
        for string in pseudo_normalised_df['string'].unique():
            string_df = pseudo_normalised_df[pseudo_normalised_df['string'] == string]
            ax.text(string_df['dom_x'].iloc[0], string_df['dom_y'].iloc[0], TEXT_Z,
                    f'{int(string)}', fontsize=10, ha='center', color='black')
        
        # **PCA axes**
        major_axis_endpoints = np.array([
            centre[:2] - major_axis_length * eigenvectors[0],
            centre[:2] + major_axis_length * eigenvectors[0]
        ])
        minor_axis_endpoints = np.array([
            centre[:2] - minor_axis_length * eigenvectors[1],
            centre[:2] + minor_axis_length * eigenvectors[1]
        ])
        major_axis_3d = np.column_stack((major_axis_endpoints, np.full(2, TEXT_Z)))
        minor_axis_3d = np.column_stack((minor_axis_endpoints, np.full(2, TEXT_Z)))
        ax.plot(major_axis_3d[:, 0], major_axis_3d[:, 1], major_axis_3d[:, 2], color=getColour(2), linewidth=3, label="PCA XY major")
        ax.plot(minor_axis_3d[:, 0], minor_axis_3d[:, 1], minor_axis_3d[:, 2], color=getColour(3), linewidth=3, label="PCA XY minor")

        # **Max extent line**
        max_extent_3d = np.array([
            [xy_end1[0], xy_end1[1], TEXT_Z],
            [xy_end2[0], xy_end2[1], TEXT_Z]
        ])
        ax.plot(max_extent_3d[:, 0], max_extent_3d[:, 1], max_extent_3d[:, 2], 
                color=getColour(0), linewidth=3, linestyle="dashed", label="Max XY extent")

        # **Z Stretch lines**
        z_mean_at_this_xy = (575, 475)
        z_max_at_this_xy = (625, 525)
        ax.plot([z_mean_at_this_xy[0], z_mean_at_this_xy[0]], [z_mean_at_this_xy[1], z_mean_at_this_xy[1]], mean_z_positions, 
                color=getColour(5), linewidth=3, label="Mean Z Stretch")

        ax.plot([z_max_at_this_xy[0], z_max_at_this_xy[0]], [z_max_at_this_xy[1], z_max_at_this_xy[1]], max_z_positions, 
                color=getColour(8), linewidth=3,  label="Max Z Stretch")
        
        # Create top boundary polygon (XY Boundary)
        verts_top = [np.column_stack((border_xy[:, 0], border_xy[:, 1], np.full(border_xy.shape[0], TEXT_Z)))]
        ax.add_collection3d(Poly3DCollection(verts_top, facecolors=getColour(6), alpha=0.2, label='XY Boundary (Top)'))
        
        for x, y in border_xy:
            ax.plot([x, x], [y, y], TEXT_Z, color=getColour(6), alpha=0.5)
        ax.legend()
            
        
        # **Create a marker size legend**
        size_legend_values = np.linspace(Qmin, Qmax, num=4)
        size_legend_markers = min_marker_size + (size_legend_values - Qmin) / (Qmax - Qmin) * (max_marker_size - min_marker_size)

        # Position the legend outside the plot
        legend_ax = fig.add_axes([0.85, 0.15, 0.075, 0.1])  # [left, bottom, width, height]
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        legend_ax.set_title(r"$Q_{\text{adjusted}} = \log_{10}(Q_{\text{total}}) - 2$" + "\n" +
                            r"$T_{\text{adjusted}} = (T - 10^4)\times (3\times 10^4)^{-1}$", fontsize=14)

        for size, q in zip(size_legend_markers, size_legend_values):
            legend_ax.scatter([], [], s=size, edgecolors='black', facecolors='none', label=f"{q:.2f}")

        legend_ax.legend(loc="center", frameon=False, fontsize=10)

        # **Statistics Dictionary**
        event_no = pmt_event_df["event_no"].iloc[0]
        subdir_no, part_no, original_event_no, energy_range, flavour = self.get_flavour_ER_subdir_part(event_no)
        
        d_class = {
            "event_no": f"{original_event_no}",
            "part, shard": f"{part_no}, {int(shard_no)}",
            "PCA XY major": f"{major_axis_length:.3f}",
            "PCA XY minor": f"{minor_axis_length:.3f}",
            "max XY extent": f"{max_extent_2d:.3f}",
            "mean Z stretch": f"{mean_z_stretch:.3f}",
            "max Z stretch": f"{max_z_stretch:.3f}",
        }
        
        original_Qtot = pmt_event_df["Qtotal"]
        original_t1 = pmt_event_df["t1"]
        N_dom_original = len(pmt_event_df)
        d_original_event = {
            "original values": "",
            "N_doms": f"{N_dom_original}",
            "Q max": f"{original_Qtot.max():.3f}",
            "Q mean": f"{original_Qtot.mean():.3f}",
            "Q median": f"{original_Qtot.median():.3f}",
            "t1 max": f"{original_t1.max():.3f}",
            "t1 min": f"{original_t1.min():.3f}",
            "t1 mean": f"{original_t1.mean():.3f}",
            "t1 median": f"{original_t1.median():.3f}",
        }
        N_dom_cut = len(pseudo_normalised_df)
        d_pseudo_normalised = {
            "adjusted values": "",
            f"N_DOM(Q>{Q_cut})": f"{N_dom_cut}({N_dom_cut/N_dom_original:.2f})",
            f"Q max": f"{Qtotal.max():.3f}",
            f"Q mean": f"{Qtotal.mean():.3f}",
            f"Q median": f"{Qtotal.median():.3f}",
            f"t1 max": f"{t1.max():.3f}",
            f"t1 min": f"{t1.min():.3f}",
            f"t1 mean": f"{t1.mean():.3f}",
            f"t1 median": f"{t1.median():.3f}",
        }

        # ✅ Add text using fig.text() to ensure correct positioning
        fig.text(0.15, 0.80, nice_string_output(d_class), fontsize=12, family='monospace', verticalalignment='top', color='black')
        fig.text(0.15, 0.20, nice_string_output(d_original_event), fontsize=12, family='monospace', verticalalignment='top', color='black')
        fig.text(0.62, 0.25, nice_string_output(d_pseudo_normalised), fontsize=12, family='monospace', verticalalignment='top', color='black')

        # **Set plot labels**
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        # ax.set_title(r"DOM Positions with $Q_{\text{adjusted}}$ and $t1_{\text{adjusted}}$")
        ax.set_title(f"$Q_{{\\text{{adjusted}}}}$ & $t1_{{\\text{{adjusted}}}}$ of ${flavour.latex}$      ({energy:.3e}GeV)")


        # Adjust viewing angle
        ax.view_init(elev=elev, azim=azim)

        # plt.legend()  # Ensure labels appear
        plt.show()
        
        return fig, ax


    def get_normalised_dom_features(self, pmt_event_df: pd.DataFrame, Q_cut: float = -1) -> pd.DataFrame:
        selected_columns = ["dom_x", "dom_y", "dom_z", "Qtotal", "t1"]
        event_np = pmt_event_df[selected_columns].to_numpy()
        normaliser = PseudoNormaliser()
        
        normalised_np = normaliser(event_np, column_names=selected_columns)
        normalised_df = pd.DataFrame(normalised_np, columns=selected_columns)
        normalised_df[["dom_x", "dom_y", "dom_z"]] = pmt_event_df[["dom_x", "dom_y", "dom_z"]].to_numpy()
        normalised_df = normalised_df[normalised_df["Qtotal"] > Q_cut] if Q_cut is not None else normalised_df

        return normalised_df
    
    def calculate_horizontal_PCA(self, strings_df: pd.DataFrame):
        xy_points = strings_df[['dom_x', 'dom_y']].to_numpy()

        # Extract the outermost boundary points using Convex Hull
        hull = ConvexHull(xy_points)
        boundary_points = xy_points[hull.vertices]  

        centre = np.mean(boundary_points, axis=0)

        # Perform PCA to find the principal axes
        pca = PCA(n_components=2)
        pca.fit(boundary_points)

        # Extract eigenvalues and eigenvectors
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        # Compute major and minor axis lengths (sqrt of eigenvalues)
        major_axis_length, minor_axis_length = np.sqrt(eigenvalues)

        return major_axis_length, minor_axis_length, eigenvectors, centre, boundary_points

    def compute_max_extent(self, boundary_points):
        dist_matrix = squareform(pdist(boundary_points))  # Compute all pairwise distances
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)  # Find indices of max distance
        point1, point2 = boundary_points[i], boundary_points[j]
        
        return point1, point2, dist_matrix[i, j]
    
    def calculate_vertical_stretch(self, strings_df: pd.DataFrame):
        stretch_per_string = strings_df.groupby("string")["dom_z"].agg(lambda x: x.max() - x.min())

        mean_z_stretch = stretch_per_string.mean()
        max_z_stretch = stretch_per_string.max()

        mean_string = (stretch_per_string - mean_z_stretch).abs().idxmin()  # String closest to mean stretch
        max_string = stretch_per_string.idxmax()  # String with max stretch

        mean_z_positions = strings_df.loc[strings_df["string"] == mean_string, "dom_z"].agg(["min", "max"]).values
        max_z_positions = strings_df.loc[strings_df["string"] == max_string, "dom_z"].agg(["min", "max"]).values

        return mean_z_stretch, max_z_stretch, mean_z_positions, max_z_positions


    
    def add_string_column_to_event_df(self, event_df: pd.DataFrame, position_df: pd.DataFrame, tolerance=1.0):
        event_df = event_df.copy()  # Work on a copy to avoid modifying the original DataFrame

        # Create a new 'string' column and initialize it with NaN
        event_df['string'] = np.nan

        # Loop through each DOM in the event_df and find the closest match in position_df
        for idx, row in event_df.iterrows():
            # Calculate distances in x, y (we ignore z here, as it's not relevant for string determination)
            distances = np.sqrt((position_df['dom_x'] - row['dom_x'])**2 + (position_df['dom_y'] - row['dom_y'])**2)
            
            # Find the minimum distance, and check if it is within the tolerance
            min_distance_idx = distances.idxmin()
            if distances[min_distance_idx] <= tolerance:
                # Assign the string number from the reference position
                event_df.at[idx, 'string'] = position_df.at[min_distance_idx, 'string']
        
        return event_df

    
    def get_flavour_ER_subdir_part(self, event_no: int):
        event_no_str = str(event_no)

        subdir_id = event_no_str[1:3]  # 2-digit subdirectory
        part_no = int(event_no_str[3:7])  # 4-digit part number
        original_event_no = int(event_no_str[7:])  # Remaining digits = event index
        subdir_no = f"220{subdir_id}"
        energy_range = EnergyRange.get_energy_range(subdir_no)
        flavour = EnergyRange.get_flavour(subdir_no)

        return subdir_no, part_no, original_event_no, energy_range, flavour

    
if __name__ == "__main__":
    root_dir_noCR_CC_IN = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied_filtered/Snowstorm/CC_CRclean_Contained/"
    # N_dom_cut_e_TeV = 1225
    # N_dom_cut_mu_TeV = 1250
    # N_dom_cut_tau_TeV = 1100
    N_dom_cut_e_TeV = 3000
    N_dom_cut_mu_TeV = 3000
    N_dom_cut_tau_TeV = 3000
    pdf_dir = "/lustre/hpc/icecube/cyan/factory/DOMification/EventPeek/"
    
    # Energy range and flavour instances
    er = EnergyRange.ER_1_PEV_100_PEV
    # flavours = [Flavour.E, Flavour.MU, Flavour.TAU]
    # cuts = [N_dom_cut_e_TeV, N_dom_cut_mu_TeV, N_dom_cut_tau_TeV]
    
    select_flavour_cut_tuple = [
                                (Flavour.E, N_dom_cut_e_TeV), 
                                (Flavour.MU, N_dom_cut_mu_TeV), 
                                (Flavour.TAU, N_dom_cut_tau_TeV)
                                ]
    Q_adj_cut = -1
    for flavour, N_dom_cut in select_flavour_cut_tuple:
        start_time = time.time()
        
        # Use the string method to get the simplified energy range name
        output_pdf_file = f"EventPeek_{flavour.alias}_{er.string}_over_{N_dom_cut}DOMs.pdf"
        
        print(f"--------Processing {flavour.alias} events for {er.string} energy range...--------")
        EventPeeker()(root_dir_noCR_CC_IN, er, flavour, pdf_dir, output_pdf_file, N_dom_cut, Q_adj_cut)
        
        print(f"--------Finished processing {flavour.alias} events for {er.string} energy range--------")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

