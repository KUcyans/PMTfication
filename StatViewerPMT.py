import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
import os

import pyarrow.parquet as pq

from datetime import datetime
from tqdm import tqdm

from matplotlib.backends.backend_pdf import PdfPages 
import argparse

sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
from ExternalFunctions import nice_string_output, add_text_to_ax

class PMTfiedStatViewer:
    def __init__(self, pmt_file, output_pdf = None):
        setMplParam()
        self.pmt_file = pmt_file
        self.output_pdf = output_pdf
        self.df_pmt = self._convertParquetToDF(pmt_file)
        self.df_pmt_processed = self._lazy_normalisation(self.df_pmt)
        self._saveFigures_twice()

    def _saveFigures_twice(self, isShow: bool = False):
        bin_width_raw, bin_width_processsed = self._build_binwidth_array()
        self._saveFigures(self.df_pmt, self.output_pdf + "_raw", isShow, bin_width_raw)
        logging.info(f"Figures saved to {self.output_pdf}_raw.pdf")
        self._saveFigures(self.df_pmt_processed, self.output_pdf + "_processed", isShow, bin_width_processsed)
        logging.info(f"Figures saved to {self.output_pdf}_processed.pdf")
    
    def _saveFigures(self, df: pd.DataFrame, output_pdf: str, isShow: bool = False, bin_width: np.ndarray = None):
        with PdfPages(f"{output_pdf}.pdf") as pdf:
            plot_functions = [
                lambda: self._plot_dom_position(df, "x", bin_width=bin_width['xy']),
                lambda: self._plot_dom_position(df, "y", bin_width=bin_width['xy']),
                lambda: self._plot_dom_position(df, "z", bin_width=bin_width['z']),
                lambda: self._plot_q(df, bin_width=bin_width['q']),
                lambda: self._plot_Q(df, bin_width=bin_width['Q']),
                lambda: self._plot_t(df, bin_width=bin_width['t']),
                lambda: self.plot_T(df, bin_width=bin_width['T']),
                lambda: self._plot_sigma_T(df, bin_width=bin_width['T']),
            ]
            for plot_function in tqdm(plot_functions, desc="Plotting figures"):
                fig, ax = plot_function()
                pdf.savefig(fig)
                if isShow:
                    plt.show()
                plt.close()
    
    def _build_process_params(self) -> dict:
        self.position_scaler = 500
        self.t_scaler = 1e-4
        self.t_shifter = 10_000
        self.Q_shifter = 2
        
    def _build_binwidth_array(self) -> tuple:
        bin_width_raw = {
            "xy": 50, # position x, y
            "z": 50, # position z
            "q": 5, # q
            "Q": 100, # Q
            "t": 100, # t
            "T": 100, # T, sigma_T
        }
        bin_width_processed = {
            "xy": 0.1, # position x, y
            "z": 0.1, # position z
            "q": 0.1, # q
            "Q": 0.1, # Q
            "t": 0.1, # t
            "T": 0.1, # T, sigma_T
        }
        return bin_width_raw, bin_width_processed
            
    def _convertParquetToDF(self, file:str) -> pd.DataFrame:
        table = pq.read_table(file)
        df = table.to_pandas()
        return df
    
    def _lazy_normalisation(self, df_initial: pd.DataFrame) -> pd.DataFrame:
        self._build_process_params()
        df = df_initial.copy()
        AVOID_ZERO_DIVISION = 1e-10
        df[['dom_x', 'dom_y', 'dom_z']] /= self.position_scaler
        df[['dom_x_rel', 'dom_y_rel', 'dom_z_rel']] /= self.position_scaler
        
        for col in ['t1', 't2', 't3']:
            df.loc[df[col] != -1, col] = (df[col] - self.t_shifter) * self.t_scaler
        
        for col in ['q1', 'q2', 'q3']:
            df.loc[df[col] != -1, col] = np.log10(df[col]+AVOID_ZERO_DIVISION)
        
        for col in ['Q25', 'Q75', 'Qtotal']:
            df.loc[df[col] != -1, col] = np.log10(df[col]+AVOID_ZERO_DIVISION) - self.Q_shifter
        
        for col in ['T10', 'T50', 'sigmaT']:
            df.loc[df[col] != -1, col] *= self.t_scaler
        return df

        
    def _plot_histogram(self, *data, bins, xlabel, ylabel, title, log_scale_y=False, log_scale_x=False, **labels):
        logging.info(f"Plotting histogram for {title}")
        fig, ax = plt.subplots(figsize=(11, 7))
        
        for i, dataset in enumerate(data, start=1):
            label = labels.get(f"label{i}", f"Data {i}")
            if i % 5 == 1:
                hatch = ''
            elif i % 5 == 2:
                hatch = '//'
            elif i % 5 == 3:
                hatch = ''
            elif i % 5 == 4:
                hatch = '\\'
            else:
                hatch = ''
            ax.hist(dataset, bins=bins, histtype='step', linewidth=2, label=label, hatch=hatch)
        
        if log_scale_y:
            ax.set_yscale("log")
        
        if log_scale_x:
            ax.set_xscale("log")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if len(data) > 1:
            ax.legend()
        
        return fig, ax
    
    def _plot_dom_position(self, df: pd.DataFrame, axis: str, bin_width = 50):
        if axis not in ["x", "y", "z"]:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")
        
        dom_col = f"dom_{axis}"
        rel_dom_col = f"dom_{axis}_rel"
        
        dom_values = df[dom_col]
        rel_dom_values = df[rel_dom_col]
        
        bins = np.arange(dom_values.min(), dom_values.max() + bin_width, bin_width)
        
        fig, ax = self._plot_histogram(
            dom_values, rel_dom_values,
            bins=bins,
            xlabel=f"DOM {axis}",
            ylabel="Counts",
            title=f"DOM {axis} distribution",
            log_scale_y=False,
            label1=f"DOM {axis} position",
            label2=f"DOM {axis} relative position"
        )
        
        stats_dom = {
            f"DOM {axis}": "",
            "min": f"{dom_values.min():.2f}",
            "max": f"{dom_values.max():.2f}",
            "mean": f"{dom_values.mean():.2f}",
            "std": f"{dom_values.std():.2f}",
            "bin_width": f"{bin_width}",
        }
        stats_rel_dom = {
            f"DOM {axis} rel": "",
            "min": f"{rel_dom_values.min():.2f}",
            "max": f"{rel_dom_values.max():.2f}",
            "mean": f"{rel_dom_values.mean():.2f}",
            "std": f"{rel_dom_values.std():.2f}",
            "bin_width": f"{bin_width}",
        }
        
        add_text_to_ax(0.075, 0.95, nice_string_output(stats_dom), ax, fontsize=12)
        add_text_to_ax(0.70, 0.95, nice_string_output(stats_rel_dom), ax, fontsize=12)
        
        return fig, ax

    def _plot_Q(self, df: pd.DataFrame, bin_width=100):
        q_columns = ["Qtotal", "Q25", "Q75"]
        q_data = {col: df[col] for col in q_columns}
        
        bins = np.arange(q_data["Qtotal"].min(), q_data["Qtotal"].max() + bin_width, bin_width)
        
        data = [q_data[col] for col in q_columns]
        labels = {f"label{i+1}": col for i, col in enumerate(q_columns)}
        
        fig, ax = self._plot_histogram(
            *data,
            bins=bins,
            xlabel="Q",
            ylabel="Counts",
            title="Accumulated charge distribution",
            log_scale_y=True,
            **labels
        )
        for i, col in enumerate(q_columns):
            stats = {
                col: "",
                "min": f"{q_data[col].min():.2f}",
                "max": f"{q_data[col].max():.2f}",
                "mean": f"{q_data[col].mean():.2f}",
                "std": f"{q_data[col].std():.2f}",
                "bin_width": f"{bin_width}",
            }
            add_text_to_ax(0.1 + i * 0.3, 0.95, nice_string_output(stats), ax, fontsize=10)
        
        return fig, ax
    
    def _plot_q(self, df: pd.DataFrame, bin_width=0.1):
        q_columns = [col for col in df.columns if col.startswith('q') and col[1:].isdigit()]
        q_data = {col: df[col] for col in q_columns}
        
        first_q = q_columns[0]
        bins = np.arange(q_data[first_q].min(), q_data[first_q].max() + bin_width, bin_width)
        
        data_args = {}
        for i, col in enumerate(q_columns):
            data_args[f"data{i + 1}"] = q_data[col]
            data_args[f"label{i + 1}"] = col
        
        fig, ax = self._plot_histogram(
            *q_data.values(),
            bins=bins,
            xlabel="Q",
            ylabel="Counts",
            title=f"Distribution of first {len(q_columns)} charges",
            log_scale_y=True,
            **data_args
        )
        
        for i, col in enumerate(q_columns):
            stats = {
                "charge": col,
                "min": f"{q_data[col].min():.2f}",
                "max": f"{q_data[col].max():.2f}",
                "mean": f"{q_data[col].mean():.2f}",
                "std": f"{q_data[col].std():.2f}",
                "bin_width": f"{bin_width}",
            }
            x_pos = 0.01 + i * 0.20
            add_text_to_ax(x_pos, 0.95, nice_string_output(stats), ax, fontsize=8)
        
        return fig, ax
    
    def _plot_t(self, df: pd.DataFrame, bin_width = 100):
        t_columns = [col for col in df.columns if col.startswith('t') and col[1:].isdigit()]
        t_data = {col: df[col] for col in t_columns}
        
        first_t = t_columns[0]
        bins = np.arange(t_data[first_t].min(), t_data[first_t].max() + bin_width, bin_width)
        data_args = {}
        
        for i, col in enumerate(t_columns):
            data_args[f"data{i + 1}"] = t_data[col]
            data_args[f"label{i + 1}"] = col
            
        fig, ax = self._plot_histogram(
            *t_data.values(),
            bins=bins,
            xlabel="t",
            ylabel="Counts",
            title=f"Distribution of first {len(t_columns)} pulse times",
            log_scale_y=True,
            **data_args
        )
        
        for i, col in enumerate(t_columns):
            stats = {
                "time": col,
                "min": f"{t_data[col].min():.2f}",
                "max": f"{t_data[col].max():.2f}",
                "mean": f"{t_data[col].mean():.2f}",
                "std": f"{t_data[col].std():.2f}",
                "bin_width": f"{bin_width}",
            }
            x_pos = 0.01 + i * 0.20
            add_text_to_ax(x_pos, 0.95, nice_string_output(stats), ax, fontsize=8)
            
        return fig, ax
        
    def plot_T(self, df: pd.DataFrame, bin_width = 100):
        t_columns = ["T10", "T50"]
        t_data = {col: df[col] for col in t_columns}
        
        bins = np.arange(t_data["T10"].min(), t_data["T10"].max() + bin_width, bin_width)
        
        data = [t_data[col] for col in t_columns]
        labels = {f"label{i+1}": col for i, col in enumerate(t_columns)}
        
        fig, ax = self._plot_histogram(
            *data,
            bins=bins,
            xlabel="T",
            ylabel="Counts",
            title="T10 and T50 distribution",
            log_scale_y=True,
            **labels
        )
        
        for i, col in enumerate(t_columns):
            stats = {
                col: "",
                "min": f"{t_data[col].min():.1f}",
                "max": f"{t_data[col].max():.1f}",
                "mean": f"{t_data[col].mean():.1f}",
                "std": f"{t_data[col].std():.1f}",
                "bin_width": f"{bin_width}",
            }
            add_text_to_ax(0.25 + i * 0.4, 0.95, nice_string_output(stats), ax, fontsize=12)
        
        return fig, ax
    
    def _plot_sigma_T(self, df:pd.DataFrame, bin_width = 100):
        sigma_T = df["sigmaT"]
        
        bins = np.arange(sigma_T.min(), sigma_T.max() + bin_width, bin_width)
        
        fig, ax = self._plot_histogram(
            sigma_T,
            bins=bins,
            xlabel="sigma_T",
            ylabel="Counts",
            title="sigma_T distribution",
            log_scale_y=True,
            label1="sigma_T"
        )
        
        stats_sigma_T = {
            "sigma_T": "",
            "min": f"{sigma_T.min():.1f}",
            "max": f"{sigma_T.max():.1f}",
            "mean": f"{sigma_T.mean():.1f}",
            "std": f"{sigma_T.std():.1f}",
            "bin_width": f"{bin_width}",
        }
        add_text_to_ax(0.60, 0.95, nice_string_output(stats_sigma_T), ax, fontsize=12)
        
        return fig, ax
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMTfiedStatViewer")
    parser.add_argument("subdir", type=str, help="Subdirectory")
    parser.add_argument("part", type=str, help="Part")
    parser.add_argument("shard", type=str, help="Shard")
    args = parser.parse_args()
    
    snowstorm_pmt_dir = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/PMTfied/Snowstorm/"
    file_dir = f"{snowstorm_pmt_dir}/{args.subdir}/{args.part}/PMTfied_{args.shard}.parquet"
    output_file_pdf = f"StatView_pmt_{args.subdir}_{args.part}_{args.shard}"
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_pdf_with_timestamp = os.path.join("./StatView/" + f"[{timestamp}]{output_file_pdf}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Reading PMT file {os.path.basename(file_dir)}")
    logging.info(f"Saving PDF to {output_pdf_with_timestamp}.pdf")
    
    pmt_viewer = PMTfiedStatViewer(file_dir, output_pdf_with_timestamp)
    logging.info("Finished")