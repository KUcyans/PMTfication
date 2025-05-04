import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3 as sql
import sys
from collections import defaultdict
import os
import matplotlib
import time
from tqdm import tqdm

from matplotlib.backends.backend_pdf import PdfPages 
import argparse

sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
from ExternalFunctions import nice_string_output, add_text_to_ax

# Now import from Enum
sys.path.append('/groups/icecube/cyan/factory/DOMification')
from Enum.Flavour import Flavour
from Enum.EnergyRange import EnergyRange
from EventPeek.PseudoNormaliser import PseudoNormaliser

class PulseMapFeatureStatViewer:
    def __init__(self, 
                 source_root: str,
                    output_pdf: str,
                    N_events: int,
                    energy_range: EnergyRange,
                    flavour: Flavour,
                    event_indices: list[int],
                    N_pulses_cut: int,
                    ):
        setMplParam()
        self.source_root = source_root
        self.output_pdf = output_pdf
        self.N_events = N_events
        self.energy_range = energy_range
        self.flavour = flavour
        flavour_to_colour = {Flavour.MU: 0, Flavour.TAU: 1, Flavour.E: 2}
        self.colour_i = flavour_to_colour.get(flavour)
        
        self.event_indices = event_indices
        self.N_pulses_cut = N_pulses_cut
        
        self.this_df = self._get_df_from_first_part(source_root, energy_range, flavour, N_events)
        
    def __call__(self):
        with PdfPages(self.output_pdf) as pdf:
            # Plot and save: N active DOMs
            fig = self.plot_N_active_DOMs_per_event()
            pdf.savefig(fig)
            plt.close(fig)

            # Plot and save: Pulse distribution per DOM for multiple events
            fig = self.plot_pulse_count_distribution_multi_event()
            if fig:
                pdf.savefig(fig)
                plt.close(fig)

            # Per-event plots
            for event_id in self.event_indices:
                # Pulse count distribution for a single event
                fig = self.plot_pulse_count_distribution_single_event(event_id)
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)

                # Charge vs Time per DOM (can return multiple figures)
                fig_list = self.plot_charge_time_single_event_by_DOM(event_id, N_pulses_cut=self.N_pulses_cut)
                for f in fig_list:
                    pdf.savefig(f)
                    plt.close(f)

    def _get_feature_by_events(self, df: pd.DataFrame, feature: str) -> dict:
        """
        Return nested dict: {event_no: {(string, dom_number): [feature values]}}
        each element of [feature values] corresponds to a pulse
        """
        grouped = df.groupby(['event_no', 'string', 'dom_number'])[feature].apply(list)
        feature_dict = {}
        for (event_no, string, dom), values in grouped.items():
            feature_dict.setdefault(event_no, {})[(string, dom)] = values
        return feature_dict

    # def _convertDBtoDF(self, file: str, table: str, N_events: int = None) -> pd.DataFrame:
    #     con = sql.connect(file)
    #     print(f"Reading {N_events} events from {table} of {file}")
    #     event_no_query = f"SELECT DISTINCT event_no FROM {table} LIMIT {N_events}"
    #     event_nos = pd.read_sql_query(event_no_query, con)["event_no"].tolist()

    #     event_no_list = ', '.join(map(str, event_nos))
    #     query = f"SELECT * FROM {table} WHERE event_no IN ({event_no_list})"

    #     df = pd.read_sql_query(query, con)
    #     con.close()

    #     return df
    
    def _convertDBtoDF_iterative(self, file: str, table: str, N_events: int) -> pd.DataFrame:
        con = sql.connect(file)
        event_no_query = f"SELECT DISTINCT event_no FROM {table} LIMIT {N_events}"
        event_nos = pd.read_sql_query(event_no_query, con)["event_no"].tolist()

        dfs = []
        for batch_start in tqdm(range(0, len(event_nos), 100), desc="Reading DB in batches"):
            batch_event_nos = event_nos[batch_start:batch_start+100]
            placeholders = ','.join('?' for _ in batch_event_nos)
            query = f"SELECT * FROM {table} WHERE event_no IN ({placeholders})"
            df_batch = pd.read_sql_query(query, con, params=batch_event_nos)
            dfs.append(df_batch)
        con.close()
        return pd.concat(dfs, ignore_index=True)

    
    def _get_first_db_part_file(self, source_root:str, energy_range: EnergyRange, flavour: Flavour) -> str:
        subdir = EnergyRange.get_subdir(energy_range, flavour)
        db_file = os.path.join(source_root, subdir, "merged_part_1.db")
        return db_file
    
    def _get_df_from_first_part(self, source_root:str, energy_range: EnergyRange, flavour: Flavour, N_events: int) -> pd.DataFrame:
        db_file = self._get_first_db_part_file(source_root, energy_range, flavour)
        df = self._convertDBtoDF_iterative(db_file, 'SRTInIcePulses', N_events=N_events)
        return df
    
    ## plotter functions
    def plot_N_active_DOMs_per_event(self) -> matplotlib.figure.Figure:
        """
        Plot the distribution of the number of unique DOMs per event.
        """
        charge_dict = self._get_feature_by_events(self.this_df, 'charge')
        # {event_no: {(string, dom_number): [values]}}

        # Count unique DOMs per event
        N_doms = []
        for i, (event_no, dom_map) in enumerate(charge_dict.items()):
            if i >= self.N_events:
                break
            N_doms.append(len(dom_map))

        N_doms = np.array(N_doms)
        binwidth = 10
        
        # Histogram setup
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(N_doms, binwidth=binwidth)
        fig, ax = plt.subplots(figsize=(17, 11))
        ax.hist(N_doms, bins=bins, color=getColour(self.colour_i), histtype='step', linewidth=2)
        # vertical line for median, 128, 256
        ax.axvline(x=np.median(N_doms), color='black', linestyle='--', linewidth=2, label='median')
        ax.axvline(x=128, color=getColour(3), linestyle='-.', linewidth=2, label='128 DOMs')
        ax.axvline(x=256, color=getColour(5), linestyle=':', linewidth=2, label='256 DOMs')

        # Plot formatting
        ax.set_title(fr"Distribution of active DOMs per event "
            fr"for ${self.flavour.latex}$ in {self.energy_range.latex} ({self.N_events} events)", fontsize=22)

        ax.set_xlabel("Number of active DOMs per event", fontsize=20)
        ax.set_ylabel("Number of events", fontsize=20)

        d = {
            'N events': self.N_events,
            'max N DOMs per event': np.max(N_doms),
            'min N DOMs per event': np.min(N_doms),
            'median N DOMs per event': np.median(N_doms),
            'mean N DOMs per event': np.mean(N_doms),
            'std N DOMs per event': np.std(N_doms),
            'fraction (N DOMs<128)': np.sum(N_doms < 128) / len(N_doms),
            'fraction (N DOMs<256)': np.sum(N_doms < 256) / len(N_doms),
            'binwidth': binwidth,
            'Nbins': Nbins,
        }
        add_text_to_ax(0.60, 0.85, nice_string_output(d), ax, fontsize=18)
        ax.legend(fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_pulse_count_distribution_single_event(self, event_index: int) -> matplotlib.figure.Figure:
        """
        Plot the distribution of pulse counts across all DOMs in a single event.
        All DOMs with at least one pulse are included.
        """

        charge_dict = self._get_feature_by_events(self.this_df, 'charge')

        try:
            event_no = list(charge_dict.keys())[event_index]
            dom_charge_map = charge_dict[event_no]
        except IndexError:
            print(f"Invalid event_index: {event_index}")
            return

        # Count pulses for all DOMs (at least one pulse per DOM is guaranteed)
        counts = np.array([len(charges) for charges in dom_charge_map.values()])

        binwidth = 5
        Nbins, binwidth, bins, _, _ = getHistoParam(counts, binwidth=binwidth)

        # Plot
        fig, ax = plt.subplots(figsize=(17, 11))
        ax.hist(counts, bins=bins, color=getColour(self.colour_i), histtype='step', linewidth=2)

        ax.set_title(fr"Distribution of pulses across DOMs in event {event_no} "
                    fr"(${self.flavour.latex}$, {self.energy_range.latex})", fontsize=22)
        ax.set_xlabel("Number of pulses per DOM", fontsize=20)
        ax.set_ylabel("Number of DOMs", fontsize=20)

        # Annotate
        d = {
            'Event no': event_no,
            'Active DOMs': len(dom_charge_map),
            'max N pulses in DOM': np.max(counts),
            'min N pulses in DOM': np.min(counts),
            'median N pulses per DOM': f"{np.median(counts):.0f}",
            'mean N pulses per DOM': np.mean(counts),
            'std N pulses per DOM': np.std(counts),
            'binwidth': binwidth,
            'Nbins': Nbins,
        }

        add_text_to_ax(0.60, 0.85, nice_string_output(d), ax, fontsize=18)

        plt.tight_layout()
        return fig

    def plot_pulse_count_distribution_multi_event(self) -> matplotlib.figure.Figure:
        """
        Plot distribution of number of pulses per DOM across multiple events.
        All DOMs with at least one pulse are included.
        """
        charge_dict = self._get_feature_by_events(self.this_df, 'charge')
        
        # Collect pulse counts
        pulse_counts = []
        for i, dom_charge_map in enumerate(charge_dict.values()):
            if i >= self.N_events:
                break
            for charges in dom_charge_map.values():
                pulse_counts.append(len(charges))

        if not pulse_counts:
            print(f"No active DOMs found in first {self.N_events} events.")
            return

        counts = np.array(pulse_counts)
        binwidth = 5
        Nbins, binwidth, bins, _, _ = getHistoParam(counts, binwidth=binwidth)

        # Plot
        fig, ax = plt.subplots(figsize=(17, 11))
        ax.hist(counts, bins=bins, color=getColour(self.colour_i), histtype='step', linewidth=2)

        ax.set_title(fr"Distribution of pulses across DOMs in {self.N_events} events "
                    fr"(${self.flavour.latex}$, {self.energy_range.latex})", fontsize=22)
        ax.set_xlabel("Number of pulses per DOM", fontsize=20)
        ax.set_ylabel("Number of DOMs", fontsize=20)

        # Annotate
        d = {
            'Total events': self.N_events,
            'Total active DOMs': len(counts),
            'Active DOMs per event': f"{len(counts) / self.N_events:.2f}",
            'max N pulses in DOM': np.max(counts),
            'min N pulses in DOM': np.min(counts),
            'median N pulses per DOM': f"{np.median(counts):.0f}",
            'mean N pulses per DOM': np.mean(counts),
            'std N pulses per DOM': np.std(counts),
            'binwidth': binwidth,
            'Nbins': Nbins,
        }

        add_text_to_ax(0.60, 0.85, nice_string_output(d), ax, fontsize=18)

        plt.tight_layout()
        return fig
    
    def plot_charge_time_single_event_by_DOM(self, event_index: int, N_pulses_cut: int) -> list[matplotlib.figure.Figure]:
        """
        Return list of figures showing charge vs time scatter plots for DOMs in a selected event.
        Each figure shows one DOM's hits if the number of pulses exceeds the threshold.
        """
        charge_dict = self._get_feature_by_events(self.this_df, 'charge')
        time_dict = self._get_feature_by_events(self.this_df, 'dom_time')

        try:
            event_no = list(charge_dict.keys())[event_index]
            charge_map = charge_dict[event_no]
            time_map = time_dict[event_no]
        except IndexError:
            print(f"Invalid event_index: {event_index}")
            return []
        
        N_DOMs = len(charge_map)
        
        fig_list = []
        for (string, dom) in charge_map:
            charge_list = charge_map[(string, dom)]
            time_list = time_map.get((string, dom), [])
            if len(charge_list) != len(time_list) or len(charge_list) <= N_pulses_cut:
                continue

            fig, ax = plt.subplots(figsize=(17, 11))
            colour = getColour(self.colour_i)
            ax.vlines(time_list, 0, charge_list, color=colour, linewidth=2, zorder=1)
            ax.scatter(time_list, charge_list, s=30, color=colour, edgecolor='black', linewidth=1, zorder=2)

            ax.set_title(fr"Charge vs Arrival Time for (string {string:.0f}, DOM {dom:.0f}) "
             fr"in event {event_no}({N_DOMs} active DOMs) of "
             fr"${self.flavour.latex}$ in {self.energy_range.latex}", fontsize=22)
            ax.set_xlabel("Arrival time (ns)", fontsize=20)
            ax.set_ylabel("Charge", fontsize=20)

            d = {
                # f'1 of {N_DOMs} active DOMs': "",
                'N pulses': len(charge_list),
                'max charge': max(charge_list),
                'min charge': min(charge_list),
                'median charge': np.median(charge_list),
                'mean charge': np.mean(charge_list),
                'max time': f"{max(time_list):.0f}",
                'min time': f"{min(time_list):.0f}",
                'median time': f"{np.median(time_list):.0f}",
                'mean time': np.mean(time_list),
            }
            d_first ={
                '(t1,q1)' : f"({time_list[0]:.0f}, {charge_list[0]:.3f})",
                '(t2,q2)' : f"({time_list[1]:.0f}, {charge_list[1]:.3f})",
                '(t3,q3)' : f"({time_list[2]:.0f}, {charge_list[2]:.3f})",
                '(t4,q4)' : f"({time_list[3]:.0f}, {charge_list[3]:.3f})",
                '(t5,q5)' : f"({time_list[4]:.0f}, {charge_list[4]:.3f})",
            }
            T10, T50 = self._get_elapsed_time_until_charge_fraction(charge_list, time_list)
            Q25, Q75, Qtotal = self._get_accumulated_charge_after_ns(charge_list, time_list)
            d_quantiles = {
                r"$Q_{0-25ns}$":f"{Q25:.3f}",
                r"$Q_{0-75ns}$":f"{Q75:.3f}",
                r"$Q_{\infty}$" :f"{Qtotal:.3f}",
                r'$T_{10\%}$':f"{T10:.0f}",
                r'$T_{50\%}$':f"{T50:.0f}",
                r'$\sigma_T $  ':f"{np.std(time_list):.3f}",
                }
            add_text_to_ax(0.35, 0.95, nice_string_output(d), ax, fontsize=18)
            add_text_to_ax(0.70, 0.95, nice_string_output(d_first), ax, fontsize=18)
            add_text_to_ax(0.70, 0.77, nice_string_output(d_quantiles), ax, fontsize=18)
            plt.tight_layout()
            fig_list.append(fig)

        return fig_list
    
    def _get_elapsed_time_until_charge_fraction(self, charge_list, time_list, p1=10, p2=50):
        _fillIncomplete = -1.0
        if len(charge_list) < 2:
            return _fillIncomplete, _fillIncomplete

        # Sort by time
        sorted_indices = np.argsort(time_list)
        charges = np.array(charge_list)[sorted_indices]
        times = np.array(time_list)[sorted_indices]

        Qtotal = np.sum(charges)
        cumulated = np.cumsum(charges)

        idx1 = np.searchsorted(cumulated, p1 / 100 * Qtotal, side="right")
        idx2 = np.searchsorted(cumulated, p2 / 100 * Qtotal, side="right")

        T1 = times[idx1] - times[0] if idx1 < len(times) else _fillIncomplete
        T2 = times[idx2] - times[0] if idx2 < len(times) else _fillIncomplete

        return T1, T2

    def _get_accumulated_charge_after_ns(self, charge_list, time_list, ns1=25, ns2=75):
        _fillIncomplete = -1.0
        if len(charge_list) < 1:
            return _fillIncomplete, _fillIncomplete, _fillIncomplete

        t0 = time_list[0]
        time_offsets = np.array(time_list) - t0
        charges = np.array(charge_list)

        Q1 = np.sum(charges[time_offsets < ns1])
        Q2 = np.sum(charges[time_offsets < ns2])
        Qtotal = np.sum(charges)

        return Q1, Q2, Qtotal

if __name__ == "__main__":
    source_root = "/lustre/hpc/project/icecube/HE_Nu_Aske_Oct2024/sqlite_pulses/Snowstorm/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("N_events", type=int, help="Number of events to process")
    
    parser.add_argument("energy_id", type=int, choices=[0, 1, 2],
                        help="Energy range: 0 = 10TeV–1PeV, 1 = 1PeV–100PeV")
    parser.add_argument("flavour_id", type=int, choices=[0, 1, 2],
                        help="Flavour: 0 = nu_e, 1 = nu_mu, 2 = nu_tau")
    args = parser.parse_args()

    # Integer-to-enum mapping
    energy_map = {
        0: EnergyRange.ER_10_TEV_1_PEV,
        1: EnergyRange.ER_1_PEV_100_PEV,
        2: EnergyRange.ER_100_TEV_100_PEV
    }
    flavour_map = {
        0: Flavour.E,
        1: Flavour.MU,
        2: Flavour.TAU,
    }
    N_events = args.N_events
    er = energy_map[args.energy_id]
    flavour = flavour_map[args.flavour_id]

    ALL_SETUPS = {
        EnergyRange.ER_10_TEV_1_PEV: {
            "pulse_cut": 150,
            "events": {
                Flavour.E: [7],
                Flavour.MU: [23],
                Flavour.TAU: [16],
            }
        },
        EnergyRange.ER_1_PEV_100_PEV: {
            "pulse_cut": 250,
            "events": {
                Flavour.E: [16],
                Flavour.MU: [31],
                Flavour.TAU: [0],
            }
        },
    }
    SETUP = ALL_SETUPS[er]
    event_ids = SETUP["events"][flavour]
    N_pulses_cut = SETUP["pulse_cut"]
    output_pdf = f"PulseMapFeatures_{er.string}_{flavour.alias}({N_events}events).pdf"
    
    print(f"Flavour: {flavour}, Energy Range: {er}, N_events: {N_events}")
    print(f" energy range string: {er.string}")
    print(f" flavour alias: {flavour.alias}")
    print(f"Output PDF: {output_pdf}")
    
    start_time = time.time()
    PulseMapFeatureStatViewer(source_root=source_root,
                                        output_pdf=output_pdf,
                                        N_events=N_events,
                                        energy_range=er,
                                        flavour=flavour,
                                        event_indices=SETUP["events"][flavour],
                                        N_pulses_cut=SETUP["pulse_cut"],
                                        )()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to complete: {elapsed_time:.2f} seconds")
