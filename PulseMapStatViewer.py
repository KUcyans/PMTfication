import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3 as sql
import sys
from collections import defaultdict

import time
from tqdm import tqdm

from matplotlib.backends.backend_pdf import PdfPages 
import argparse

sys.path.append('/groups/icecube/cyan/Utils')
from PlotUtils import setMplParam, getColour, getHistoParam 
from ExternalFunctions import nice_string_output, add_text_to_ax

class PulseMapStatViewer:
    def __init__(self, source_db, output_pdf, N_events):    
        setMplParam()
        self.df_features = self._convertDBtoDF(source_db, 'SRTInIcePulses', N_events)
        self.df_truth = self._convertDBtoDF(source_db, 'Truth' if 'Truth' in self.df_features.columns else 'truth', N_events)
        self._saveFigures(self.df_features, self.df_truth, output_pdf)
    
    def _convertDBtoDF(self, file: str, table: str, N_events: int = None) -> pd.DataFrame:
        con = sql.connect(file)
        N_events_total = self._get_total_events(con, table)
        print(f"Total number of events in {table}: {N_events_total}")
        if N_events is None or N_events >= N_events_total:
            # If no limit is set, retrieve all rows
            print(f'Reading ALL {N_events_total} events from {table}...')
            query = f"SELECT * FROM {table}"
        else:
            # First, get the distinct event_no values, limited by N_events
            print(f"Reading {N_events} events from {table}...")
            event_no_query = f"SELECT DISTINCT event_no FROM {table} LIMIT {N_events}"
            event_nos = pd.read_sql_query(event_no_query, con)["event_no"].tolist()
            
            # Construct the main query to filter by the selected event_nos
            event_no_list = ', '.join(map(str, event_nos))  # Convert list to a comma-separated string
            query = f"SELECT * FROM {table} WHERE event_no IN ({event_no_list})"
        
        # Run the main query to retrieve the filtered data
        df = pd.read_sql_query(query, con)
        con.close()
        
        return df
    
    def _get_total_events(self, con: sql.Connection, table: str) -> int:
        """Returns the total number of unique event_no in the specified table."""
        query = f"SELECT COUNT(DISTINCT event_no) AS total_events FROM {table}"
        result = pd.read_sql_query(query, con)
        return result['total_events'][0]        
    
    def _saveFigures(self, df_features, df_truth, file, isShow=False):
        # Create a PdfPages object to save all figures into one PDF file
        with PdfPages(f"{file}.pdf") as pdf:
            
            # List of functions that create plots
            plot_functions = [
                lambda: self._plotNDOMsPerEvent(df_features, file),
                lambda: self._plotNpulsesPerEvent(df_features, file),
                lambda: self._plotNpulsesPerDOM(df_features, file),
                lambda: self._plotChargePerPulse(df_features, file),
                lambda: self._plotTotalChargePerDOM(df_features, file),
                lambda: self._plotTotalChargePerEvent(df_features, file),
                lambda: self._plotPosition(df_features, 'dom_x', file),
                lambda: self._plotPosition(df_features, 'dom_y', file),
                lambda: self._plotPosition(df_features, 'dom_z', file),
                
                lambda: self._plotLog10energy(df_truth, file),
                lambda: self._plotZenith(df_truth, file),
                lambda: self._plotCosZenith(df_truth, file),
                lambda: self._plotAzimuth(df_truth, file),
                lambda: self._plotPosition(df_truth, 'position_y', file),
                lambda: self._plotPosition(df_truth, 'position_x', file),
                lambda: self._plotPosition(df_truth, 'position_z', file),
            ]
            
            # Iterate over the plotting functions, execute them, and save the generated figures to the PDF
            for plot_func in tqdm(plot_functions, desc="Generating plots"):
                fig, ax = plot_func()  # Call the plotting function to generate the figure
                pdf.savefig(fig)       # Save the current figure to the PDF
                if isShow:
                    plt.show()
                plt.close(fig)         # Close the figure to free up memory
                
            print(f"All plots have been saved in plots_{file}.pdf")
    
    def _plotNDOMsPerEvent(self, df_features, file):
        event_grouped = df_features.groupby('event_no')
        
        NDOMs_per_event = []
        
        for event_no, event_df in event_grouped:
            DOM_grouped = event_df.groupby(['string', 'dom_number'])
            
            NDOMs_per_event.append(len(DOM_grouped))
        
        NDOMs_per_event = np.array(NDOMs_per_event)
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(NDOMs_per_event, Nbins=33)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(NDOMs_per_event, bins=bins, color=getColour(0), histtype='step', linewidth=2)
        ax.set_xlabel('# of DOM/event')
        ax.set_ylabel('#')
        ax.set_title(f'# of DOMs per event ({file})')
        ax.set_yscale('log')
        d = {'Nevents': len(NDOMs_per_event), 
            'mean': np.mean(NDOMs_per_event), 
            'std': np.std(NDOMs_per_event),
            'max': np.max(NDOMs_per_event),
            'min': np.min(NDOMs_per_event),
            'binwidth': binwidth,
            'Nbins': Nbins}
        add_text_to_ax(0.75, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
        
    def _plotNpulsesPerEvent(self, df_features, file):
        event_grouped = df_features.groupby('event_no')
        Npulses = np.array([len(event_df) for event_no, event_df in event_grouped])
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(Npulses, Nbins=50)
        
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(Npulses, bins=bins, color=getColour(0), histtype='step', linewidth=2)
        ax.set_xlabel('# of pulses per event')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_title(f'# pulses/event ({file})')
        d = {'Nevents': len(Npulses), 
            'mean': np.mean(Npulses), 
            'std': np.std(Npulses),
            'max': np.max(Npulses),
            'min': np.min(Npulses),
            'binwidth': binwidth,
            'Nbins': Nbins}
        add_text_to_ax(0.75, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotNpulsesPerDOM(self, df_features, file):
        event_grouped_original = df_features.groupby('event_no')
        Nevents = len(event_grouped_original)
        
        Npulses_per_DOM = defaultdict(int)
        
        for event_no, event_df in event_grouped_original:
            DOM_grouped = event_df.groupby(['string', 'dom_number'])
            
            for (dom_string, dom_number), dom_df in DOM_grouped:
                Npulses_per_DOM[(dom_string, dom_number)] += len(dom_df)
        
        Npulses_per_DOM = dict(Npulses_per_DOM)
        print(f'unique DOMs: {len(Npulses_per_DOM)}')
        
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(np.array(list(Npulses_per_DOM.values())), Nbins=50)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(list(Npulses_per_DOM.values()), bins=bins, histtype='step', lw=2)
        ax.set_xlabel('# of pulses per DOM')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_title(f'# of pulses/DOM ({file})')
        d = {'Nevents': Nevents,
            'mean': np.mean(list(Npulses_per_DOM.values())),
            'std': np.std(list(Npulses_per_DOM.values())),
            'max': np.max(list(Npulses_per_DOM.values())),
            'min': np.min(list(Npulses_per_DOM.values())),
            'binwidth': binwidth,
            'Nbins': Nbins}
        add_text_to_ax(0.75, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotChargePerPulse(self, df_features, file):
        charges = df_features.charge
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(charges, Nbins=50)
        log_bins = np.logspace(np.log10(np.min(charges)), np.log10(np.max(charges)), Nbins)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(charges, bins=log_bins, histtype='step', lw=2)
        ax.set_xlabel('Charge')
        ax.set_ylabel('#')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_title(f'Charge/pulse ({file})')
        d = {'mean': np.mean(charges),
            'std': np.std(charges),
            'max': np.max(charges),
            'min': np.min(charges),
            'binwidth': binwidth,
            'Nbins': Nbins}
        add_text_to_ax(0.75, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotTotalChargePerDOM(self, df_features, file):
        event_grouped_original = df_features.groupby('event_no')
        Nevents = len(event_grouped_original)
        
        total_charge_per_DOM = defaultdict(int)
        
        for event_no, event_df in event_grouped_original:
            DOM_grouped = event_df.groupby(['string', 'dom_number'])
            
            for (dom_string, dom_number), dom_df in DOM_grouped:
                total_charge_per_DOM[(dom_string, dom_number)] += np.sum(dom_df.charge)
        
        total_charge_per_DOM = dict(total_charge_per_DOM)
        total_charge_per_DOM = np.array(list(total_charge_per_DOM.values()))
        
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(total_charge_per_DOM, Nbins=50)
        log_bins = np.logspace(np.log10(np.min(total_charge_per_DOM)), np.log10(np.max(total_charge_per_DOM)), Nbins)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(total_charge_per_DOM, bins=log_bins, histtype='step', lw=2)
        ax.set_xlabel('Total charge per DOM')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f'Total charge/DOM ({file})')
        d = {'Nevents': Nevents,
            'mean': np.mean(total_charge_per_DOM),
            'std': np.std(total_charge_per_DOM),
            'max': np.max(total_charge_per_DOM),
            'min': np.min(total_charge_per_DOM),
            'binwidth': binwidth,
            'Nbins': Nbins}
        add_text_to_ax(0.72, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotTotalChargePerEvent(self, df_features, file):
        event_grouped_original = df_features.groupby('event_no')
        Nevents = len(event_grouped_original)
        
        total_charge_per_event = defaultdict(int)
        
        for event_no, event_df in event_grouped_original:
            total_charge_per_event[event_no] = np.sum(event_df.charge)
        
        total_charge_per_event = dict(total_charge_per_event)
        total_charge_per_event = np.array(list(total_charge_per_event.values()))
        
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(total_charge_per_event, Nbins=50)
        log_bins = np.logspace(np.log10(np.min(total_charge_per_event)), np.log10(np.max(total_charge_per_event)), Nbins)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(total_charge_per_event, bins=log_bins, histtype='step', lw=2)
        ax.set_xlabel('Total charge per event')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f'Total charge/event ({file})')
        d = {'Nevents': Nevents,
            'mean': np.mean(total_charge_per_event),
            'std': np.std(total_charge_per_event),
            'max': np.max(total_charge_per_event),
            'min': np.min(total_charge_per_event),
            'binwidth': binwidth,
            'Nbins': Nbins}
        add_text_to_ax(0.7, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotLog10energy(self, df_truth, file):
        energies = df_truth.energy
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(energies, Nbins=50)
        log_bins = np.logspace(np.log10(np.min(energies)), np.log10(np.max(energies)), Nbins)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(energies, bins=log_bins, histtype='step', lw=2, color=getColour(2))
        ax.set_xlabel('log10(energy)')
        ax.set_ylabel('#')
        ax.set_xscale('log')
        ax.set_title(f'log10(energy) ({file})')
        d = {'mean': np.mean(energies),
            'std': np.std(energies),
            'max': np.max(energies),
            'min': np.min(energies),
            'Nbins': Nbins}
        add_text_to_ax(0.7, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotZenith(self, df_truth, file):
        zeniths = df_truth.zenith
        
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(zeniths, Nbins=50)
        
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(zeniths, bins=bins, histtype='step', lw=2, color=getColour(2))
        
        ax.set_xlabel('Zenith (radians)')
        ax.set_ylabel('#')
        ax.set_title(f'Zenith({file})')
        
        # Setting the x-axis to multiples of π
        ticks = np.arange(0, np.pi + 0.1, np.pi / 8)  # Adjust range and step as needed
        tick_labels = [f"{tick/np.pi:.1f}π" if tick != 0 else "0" for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0, np.pi)
        # Adding the summary statistics
        d = {
            'mean': np.mean(zeniths),
            'std': np.std(zeniths),
            'max': np.max(zeniths),
            'min': np.min(zeniths),
            'binwidth': binwidth,
            'Nbins': Nbins
        }
        
        add_text_to_ax(0.6, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotCosZenith(self, df_truth, file):
        cos_zeniths = np.cos(df_truth.zenith)
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(cos_zeniths, Nbins=50)
        
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(cos_zeniths, bins=bins, histtype='step', lw=2, color=getColour(2))
        
        ax.set_xlabel('cos(Zenith)')
        ax.set_ylabel('#')
        ax.set_title(f'cos(Zenith) ({file})')
        
        # Adding the summary statistics
        d = {
            'mean': np.mean(cos_zeniths),
            'std': np.std(cos_zeniths),
            'max': np.max(cos_zeniths),
            'min': np.min(cos_zeniths),
            'binwidth': binwidth,
            'Nbins': Nbins
        }
        
        add_text_to_ax(0.45, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotAzimuth(self, df_truth, file):
        azimuths = df_truth.azimuth
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(azimuths, Nbins=50)
        
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(azimuths, bins=bins, histtype='step', lw=2, color=getColour(2))
        
        ax.set_xlabel('Azimuth (radians)')
        ax.set_ylabel('#')
        ax.set_title(f'Azimuth ({file})')
        
        # Setting the x-axis to multiples of π
        ticks = np.arange(0, 2*np.pi + 0.1, np.pi / 4)  # Adjust range and step as needed
        tick_labels = [f"{tick/np.pi:.1f}π" if tick != 0 else "0" for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        
        # Adding the summary statistics
        d = {
            'mean': np.mean(azimuths),
            'std': np.std(azimuths),
            'max': np.max(azimuths),
            'min': np.min(azimuths),
            'binwidth': binwidth,
            'Nbins': Nbins
        }
        
        add_text_to_ax(0.73, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax
    
    def _plotPosition(self, df_truth, position, file):
        positions = df_truth[position]
        Nbins, binwidth, bins, counts, bin_centers = getHistoParam(positions, Nbins=50)
        
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(positions, bins=bins, histtype='step', lw=2, color=getColour(2))
        
        ax.set_xlabel(position)
        ax.set_ylabel('#')
        ax.set_title(f'{position} ({file})')
        
        # Adding the summary statistics
        d = {
            'mean': np.mean(positions),
            'std': np.std(positions),
            'max': np.max(positions),
            'min': np.min(positions),
            'binwidth': binwidth,
            'Nbins': Nbins
        }
        
        add_text_to_ax(0.6, 0.95, nice_string_output(d), ax, fontsize=12)
        return fig, ax

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run PulseMapStatViewer with input database and output PDF.")
    parser.add_argument("source_db", type=str, help="Path to the source SQLite database file.")
    parser.add_argument("output_pdf", type=str, help="Filename for the output PDF file.")
    parser.add_argument("--N", type=int, default=None, help="Number of events to read from the database for data.")

    # Parse arguments
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Instantiate and run the viewer
    viewer = PulseMapStatViewer(args.source_db, args.output_pdf, args.N)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print and log the time taken
    print(f"Time taken to complete: {elapsed_time:.2f} seconds")