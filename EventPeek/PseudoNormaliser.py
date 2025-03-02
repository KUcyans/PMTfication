import numpy as np
import pyarrow as pa

class PseudoNormaliser:
    def __init__(self):
        self.position_scaler = 2e-3
        self.t_scaler = 3e-4
        self.t_shifter = 1e4
        self.Q_shifter = 2

    def __call__(self, features_np: np.ndarray, column_names: list[str]) -> np.ndarray:
        """
        Apply the normalisation steps directly to a NumPy array.
        """
        features_np = self._log10_charge(features_np, column_names)
        features_np = self._pseudo_normalise_dom_pos(features_np, column_names)
        features_np = self._pseudo_normalise_time(features_np, column_names)
        return features_np

    def _log10_charge(self, features_np: np.ndarray, column_names: list[str]) -> np.ndarray:
        q_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'Q25', 'Q75', 'Qtotal']
        for col_name in q_columns:
            if col_name in column_names:
                idx = column_names.index(col_name)
                features_np[:, idx] = np.log10(np.clip(features_np[:, idx], a_min=1e-9, a_max=None)) - self.Q_shifter
        return features_np

    def _pseudo_normalise_dom_pos(self, features_np: np.ndarray, column_names: list[str]) -> np.ndarray:
        pos_columns = ['dom_x', 'dom_y', 'dom_z', 'dom_x_rel', 'dom_y_rel', 'dom_z_rel']
        for col_name in pos_columns:
            if col_name in column_names:
                idx = column_names.index(col_name)
                features_np[:, idx] *= self.position_scaler
        return features_np

    def _pseudo_normalise_time(self, features_np: np.ndarray, column_names: list[str]) -> np.ndarray:
        t_columns_shift = ['t1', 't2', 't3']
        t_columns_scale = ['t1', 't2', 't3', 'T10', 'T50', 'sigmaT']

        # Time shift first
        for col_name in t_columns_shift:
            if col_name in column_names:
                idx = column_names.index(col_name)
                features_np[:, idx] -= self.t_shifter

        # Then scale
        for col_name in t_columns_scale:
            if col_name in column_names:
                idx = column_names.index(col_name)
                features_np[:, idx] *= self.t_scaler

        return features_np
