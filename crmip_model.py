import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from matplotlib.patches import FancyArrowPatch
import time
from joblib import Parallel, delayed
import re

class CRMIPModel:
    def __init__(self, days, date, inj_rates, prod_rates, wc_data=None, well_coords=None):
        self.days = np.asarray(days, dtype=float)
        self.date = pd.to_datetime(date)
        self.dt = self.compute_dt(self.days)
        self.I = np.asarray(inj_rates, dtype=float)
        self.Qobs = np.asarray(prod_rates, dtype=float)
        self.inj_cols = inj_rates.columns
        self.prod_cols = prod_rates.columns
        self.T, self.n_inj = self.I.shape
        self.n_prod = self.Qobs.shape[1]
        self.WCobs = np.asarray(
            wc_data, dtype=float) if wc_data is not None else None
        if self.WCobs is not None:
            assert self.WCobs.shape == self.Qobs.shape, "Water cut data must match shape of prod_rates"
        self.well_coords = well_coords
        self.connectivity_mask = np.ones((self.n_prod, self.n_inj), dtype=bool)
        self.distance_threshold = None
        self.aquifer_index = None
        self.gentil_params = None
        self.CI_j = None

    def get_well_number(self, well_name):
        digits = re.findall(r'\d+', well_name)
        if digits:
            num_str = digits[0].lstrip('0')
            if not num_str:
                num_str = '0'
            return num_str
        return well_name[:4]  # Fallback to first 4 chars if no digits

    def compute_dt(self, days):
        dt = np.empty_like(days)
        dt[0] = 0.0
        dt[1:] = np.diff(days)
        return dt

    def _initial_q0_guess(self):
        Q0_guess = self.Qobs[0, :]
        inj_mean = self.I.mean(axis=0)
        inj_frac = inj_mean / \
            inj_mean.sum() if inj_mean.sum() > 0 else np.ones(self.n_inj)/self.n_inj
        has_aquifer = self.aquifer_index is not None
        real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
        q0s = np.zeros((self.n_prod, self.n_inj))
        for j in range(self.n_prod):
            q0s[j, :real_n_inj] = Q0_guess[j] * inj_frac[:real_n_inj]
            if self.connectivity_mask.shape[1] == self.n_inj:
                q0s[j, ~self.connectivity_mask[j, :]] = 0.0
            else:
                print(
                    f"Warning: connectivity_mask dimension mismatch (shape: {self.connectivity_mask.shape}, expected: ({self.n_prod}, {self.n_inj})). Skipping mask application.")
        if has_aquifer:
            connected_inj_per_prod = np.zeros((self.T, self.n_prod))
            for j in range(self.n_prod):
                conn_mask = self.connectivity_mask[j, :real_n_inj]
                connected_inj_per_prod[:, j] = self.I[:,
                                                      :real_n_inj][:, conn_mask].sum(axis=1)
            shortfall_per_prod = np.maximum(
                0.0, self.Qobs - connected_inj_per_prod)
            mean_short_per_prod = shortfall_per_prod.mean(axis=0)
            mean_total_short = np.sum(mean_short_per_prod)
            f_a = np.zeros(self.n_prod)
            if mean_total_short > 0:
                f_a = mean_short_per_prod / mean_total_short
            for j in range(self.n_prod):
                q0s[j, self.aquifer_index] = Q0_guess[j] * f_a[j]
        return q0s

    def fit_stepwise(self, max_time_per_step=None, verbose=False, n_jobs=-1, optimizer_method='L-BFGS-B'):
        T, n_inj, n_prod = self.T, self.n_inj, self.n_prod
        self.f_time = np.zeros((T, n_prod, n_inj))
        self.tau_time = np.zeros((T, n_prod, n_inj))
        self.qij_time = np.zeros((T, n_prod, n_inj))
        self.Qsim_stepwise = np.zeros((T, n_prod))
        q0s = self._initial_q0_guess()
        self.qij_time[0, :, :] = q0s
        tau_guess = np.full(
            (n_prod, n_inj), (self.days[-1] - self.days[0]) / 4.0)
        if self.well_coords is not None:
            for j in range(n_prod):
                prod = self.prod_cols[j]
                px, py = self.well_coords.get(prod, (0, 0))
                dists = []
                has_aquifer = self.aquifer_index is not None
                real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
                for i in range(real_n_inj):
                    ix, iy = self.well_coords.get(self.inj_cols[i], (0, 0))
                    dist = np.sqrt((px - ix)**2 + (py - iy)**2)
                    dists.append(dist if np.isfinite(dist) else np.inf)
                dists = np.array(dists)
                valid_dists = dists[np.isfinite(dists)]
                if len(valid_dists) > 0:
                    mean_dist = valid_dists.mean()
                    tau_guess[j, :real_n_inj] = np.clip(
                        50 + 10 * dists / mean_dist, 50, 2000)
                    tau_guess[j, :real_n_inj][~np.isfinite(
                        dists)] = np.random.uniform(100, 1000)
                else:
                    tau_guess[j, :] = np.random.uniform(100, 1000, n_inj)
                if has_aquifer:
                    tau_guess[j, self.aquifer_index] = np.random.uniform(
                        50, 200)
        else:
            tau_guess = np.random.uniform(100, 1000, (n_prod, n_inj))
        self.tau_time[0, :, :] = tau_guess
        has_aquifer = self.aquifer_index is not None
        real_injectors = list(range(n_inj))
        real_n_inj = len(real_injectors)
        if has_aquifer:
            real_injectors.remove(self.aquifer_index)
            real_n_inj -= 1

        def optimize_producer(j, k):
            prod = self.prod_cols[j]
            if self.Qobs[k, j] <= 1e-12:
                return np.zeros(n_inj), np.zeros(n_inj), np.zeros(n_inj), 0.0
            q_prev = self.qij_time[k-1, j, :]

            # Create combined mask: distance-based and active injectors
            active_mask = self.I[k, :] > 0
            combined_mask = self.connectivity_mask[j, :] & active_mask
            n_active = np.sum(combined_mask)
            if n_active == 0:
                # No active injection, just decay
                a = np.exp(-self.dt[k] / self.tau_time[k-1, j, :])
                q_new = q_prev * a
                qsim = q_new.sum()
                f_opt = self.f_time[k-1, j, :].copy()
                # Carry over, but zero inactive
                tau_opt = self.tau_time[k-1, j, :].copy()
                f_opt[~combined_mask] = 0.0  # Ensure inactive f_ij = 0
                # NEW: Zero tau for inactive (instant decay)
                tau_opt[~combined_mask] = 0.0
                return f_opt, tau_opt, q_new, qsim

            def obj_x(x):
                # x: f_active, tau_active for combined_mask
                f_active = np.clip(x[:n_active], 0, np.inf)
                tau_active = np.maximum(x[n_active:], 1e-3)
                a_active = np.exp(-self.dt[k] / tau_active)
                q_new_active = q_prev[combined_mask] * a_active + \
                    (1 - a_active) * (f_active * self.I[k, combined_mask])
                q_new = np.zeros(n_inj)
                q_new[combined_mask] = q_new_active
                q_new[~combined_mask] = q_prev[~combined_mask] * \
                    np.exp(-self.dt[k] / self.tau_time[k-1, j, ~combined_mask])
                qsim = np.sum(q_new)
                resid = self.Qobs[k, j] - qsim
                # Add regularization for smoothness in tau
                prev_tau_active = self.tau_time[k-1, j, combined_mask]
                reg_term = 0.01 * np.sum((tau_active - prev_tau_active)**2)
                return resid**2 + reg_term

            # Initial for active only, with perturbation for variety
            f0_active = self.f_time[k-1, j, combined_mask] * 0.5
            f0_active = np.clip(f0_active, 1e-6, 0.99)
            prev_tau_active = self.tau_time[k-1, j, combined_mask]
            tau0_active = prev_tau_active + \
                np.random.uniform(-50, 50, len(prev_tau_active))
            tau0_active = np.clip(tau0_active, 10, 2000)
            x0 = np.concatenate([f0_active, tau0_active])

            # Bounds for active only, tightened for tau
            bounds_active = []
            active_indices = np.where(combined_mask)[0]
            for ii in active_indices:
                low_f = 0.0
                up_f = 1.0 if self.connectivity_mask[j, ii] else 0.0
                if has_aquifer and ii == self.aquifer_index:
                    up_f = np.inf
                bounds_active.append((low_f, up_f))
            bounds_active += [(60, 2000)] * n_active  # Tighter bounds for tau
            bounds = bounds_active

            start = time.time()
            last_x = x0.copy()

            def cb(xk, state=None):
                nonlocal last_x
                last_x = xk.copy()
                if max_time_per_step is not None and (time.time() - start) > max_time_per_step:
                    raise KeyboardInterrupt

            try:
                options = {'maxiter': 200}  # Increased iterations
                res = minimize(obj_x, x0, method=optimizer_method,
                               bounds=bounds, options=options, callback=cb)
                x_opt = res.x
            except KeyboardInterrupt:
                x_opt = last_x
                if verbose:
                    print(f"  step {k} prod {j} timed out, using best-so-far")

            # Reconstruct full f_opt, tau_opt
            f_opt = self.f_time[k-1, j, :].copy()
            tau_opt = self.tau_time[k-1, j, :].copy()
            f_opt[combined_mask] = np.clip(x_opt[:n_active], 0.0, np.inf)
            tau_opt[combined_mask] = np.maximum(x_opt[n_active:], 1e-3)
            f_opt[~combined_mask] = 0.0  # Ensure inactive f_ij = 0
            # NEW: Zero tau for inactive (instant decay)
            tau_opt[~combined_mask] = 0.0

            a = np.exp(-self.dt[k] / tau_opt)
            q_new = q_prev * a + (1 - a) * (f_opt * self.I[k, :])
            qsim = q_new.sum()
            return f_opt, tau_opt, q_new, qsim

        for k in range(1, T):
            if verbose:
                print(f"Step {k}/{T-1}")
            results = Parallel(n_jobs=n_jobs)(
                delayed(optimize_producer)(j, k) for j in range(n_prod))
            for j, (f_opt, tau_opt, q_new, qsim) in enumerate(results):
                self.f_time[k, j, :] = f_opt
                self.tau_time[k, j, :] = tau_opt
                self.qij_time[k, j, :] = q_new
                self.Qsim_stepwise[k, j] = qsim
            # Normalize per injector columns for real injectors
            for ii in real_injectors:
                col_sum = np.sum(self.f_time[k, :, ii])
                if col_sum > 1.0:
                    self.f_time[k, :, ii] /= col_sum
            # Normalize per producer rows for real injectors only
            for jj in range(n_prod):
                row_sum_real = np.sum(self.f_time[k, jj, :real_n_inj])
                if row_sum_real > 1.0:
                    self.f_time[k, jj, :real_n_inj] /= row_sum_real
            # Re-compute q after normalization
            for j in range(n_prod):
                f_opt = self.f_time[k, j, :]
                tau_opt = self.tau_time[k, j, :]
                a = np.exp(-self.dt[k] / tau_opt)
                q_prev = self.qij_time[k-1, j, :]
                q_new = q_prev * a + (1 - a) * (f_opt * self.I[k, :])
                self.qij_time[k, j, :] = q_new
                self.Qsim_stepwise[k, j] = np.sum(q_new)
        self.f_time[0, :, :] = (
            self.qij_time[0, :, :] / (self.qij_time[0, :, :].sum(axis=1, keepdims=True) + 1e-12))
        # Normalize initial f_time[0] columns for real injectors
        active_mask_t0 = self.I[0, :] > 0
        for ii in real_injectors:
            col_sum = np.sum(self.f_time[0, :, ii])
            if col_sum > 1.0:
                self.f_time[0, :, ii] /= col_sum
            if not active_mask_t0[ii]:
                self.f_time[0, :, ii] = 0.0  # Ensure inactive f_ij = 0 at t=0
        # Normalize initial f_time[0] rows for real injectors only
        for jj in range(n_prod):
            row_sum_real = np.sum(self.f_time[0, jj, :real_n_inj])
            if row_sum_real > 1.0:
                self.f_time[0, jj, :real_n_inj] /= row_sum_real
        # Adjust qij[0] to maintain row sums
        for j in range(n_prod):
            row_sum_f = np.sum(self.f_time[0, j, :])
            if row_sum_f > 0:
                self.qij_time[0, j, :] = self.f_time[0, j, :] * \
                    self.Qobs[0, j] / row_sum_f
        self.Qsim_stepwise_trust = self.Qsim_stepwise.copy()
        return {"f_time": self.f_time, "tau_time": self.tau_time, "Qsim": self.Qsim_stepwise}

    def add_single_aquifer(self, name_prefix="AQUIFER", set_constant=True, modify_self=True, Qsim_prev=None):
        real_n_inj = self.n_inj
        if Qsim_prev is not None:
            shortfall_per_prod = np.maximum(0.0, self.Qobs - Qsim_prev)
        else:
            connected_inj_per_prod = np.zeros((self.T, self.n_prod))
            for j in range(self.n_prod):
                conn_mask = self.connectivity_mask[j, :]
                connected_inj_per_prod[:, j] = self.I[:, conn_mask].sum(axis=1)
            shortfall_per_prod = np.maximum(
                0.0, self.Qobs - connected_inj_per_prod)
        total_shortfall = np.max(shortfall_per_prod, axis=1)
        if set_constant:
            A = np.full(self.T, total_shortfall.max())
        else:
            A = total_shortfall
        aquifer = {"total": A}
        if modify_self:
            self.I = np.hstack([self.I, A.reshape(-1, 1)])
            self.inj_cols = self.inj_cols.tolist() + [name_prefix]
            self.n_inj += 1
            self.aquifer_index = self.n_inj - 1
            old_mask = self.connectivity_mask.copy()
            new_mask = np.zeros((self.n_prod, self.n_inj), dtype=bool)
            new_mask[:, :old_mask.shape[1]] = old_mask
            new_mask[:, self.aquifer_index] = True
            self.connectivity_mask = new_mask
        return aquifer, self.aquifer_index

    def add_distance_constraints(self, constraint_type='threshold', threshold=None, matrix_file=None, well_coords=None):
        if well_coords is None and constraint_type in ['threshold', 'auto']:
            raise ValueError(
                "well_coords is required for 'threshold' or 'auto' constraint_type.")
        if constraint_type not in ['threshold', 'matrix', 'auto']:
            raise ValueError(
                "constraint_type must be 'threshold', 'matrix', or 'auto'.")
        injectors = list(self.inj_cols)
        producers = list(self.prod_cols)
        real_inj_mask = np.array(
            [not inj.startswith('AQUIFER') for inj in injectors])
        real_inj_indices = np.where(real_inj_mask)[0]
        connectivity_mask = np.ones((self.n_prod, self.n_inj), dtype=bool)
        if constraint_type == 'threshold':
            if threshold is None:
                raise ValueError(
                    "threshold is required for 'threshold' constraint_type.")
            self.distance_threshold = threshold
            for j, prod in enumerate(producers):
                if prod not in well_coords:
                    print(
                        f"Warning: No coordinates for producer {prod}, assuming all connections active.")
                    continue
                px, py = well_coords[prod]
                for ii, i in enumerate(real_inj_indices):
                    inj = injectors[i]
                    if inj not in well_coords:
                        connectivity_mask[j, i] = False
                        continue
                    ix, iy = well_coords[inj]
                    dist = np.sqrt((px - ix)**2 + (py - iy)**2)
                    if dist > threshold:
                        connectivity_mask[j, i] = False
        elif constraint_type == 'matrix':
            if matrix_file is None:
                raise ValueError(
                    "matrix_file is required for 'matrix' constraint_type.")
            self.distance_threshold = None
            df = pd.read_excel(matrix_file, header=0, index_col=0)
            prod_map = {prod: j for j, prod in enumerate(producers)}
            inj_map = {inj: i for i, inj in enumerate(
                injectors) if not injectors[i].startswith('AQUIFER')}
            for row_name, row in df.iterrows():
                if row_name not in inj_map:
                    continue
                i = inj_map[row_name]
                for col_name, val in row.items():
                    if col_name not in prod_map:
                        continue
                    j = prod_map[col_name]
                    connectivity_mask[j, i] = bool(val)
        elif constraint_type == 'auto':
            self.distance_threshold = None
            for j, prod in enumerate(producers):
                if prod not in well_coords:
                    print(
                        f"Warning: No coordinates for producer {prod}, assuming all connections active.")
                    continue
                px, py = self.well_coords[prod]
                dists = []
                for ii, i in enumerate(real_inj_indices):
                    inj = injectors[i]
                    if inj not in well_coords:
                        dists.append(np.inf)
                        continue
                    ix, iy = well_coords[inj]
                    dist = np.sqrt((px - ix)**2 + (py - iy)**2)
                    dists.append(dist)
                if len(dists) == 0 or not np.any(np.isfinite(dists)):
                    continue
                mean_dist = np.mean([d for d in dists if np.isfinite(d)])
                for ii, i in enumerate(real_inj_indices):
                    if dists[ii] > mean_dist:
                        connectivity_mask[j, i] = False
        self.connectivity_mask = connectivity_mask
        print(
            f"Connectivity mask updated with {constraint_type} constraints. Non-zero connections: {np.sum(connectivity_mask)}")

    def fit_gentil(self, prelim_model='static_noaq', wc_threshold=0.5, a_bounds=(1e-6, 1e6), b_bounds=(0.1, 5.0)):
        has_aquifer = self.aquifer_index is not None
        real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
        if prelim_model.startswith('static'):
            method_key = list(self.results.keys())[0]
            f_matrix = np.zeros((self.n_prod, real_n_inj))
            for j in range(self.n_prod):
                f_matrix[j, :] = self.results[method_key][j]['f'][:real_n_inj]
            cum_I = np.cumsum(self.I[:, :real_n_inj]
                              * self.dt.reshape(-1, 1), axis=0)
        else:
            f_matrix = np.mean(self.f_time[:, :, :real_n_inj], axis=0)
            cum_I = np.cumsum(self.I[:, :real_n_inj]
                              * self.dt.reshape(-1, 1), axis=0)
        self.CI_j = np.zeros((self.T, self.n_prod))
        for j in range(self.n_prod):
            self.CI_j[:, j] = np.sum(f_matrix[j, :] * cum_I, axis=1)
        self.gentil_params = {}
        for j in range(self.n_prod):
            wc_hist = self.WCobs[:, j]
            if np.any(wc_hist < wc_threshold):
                high_wc_mask = wc_hist >= wc_threshold
                if np.sum(high_wc_mask) < 3:
                    print(
                        f"Warning: Insufficient high WC data for producer {self.prod_cols[j]}")
                    self.gentil_params[j] = {'a': 1.0, 'b': 1.0}
                    continue
                ci_high = self.CI_j[high_wc_mask, j]
                wor_obs = wc_hist[high_wc_mask] / \
                    (1 - wc_hist[high_wc_mask] + 1e-12)
                log_wor_obs = np.log(wor_obs + 1e-12)
                log_ci = np.log(ci_high + 1e-12)

                def obj_log(params):
                    a, b = np.exp(params[0]), params[1]
                    log_wor_sim = np.log(a) + b * log_ci
                    return np.sum((log_wor_obs - log_wor_sim)**2)

                init_params = [0.0, 1.0]
                bounds = [(-10, 10), b_bounds]
                res = minimize(obj_log, init_params,
                               bounds=bounds, method='L-BFGS-B')
                a_fit = np.exp(res.x[0])
                b_fit = res.x[1]
                self.gentil_params[j] = {'a': a_fit, 'b': b_fit}
            else:
                self.gentil_params[j] = {'a': 1.0, 'b': 1.0}

    def simulate_wc_gentil(self):
        if self.gentil_params is None:
            raise ValueError("Run fit_gentil() first.")
        WC_sim = np.zeros((self.T, self.n_prod))
        for j in range(self.n_prod):
            a, b = self.gentil_params[j]['a'], self.gentil_params[j]['b']
            ci = self.CI_j[:, j]
            wor = a * (ci ** b)
            WC_sim[:, j] = wor / (wor + 1)
        return WC_sim

    def run_comparison_models(self, verbose=True, stepwise_kwargs=None, aquifer_name_prefix="AQUIFER"):
        stepwise_kwargs = stepwise_kwargs or {}
        self.comparison_results = {}
        self.comparison_params = {}
        self.comparison_results_oil = {}
        self.comparison_wc = {}
        self.gentil_params_per_model = {}
        I_orig = self.I.copy()
        inj_cols_orig = self.inj_cols.copy()
        n_inj_orig = self.n_inj
        connectivity_mask_orig = self.connectivity_mask.copy()
        aquifer_index_orig = getattr(self, 'aquifer_index', None)
        dynamic_noaq_Qsim = None  # Initialize to None
        if verbose:
            print("Running dynamic CRMIP fit (no aquifer)...")
        try:
            self.I = I_orig.copy()
            self.inj_cols = inj_cols_orig.copy()
            self.n_inj = n_inj_orig
            self.connectivity_mask = connectivity_mask_orig.copy()
            self.aquifer_index = aquifer_index_orig
            if hasattr(self, 'f_time'):
                del self.f_time
            if hasattr(self, 'tau_time'):
                del self.tau_time
            if hasattr(self, 'qij_time'):
                del self.qij_time
            if hasattr(self, 'Qsim_stepwise'):
                del self.Qsim_stepwise
            if hasattr(self, 'Qsim_stepwise_trust'):
                del self.Qsim_stepwise_trust
            self.fit_stepwise(**stepwise_kwargs)
            if self.n_inj != n_inj_orig:
                print(
                    f"Warning: dynamic_noaq n_inj ({self.n_inj}) does not match original n_inj ({n_inj_orig}). Correcting...")
                self.n_inj = n_inj_orig
                self.f_time = self.f_time[:, :, :n_inj_orig]
                self.tau_time = self.tau_time[:, :, :n_inj_orig]
                self.qij_time = self.qij_time[:, :, :n_inj_orig]
            self.comparison_params["dynamic_noaq"] = {
                "f_time": self.f_time.copy(),
                "tau_time": self.tau_time.copy(),
                "n_inj": self.n_inj
            }
            Qsim_liq = self.Qsim_stepwise_trust
            Qsim_constr = Qsim_liq
            self.comparison_results["dynamic_noaq"] = Qsim_constr
            if self.WCobs is not None:
                self.fit_gentil(prelim_model='dynamic_noaq')
                WC_gentil = self.simulate_wc_gentil()
                self.comparison_wc["dynamic_noaq"] = WC_gentil
                self.gentil_params_per_model["dynamic_noaq"] = self.gentil_params.copy(
                )
                self.comparison_results_oil["dynamic_noaq"] = Qsim_constr * (
                    1 - WC_gentil)
            else:
                self.comparison_results_oil["dynamic_noaq"] = Qsim_constr * (
                    1 - np.zeros_like(self.Qobs))
            dynamic_noaq_Qsim = Qsim_liq
        except Exception as e:
            print(f"Dynamic fit (no aquifer) failed: {e}")
            self.comparison_results["dynamic_noaq"] = None
            self.comparison_results_oil["dynamic_noaq"] = None
        if verbose:
            print("Running dynamic CRMIP fit (with aquifer)...")
        try:
            self.add_single_aquifer(name_prefix=aquifer_name_prefix,
                                    set_constant=False, modify_self=True, Qsim_prev=dynamic_noaq_Qsim)
            self.fit_stepwise(**stepwise_kwargs)
            self.comparison_params["dynamic_withaq"] = {
                "f_time": self.f_time.copy(),
                "tau_time": self.tau_time.copy(),
                "n_inj": self.n_inj
            }
            Qsim_liq = self.Qsim_stepwise_trust
            Qsim_constr = Qsim_liq
            self.comparison_results["dynamic_withaq"] = Qsim_constr
            if self.WCobs is not None:
                self.fit_gentil(prelim_model='dynamic_withaq')
                WC_gentil = self.simulate_wc_gentil()
                self.comparison_wc["dynamic_withaq"] = WC_gentil
                self.gentil_params_per_model["dynamic_withaq"] = self.gentil_params.copy(
                )
                self.comparison_results_oil["dynamic_withaq"] = Qsim_constr * (
                    1 - WC_gentil)
            else:
                self.comparison_results_oil["dynamic_withaq"] = Qsim_constr * (
                    1 - np.zeros_like(self.Qobs))
        except Exception as e:
            print(f"Dynamic fit (with aquifer) failed: {e}")
            self.comparison_results["dynamic_withaq"] = None
            self.comparison_results_oil["dynamic_withaq"] = None
        finally:
            self.I = I_orig.copy()
            self.inj_cols = inj_cols_orig.copy()
            self.n_inj = n_inj_orig
            self.connectivity_mask = connectivity_mask_orig.copy()
            self.aquifer_index = aquifer_index_orig
            if hasattr(self, 'f_time'):
                del self.f_time
            if hasattr(self, 'tau_time'):
                del self.tau_time
            if hasattr(self, 'qij_time'):
                del self.qij_time
            if hasattr(self, 'Qsim_stepwise'):
                del self.Qsim_stepwise
            if hasattr(self, 'Qsim_stepwise_trust'):
                del self.Qsim_stepwise_trust
        if "dynamic_noaq" in self.comparison_params:
            if self.comparison_params["dynamic_noaq"]["n_inj"] != n_inj_orig:
                print(
                    f"Warning: dynamic_noaq n_inj ({self.comparison_params['dynamic_noaq']['n_inj']}) does not match original n_inj ({n_inj_orig}). Correcting...")
                self.comparison_params["dynamic_noaq"]["n_inj"] = n_inj_orig
                self.comparison_params["dynamic_noaq"]["f_time"] = self.comparison_params["dynamic_noaq"]["f_time"][:, :, :n_inj_orig]
                self.comparison_params["dynamic_noaq"]["tau_time"] = self.comparison_params[
                    "dynamic_noaq"]["tau_time"][:, :, :n_inj_orig]
        if verbose:
            print(
                "Comparison runs completed. Results and parameters stored, including oil via Gentil WC.")
        return self.comparison_results, self.comparison_params

    def plot_comparison_models(self, plot_oil=True):
        if not hasattr(self, "comparison_results"):
            raise AttributeError(
                "Run run_comparison_models() first to generate comparison results.")
        models = ["dynamic_withaq, dynamic_noaq"]
        colors = ["red","blue"]
        linestyles = ["-","*"]
        results_dict = self.comparison_results_oil if plot_oil else self.comparison_results
        if plot_oil and self.WCobs is not None:
            obs_data = self.Qobs * (1 - self.WCobs)
        else:
            obs_data = self.Qobs
        ylabel = "Oil Rate (STB/D)" if plot_oil else "Liquid Rate (STB/D)"
        for j, prod in enumerate(self.prod_cols):
            plt.figure(figsize=(12, 6))
            plt.plot(self.date, obs_data[:, j], 'ko',
                     label=f"{prod} Observed", markersize=4)
            for model, color, linestyle in zip(models, colors, linestyles):
                if results_dict.get(model) is not None:
                    label = f"{prod} {model.replace('_', ' ').title()}"
                    if plot_oil and np.any(results_dict[model][:, j] == np.minimum(results_dict[model][:, j], obs_data[:, j])):
                        label += " (Constrained)"
                    plt.plot(
                        self.date, results_dict[model][:, j], linestyle, label=label, color=color, alpha=0.8)
            plt.xlabel("Date")
            plt.ylabel(ylabel)
            plt.title(
                f"Producer {prod} - Model Comparison {'(Oil)' if plot_oil else '(Liquid)'}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

    def plot_gentil_comparison(self):
        if not hasattr(self, "comparison_wc") or self.WCobs is None:
            raise AttributeError(
                "Run run_comparison_models() with WC data first to generate Gentil comparisons.")
        models = ["dynamic_noaq", "dynamic_withaq"]
        colors = ["blue", "red"]
        linestyles = ["--", "-"]
        for j, prod in enumerate(self.prod_cols):
            plt.figure(figsize=(12, 6))
            plt.plot(self.date, self.WCobs[:, j], 'ko',
                     label=f"{prod} Observed WC", markersize=4)
            for model, color, linestyle in zip(models, colors, linestyles):
                if self.comparison_wc.get(model) is not None:
                    label = f"{prod} Gentil WC ({model.replace('_', ' ').title()})"
                    plt.plot(
                        self.date, self.comparison_wc[model][:, j], linestyle, label=label, color=color, alpha=0.8)
            plt.xlabel("Date")
            plt.ylabel("Water Cut")
            plt.title(f"Producer {prod} - Gentil WC Model Comparison")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

    def export_comparison_params(self, filename="compare_params.xlsx"):
        if not hasattr(self, "comparison_params"):
            raise AttributeError("Run run_comparison_models() first.")
        has_aquifer = self.aquifer_index is not None
        real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            for model in ["dynamic_noaq", "dynamic_withaq"]:
                if model in self.comparison_params and self.comparison_params[model]:
                    n_inj = min(
                        real_n_inj, self.comparison_params[model]["n_inj"])
                    for t in range(self.T):
                        rows = []
                        for j, prod in enumerate(self.prod_cols):
                            for i, inj in enumerate(self.inj_cols[:n_inj]):
                                rows.append({
                                    "Time": self.days[t],
                                    "Date": self.date[t],
                                    "Producer": prod,
                                    "Injector": inj,
                                    "f_ij": self.comparison_params[model]["f_time"][t, j, i],
                                    "tau_ij": self.comparison_params[model]["tau_time"][t, j, i]
                                })
                        df = pd.DataFrame(rows)
                        df.to_excel(
                            writer, sheet_name=f"{model}_t{t}", index=False)
            if self.gentil_params_per_model:
                for model, gentil_params in self.gentil_params_per_model.items():
                    gentil_df = pd.DataFrame([
                        {"Producer": self.prod_cols[j],
                            "a": params['a'], "b": params['b']}
                        for j, params in gentil_params.items()
                    ])
                    gentil_df.to_excel(
                        writer, sheet_name=f"gentil_{model}", index=False)
        print(f"Parameters exported to {filename}")

    def plot_connectivity(self, method, well_coords, arrow_width=1.0, cmap='viridis', show_all_arrows=False):
        if method not in self.comparison_params:
            raise AttributeError(
                f"Method {method} not found in comparison_params. Run run_comparison_models() first.")
        params = self.comparison_params[method]
        injectors = list(self.inj_cols)
        producers = list(self.prod_cols)
        n_inj = params.get("n_inj", len(injectors))
        f_matrix = pd.DataFrame(
            index=injectors[:n_inj], columns=producers, dtype=float)
        for j, prod in enumerate(producers):
            if not params.get(j):
                continue
            for i, inj in enumerate(injectors[:n_inj]):
                f_matrix.loc[inj, prod] = params[j]["f"][i]
        f_max = f_matrix.max().max()
        if not np.isfinite(f_max) or f_max <= 0:
            f_max = 1.0
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        plt.title(f"Injector-Producer Connectivity ({method})")
        self._plot_connectivity_base(
            ax, f_matrix, well_coords, injectors[:n_inj], producers,
            self.connectivity_mask[:,
                                   :n_inj], arrow_width, cmap, show_all_arrows, vmax=f_max
        )
        plt.show()

    def plot_connectivity_over_time(self, method, well_coords, timesteps=None, arrow_width=1.0, cmap='viridis', show_all_arrows=False):
        if method not in ["dynamic_noaq", "dynamic_withaq"]:
            raise ValueError(
                "Method must be 'dynamic_noaq' or 'dynamic_withaq' for time-varying connectivity.")
        if not hasattr(self, "comparison_params") or method not in self.comparison_params:
            raise AttributeError(
                f"Run run_comparison_models() first to generate {method} parameters.")
        f_time = self.comparison_params[method]["f_time"]
        has_aquifer = self.aquifer_index is not None
        real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
        n_inj = min(real_n_inj, self.comparison_params[method]["n_inj"])
        if f_time.shape[2] > n_inj:
            print(
                f"Warning: Truncating f_time from {f_time.shape[2]} to {n_inj} injectors to exclude aquifer")
            f_time = f_time[:, :, :n_inj]
        if f_time.shape[1] != len(self.prod_cols) or f_time.shape[2] != n_inj:
            raise ValueError(
                f"f_time shape {f_time.shape} does not match expected (*, {len(self.prod_cols)}, {n_inj})")
        if self.connectivity_mask.shape[1] < n_inj:
            raise ValueError(
                f"connectivity_mask has fewer injectors ({self.connectivity_mask.shape[1]}) than required ({n_inj})")
        connectivity_mask = self.connectivity_mask[:, :n_inj]
        if connectivity_mask.shape != (len(self.prod_cols), n_inj):
            raise ValueError(
                f"connectivity_mask shape {connectivity_mask.shape} does not match expected ({len(self.prod_cols)}, {n_inj})")
        if isinstance(well_coords, pd.DataFrame):
            well_coords = dict(zip(well_coords["Well"], zip(
                well_coords["X"], well_coords["Y"])))
        inj_cols_list = list(self.inj_cols[:n_inj])
        prod_cols_list = list(self.prod_cols)
        for well in inj_cols_list + prod_cols_list:
            coords = well_coords.get(well, (0, 0))
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                print(
                    f"Warning: Invalid coordinates for well {well}: {coords}, using (0, 0)")
                well_coords[well] = (0, 0)
            else:
                x, y = coords
                if isinstance(x, (np.ndarray, list)) or isinstance(y, (np.ndarray, list)):
                    print(
                        f"Warning: Array coordinates for well {well}: {coords}, using first values")
                    x = float(x[0]) if isinstance(
                        x, (np.ndarray, list)) and len(x) > 0 else 0.0
                    y = float(y[0]) if isinstance(
                        y, (np.ndarray, list)) and len(y) > 0 else 0.0
                    well_coords[well] = (x, y)
                if not (np.isfinite(x) and np.isfinite(y)):
                    print(
                        f"Warning: Non-finite coordinates for well {well}: ({x}, {y}), using (0, 0)")
                    well_coords[well] = (0, 0)
        timesteps = timesteps or [0, len(self.days)//2, len(self.days)-1]
        for t in timesteps:
            if not (0 <= t < self.T):
                raise ValueError(
                    f"Timestep {t} must be between 0 and {self.T-1}")
            # Apply active injector mask for this timestep
            active_mask = self.I[t, :n_inj] > 0
            combined_mask = connectivity_mask & active_mask
            f_matrix = f_time[t, :, :].copy()
            # Set f_ij = 0 for inactive injectors
            f_matrix[:, ~active_mask] = 0.0
            print(
                f"Timestep {t}: f_matrix shape {f_matrix.shape}, active injectors: {np.sum(active_mask)}")
            f_max = f_matrix.max()
            if not np.isfinite(f_max) or f_max <= 0:
                f_max = 1.0
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            plt.title(
                f"Injector-Producer Connectivity ({method}) - Day {self.days[t]}")
            self._plot_connectivity_base(
                ax, f_matrix, well_coords, inj_cols_list, prod_cols_list,
                combined_mask, arrow_width, cmap, show_all_arrows, vmax=f_max
            )
            plt.show()

    def _plot_connectivity_base(self, ax, f_matrix, well_coords, injectors, producers, connectivity_mask, arrow_width=1.0, cmap='viridis', show_all_arrows=False, vmax=None):
        if isinstance(f_matrix, np.ndarray):
            if f_matrix.shape != (len(producers), len(injectors)):
                raise ValueError(
                    f"f_matrix shape {f_matrix.shape} does not match expected ({len(producers)}, {len(injectors)})")
        elif not isinstance(f_matrix, pd.DataFrame):
            raise ValueError(
                "f_matrix must be a numpy array or pandas DataFrame")
        if connectivity_mask.shape != (len(producers), len(injectors)):
            raise ValueError(
                f"connectivity_mask shape {connectivity_mask.shape} does not match expected ({len(producers)}, {len(injectors)})")
        if isinstance(well_coords, pd.DataFrame):
            well_coords = dict(zip(well_coords["Well"], zip(
                well_coords["X"], well_coords["Y"])))
        injectors_list = list(injectors)
        producers_list = list(producers)
        missing_wells = [w for w in (
            injectors_list + producers_list) if w not in well_coords]
        if missing_wells:
            print(f"Warning: Missing coordinates for wells: {missing_wells}")
            for w in missing_wells:
                well_coords[w] = (0, 0)
        cmap_obj = plt.cm.get_cmap(cmap) if cmap else None
        if vmax is None:
            if isinstance(f_matrix, np.ndarray):
                vmax = np.max(f_matrix)
            else:
                vmax = f_matrix.max().max()
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = plt.Normalize(vmin=0, vmax=vmax)
        plotted_injector_label = False
        plotted_producer_label = False
        for well in injectors_list + producers_list:
            coords = well_coords.get(well, (0, 0))
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                print(
                    f"Warning: Invalid coordinates for well {well}: {coords}, using (0, 0)")
                x, y = 0, 0
            else:
                x, y = coords
                if not (np.isfinite(x) and np.isfinite(y)):
                    print(
                        f"Warning: Non-finite coordinates for well {well}: ({x}, {y}), using (0, 0)")
                    x, y = 0, 0
            if well in injectors_list:
                ax.scatter(x, y, c="blue", s=80, marker="s",
                           label="Injector" if not plotted_injector_label else "")
                ax.text(x, y,self.get_well_number(well), fontsize=8, color="blue", ha="right", va="top")
                plotted_injector_label = True
            elif well in producers_list:
                ax.scatter(x, y, c="black", s=50, marker="o",
                           label="Producer" if not plotted_producer_label else "")
                ax.text(x, y, self.get_well_number(well),
                        fontsize=8, color="black", ha="left", va="top")
                plotted_producer_label = True
        for j, prod in enumerate(producers_list):
            coords_p = well_coords.get(prod, (0, 0))
            if not isinstance(coords_p, (tuple, list)) or len(coords_p) != 2:
                print(
                    f"Warning: Skipping producer {prod} due to invalid coordinates: {coords_p}")
                continue
            xp, yp = coords_p
            if not (np.isfinite(xp) and np.isfinite(yp)):
                print(
                    f"Warning: Skipping producer {prod} due to non-finite coordinates: ({xp}, {yp})")
                continue
            for i, inj in enumerate(injectors_list):
                coords_i = well_coords.get(inj, (0, 0))
                if not isinstance(coords_i, (tuple, list)) or len(coords_i) != 2:
                    print(
                        f"Warning: Skipping injector {inj} due to invalid coordinates: {coords_i}")
                    continue
                xi, yi = coords_i
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    print(
                        f"Warning: Skipping injector {inj} due to non-finite coordinates: ({xi}, {yi})")
                    continue
                f_val = f_matrix[j, i] if isinstance(
                    f_matrix, np.ndarray) else f_matrix.loc[inj, prod]
                if not np.isfinite(f_val):
                    print(
                        f"Warning: Invalid f_ij value {f_val} for {inj} -> {prod}, skipping")
                    continue
                if (show_all_arrows or (f_val > 0.01 and f_val > 0)) and connectivity_mask[j, i]:
                    dx, dy = xp - xi, yp - yi
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 1e-6:
                        continue
                    frac = np.clip(f_val, 0, 1)
                    x_end, y_end = xi + dx * frac, yi + dy * frac
                    head_length = max(4, arrow_width * 1.5)
                    head_width = max(2, arrow_width * 1.0)
                    color = cmap_obj(norm(f_val)) if cmap else 'teal'
                    arrow = FancyArrowPatch(
                        (xi, yi), (x_end, y_end),
                        arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                        linewidth=arrow_width, mutation_scale=1, color=color, alpha=0.8
                    )
                    ax.add_patch(arrow)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.legend()
        if cmap:
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            cbar = plt.colorbar(sm, ax=ax, label='f_ij')
            cbar.set_label(f'f_ij (max: {vmax:.3f})')

    def plot_connectivity_heatmap(self, method, timestep=None, cmap="viridis", annot=True, figsize=(10, 8)):
        if method not in self.comparison_params:
            raise AttributeError(
                f"Method {method} not found in comparison_params. Run run_comparison_models() first.")
        has_aquifer = self.aquifer_index is not None
        real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
        if method in ["static_noaq", "static_withaq"]:
            f_matrix = np.zeros((self.n_prod, min(
                real_n_inj, self.comparison_params[method].get("n_inj", self.n_inj))))
            params = self.comparison_params[method]
            for j, prod in enumerate(self.prod_cols):
                if not params.get(j):
                    continue
                for i in range(f_matrix.shape[1]):
                    f_matrix[j, i] = params[j]["f"][i]
            title_suffix = ""
            inj_cols_heatmap = list(self.inj_cols[:f_matrix.shape[1]])
            f_df = pd.DataFrame(
                f_matrix, index=self.prod_cols, columns=inj_cols_heatmap)
            vmax = f_df.max().max()
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            plt.figure(figsize=figsize)
            sns.heatmap(f_df, cmap=cmap, annot=annot, fmt=".2f", vmin=0,
                        vmax=vmax, cbar_kws={'label': f'f_ij (max: {vmax:.3f})'})
            plt.title(
                f"Injector-Producer Connectivity Heatmap ({method}){title_suffix}")
            plt.xlabel("Injectors")
            plt.ylabel("Producers")
            plt.tight_layout()
            plt.show()
        else:
            n_inj = min(real_n_inj, self.comparison_params[method]["n_inj"])
            if timestep is None:
                f_matrix = np.mean(
                    self.comparison_params[method]["f_time"][:, :, :n_inj], axis=0)
                title_suffix = " (Averaged Over Time)"
                inj_cols_heatmap = list(self.inj_cols[:n_inj])
                f_df = pd.DataFrame(
                    f_matrix, index=self.prod_cols, columns=inj_cols_heatmap)
                vmax = f_df.max().max()
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                plt.figure(figsize=figsize)
                sns.heatmap(f_df, cmap=cmap, annot=annot, fmt=".2f", vmin=0,
                            vmax=vmax, cbar_kws={'label': f'f_ij (max: {vmax:.3f})'})
                plt.title(
                    f"Injector-Producer Connectivity Heatmap ({method}){title_suffix}")
                plt.xlabel("Injectors")
                plt.ylabel("Producers")
                plt.tight_layout()
                plt.show()
            else:
                timesteps = [timestep] if isinstance(
                    timestep, int) else timestep
                for t in timesteps:
                    if not (0 <= t < self.T):
                        raise ValueError(
                            f"Timestep {t} must be between 0 and {self.T-1}")
                    f_matrix = self.comparison_params[method]["f_time"][t, :, :n_inj].copy(
                    )
                    active_mask = self.I[t, :n_inj] > 0
                    f_matrix[:, ~active_mask] = 0.0
                    title_suffix = f" - Day {self.days[t]}"
                    inj_cols_heatmap = list(self.inj_cols[:n_inj])
                    f_df = pd.DataFrame(
                        f_matrix, index=self.prod_cols, columns=inj_cols_heatmap)
                    vmax = f_df.max().max()
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = 1.0
                    plt.figure(figsize=figsize)
                    sns.heatmap(f_df, cmap=cmap, annot=annot, fmt=".2f", vmin=0, vmax=vmax, cbar_kws={
                                'label': f'f_ij (max: {vmax:.3f})'})
                    plt.title(
                        f"Injector-Producer Connectivity Heatmap ({method}){title_suffix}")
                    plt.xlabel("Injectors")
                    plt.ylabel("Producers")
                    plt.tight_layout()
                    plt.show()

    def export_connectivity_clean(self, method, well_coords, arrow_width=1.0, cmap='viridis', show_all_arrows=False, filename=None, dpi=300):
        if method not in self.comparison_params:
            raise AttributeError(
                f"Method {method} not found in comparison_params. Run run_comparison_models() first.")
        params = self.comparison_params[method]
        injectors = list(self.inj_cols)
        producers = list(self.prod_cols)
        n_inj = params.get("n_inj", len(injectors))
        f_matrix = pd.DataFrame(
            index=injectors[:n_inj], columns=producers, dtype=float)
        for j, prod in enumerate(producers):
            if not params.get(j):
                continue
            for i, inj in enumerate(injectors[:n_inj]):
                f_matrix.loc[inj, prod] = params[j]["f"][i]
        f_max = f_matrix.max().max()
        if not np.isfinite(f_max) or f_max <= 0:
            f_max = 1.0
        fig = plt.figure(figsize=(10, 8), facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('none')
        ax.axis('off')
        plt.title(f"Injector-Producer Connectivity ({method})", pad=20)
        self._plot_connectivity_base_clean(
            ax, f_matrix, well_coords, injectors[:n_inj], producers,
            self.connectivity_mask[:,
                                   :n_inj], arrow_width, cmap, show_all_arrows, vmax=f_max
        )
        if filename is None:
            filename = f"connectivity_{method}_clean.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1,
                    transparent=True, facecolor='none', edgecolor='none')
        plt.close()
        print(f"Clean connectivity plot saved as: {filename}")

    def export_connectivity_over_time_clean(self, method, well_coords, timesteps=None, arrow_width=1.0, cmap='viridis', show_all_arrows=False, filename_prefix=None, dpi=300):
        if method not in ["dynamic_noaq", "dynamic_withaq"]:
            raise ValueError(
                "Method must be 'dynamic_noaq' or 'dynamic_withaq' for time-varying connectivity.")
        if not hasattr(self, "comparison_params") or method not in self.comparison_params:
            raise AttributeError(
                f"Run run_comparison_models() first to generate {method} parameters.")
        f_time = self.comparison_params[method]["f_time"]
        has_aquifer = self.aquifer_index is not None
        real_n_inj = self.aquifer_index if has_aquifer else self.n_inj
        n_inj = min(real_n_inj, self.comparison_params[method]["n_inj"])
        if f_time.shape[2] > n_inj:
            print(
                f"Warning: Truncating f_time from {f_time.shape[2]} to {n_inj} injectors to exclude aquifer")
            f_time = f_time[:, :, :n_inj]
        if f_time.shape[1] != len(self.prod_cols) or f_time.shape[2] != n_inj:
            raise ValueError(
                f"f_time shape {f_time.shape} does not match expected (*, {len(self.prod_cols)}, {n_inj})")
        if self.connectivity_mask.shape[1] < n_inj:
            raise ValueError(
                f"connectivity_mask has fewer injectors ({self.connectivity_mask.shape[1]}) than required ({n_inj})")
        connectivity_mask = self.connectivity_mask[:, :n_inj]
        if isinstance(well_coords, pd.DataFrame):
            well_coords = dict(zip(well_coords["Well"], zip(
                well_coords["X"], well_coords["Y"])))
        inj_cols_list = list(self.inj_cols[:n_inj])
        prod_cols_list = list(self.prod_cols)
        for well in inj_cols_list + prod_cols_list:
            coords = well_coords.get(well, (0, 0))
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                well_coords[well] = (0, 0)
        timesteps = timesteps or [0, len(self.days)//2, len(self.days)-1]
        for t in timesteps:
            if not (0 <= t < self.T):
                raise ValueError(
                    f"Timestep {t} must be between 0 and {self.T-1}")
            f_matrix = f_time[t, :, :].copy()
            active_mask = self.I[t, :n_inj] > 0
            f_matrix[:, ~active_mask] = 0.0
            combined_mask = connectivity_mask & active_mask
            f_max = f_matrix.max()
            if not np.isfinite(f_max) or f_max <= 0:
                f_max = 1.0
            fig = plt.figure(figsize=(10, 8), facecolor='none')
            ax = plt.gca()
            ax.set_facecolor('none')
            ax.axis('off')
            plt.title(
                f"Injector-Producer Connectivity ({method}) - Day {self.days[t]}", pad=20)
            self._plot_connectivity_base_clean(
                ax, f_matrix, well_coords, inj_cols_list, prod_cols_list,
                combined_mask, arrow_width, cmap, show_all_arrows, vmax=f_max
            )
            if filename_prefix is None:
                filename = f"connectivity_{method}_t{t}_clean.png"
            else:
                filename = f"{filename_prefix}_t{t}_clean.png"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1,
                        transparent=True, facecolor='none', edgecolor='none')
            plt.close()
            print(f"Clean connectivity plot saved as: {filename}")

    def _plot_connectivity_base_clean(self, ax, f_matrix, well_coords, injectors, producers, connectivity_mask, arrow_width=1.0, cmap='viridis', show_all_arrows=False, vmax=None):
        if isinstance(f_matrix, np.ndarray):
            if f_matrix.shape != (len(producers), len(injectors)):
                raise ValueError(
                    f"f_matrix shape {f_matrix.shape} does not match expected ({len(producers)}, {len(injectors)})")
        elif not isinstance(f_matrix, pd.DataFrame):
            raise ValueError(
                "f_matrix must be a numpy array or pandas DataFrame")
        if connectivity_mask.shape != (len(producers), len(injectors)):
            raise ValueError(
                f"connectivity_mask shape {connectivity_mask.shape} does not match expected ({len(producers)}, {len(injectors)})")
        if isinstance(well_coords, pd.DataFrame):
            well_coords = dict(zip(well_coords["Well"], zip(
                well_coords["X"], well_coords["Y"])))
        injectors_list = list(injectors)
        producers_list = list(producers)
        missing_wells = [w for w in (
            injectors_list + producers_list) if w not in well_coords]
        if missing_wells:
            for w in missing_wells:
                well_coords[w] = (0, 0)
        cmap_obj = plt.cm.get_cmap(cmap) if cmap else None
        if vmax is None:
            if isinstance(f_matrix, np.ndarray):
                vmax = np.max(f_matrix)
            else:
                vmax = f_matrix.max().max()
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = plt.Normalize(vmin=0, vmax=vmax)
        for well in injectors_list + producers_list:
            coords = well_coords.get(well, (0, 0))
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                x, y = 0, 0
            else:
                x, y = coords
                if not (np.isfinite(x) and np.isfinite(y)):
                    x, y = 0, 0
            if well in injectors_list:
                ax.scatter(x, y, c="blue", s=80, marker="s")
            elif well in producers_list:
                ax.scatter(x, y, c="black", s=50, marker="o")
        for j, prod in enumerate(producers_list):
            coords_p = well_coords.get(prod, (0, 0))
            if not isinstance(coords_p, (tuple, list)) or len(coords_p) != 2:
                continue
            xp, yp = coords_p
            if not (np.isfinite(xp) and np.isfinite(yp)):
                continue
            for i, inj in enumerate(injectors_list):
                coords_i = well_coords.get(inj, (0, 0))
                if not isinstance(coords_i, (tuple, list)) or len(coords_i) != 2:
                    continue
                xi, yi = coords_i
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    continue
                f_val = f_matrix[j, i] if isinstance(
                    f_matrix, np.ndarray) else f_matrix.loc[inj, prod]
                if not np.isfinite(f_val):
                    continue
                if (show_all_arrows or (f_val > 0.01 and f_val > 0)) and connectivity_mask[j, i]:
                    dx, dy = xp - xi, yp - yi
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 1e-6:
                        continue
                    frac = np.clip(f_val, 0, 1)
                    x_end, y_end = xi + dx * frac, yi + dy * frac
                    head_length = max(4, arrow_width * 1.5)
                    head_width = max(2, arrow_width * 1.0)
                    color = cmap_obj(norm(f_val)) if cmap else 'teal'
                    arrow = FancyArrowPatch(
                        (xi, yi), (x_end, y_end),
                        arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                        linewidth=arrow_width, mutation_scale=1, color=color, alpha=0.8
                    )
                    ax.add_patch(arrow)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.axis("equal")
        if cmap:
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            cbar = plt.colorbar(sm, ax=ax, label='f_ij')
            cbar.set_label(f'f_ij (max: {vmax:.3f})')