"""Simple live plotting for measurements, controls, and state estimates."""

import numpy as np
import matplotlib.pyplot as plt

# Colours per group so subplots are visually grouped
_COLORS_MEAS = plt.cm.tab10.colors
_COLORS_STATE = plt.cm.Set2.colors
_COLORS_CTRL = plt.cm.Dark2.colors


class LivePlot:
    """Dynamic live plot — one subplot per observed variable / state / control.

    Each variable gets its own y-axis so vastly different scales don't interfere.
    Subplots are grouped: measurements first, then state estimates, then controls.
    All subplots share the same x-axis (time).

    Parameters
    ----------
    measurement_provider : Provider, optional
        Provider with measurement variables (from MqttBridge).
    control_provider : Provider, optional
        Provider with control variables (from MqttBridge).
    state_names : list[str], optional
        Names for the model states, e.g. ["X", "S", "V"].
    update_interval : float
        Minimum seconds between redraws (default 0.05).
    """

    def __init__(
        self,
        measurement_provider=None,
        control_provider=None,
        state_names=[],
        update_interval=0.05,
    ):
        self._meas_prov = measurement_provider
        self._ctrl_prov = control_provider
        self._state_names = state_names
        self._update_interval = update_interval

        # State estimate history (filled via update())
        self._state_times = []
        self._state_values = []  # list of arrays, one entry per call

        # Build ordered list of (key, label, group) for every subplot
        self._subplot_keys = []   # unique string key
        self._subplot_labels = [] # y-axis / title label
        self._subplot_groups = [] # "meas" | "state" | "ctrl"

        meas_names = self._meas_prov.variable_names if self._meas_prov else []
        ctrl_names = self._ctrl_prov.variable_names if self._ctrl_prov else []

        for name in meas_names:
            self._subplot_keys.append(f"meas_{name}")
            self._subplot_labels.append(name)
            self._subplot_groups.append("meas")

        for name in self._state_names:
            self._subplot_keys.append(f"state_{name}")
            self._subplot_labels.append(name)
            self._subplot_groups.append("state")

        for name in ctrl_names:
            self._subplot_keys.append(f"ctrl_{name}")
            self._subplot_labels.append(name)
            self._subplot_groups.append("ctrl")

        n_plots = max(len(self._subplot_keys), 1)

        # Grid layout: up to 3 columns, rows grow as needed
        import math
        ncols = min(n_plots, 3)
        nrows = math.ceil(n_plots / ncols)

        plt.ion()
        self._fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            sharex=True,
            squeeze=False,
        )
        # Flatten to 1-D list
        self._axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

        # Hide axes that are not needed (bottom-right padding cells)
        for ax in self._axes[n_plots:]:
            ax.set_visible(False)

        # Map key → axis and configure each subplot
        self._ax_map = {}
        group_colors = {"meas": _COLORS_MEAS, "state": _COLORS_STATE, "ctrl": _COLORS_CTRL}
        group_counter = {"meas": 0, "state": 0, "ctrl": 0}
        group_titles = {"meas": "Measurement", "state": "State Estimate (EnKF)", "ctrl": "Control"}

        for i, (key, label, group) in enumerate(
            zip(self._subplot_keys, self._subplot_labels, self._subplot_groups)
        ):
            ax = self._axes[i]
            self._ax_map[key] = ax
            palette = group_colors[group]
            color = palette[group_counter[group] % len(palette)]
            group_counter[group] += 1

            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"{group_titles[group]}: {label}", fontsize=9, pad=2)
            ax.tick_params(labelsize=8)
            ax._lp_color = color

        # Add "Time [s]" label to the bottom row of visible subplots
        for c in range(ncols):
            # find last visible ax in this column
            for r in range(nrows - 1, -1, -1):
                ax = axes[r][c]
                if ax.get_visible():
                    ax.set_xlabel("Time [s]", fontsize=9)
                    break

        # Line handles: key → Line2D
        self._lines = {}

        self._fig.tight_layout(h_pad=1.8, w_pad=2.0)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, state=None, t=None):
        """Refresh the plot with current data.

        Parameters
        ----------
        state : array-like, optional
            Current state estimate vector, e.g. [X, S, V].
        t : float, optional
            Current time (for state estimates).
        """
        if state is not None and t is not None:
            self._state_times.append(t)
            self._state_values.append(np.atleast_1d(state).copy())

        # --- Per-variable measurement subplots ---
        if self._meas_prov:
            for name in self._meas_prov.variable_names:
                key = f"meas_{name}"
                ax = self._ax_map.get(key)
                if ax is None:
                    continue
                var = self._meas_prov.get_variable(name)
                if var is None or len(var) == 0:
                    continue
                times = var.times
                values = var.values.flatten()
                if key in self._lines:
                    self._lines[key].set_data(times, values)
                else:
                    line, = ax.plot(times, values, '.-', color=ax._lp_color, markersize=3)
                    self._lines[key] = line
                self._rescale(ax)

        # --- Per-state estimate subplots ---
        if self._state_values:
            t_arr = np.array(self._state_times)
            s_arr = np.array(self._state_values)
            for i, name in enumerate(self._state_names):
                if i >= s_arr.shape[1]:
                    break
                key = f"state_{name}"
                ax = self._ax_map.get(key)
                if ax is None:
                    continue
                if key in self._lines:
                    self._lines[key].set_data(t_arr, s_arr[:, i])
                else:
                    line, = ax.plot(t_arr, s_arr[:, i], '-', color=ax._lp_color)
                    self._lines[key] = line
                self._rescale(ax)

        # --- Per-control subplots ---
        if self._ctrl_prov:
            for name in self._ctrl_prov.variable_names:
                key = f"ctrl_{name}"
                ax = self._ax_map.get(key)
                if ax is None:
                    continue
                var = self._ctrl_prov.get_variable(name)
                if var is None or len(var) == 0:
                    continue
                times = var.times
                values = var.values.flatten()
                if key in self._lines:
                    self._lines[key].set_data(times, values)
                else:
                    line, = ax.plot(times, values, '-', color=ax._lp_color)
                    self._lines[key] = line
                self._rescale(ax)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(self._update_interval)

    @staticmethod
    def _rescale(ax):
        """Rescale axis to fit current data."""
        ax.relim()
        ax.autoscale_view()

    def close(self):
        """Close the plot window."""
        plt.ioff()
        plt.close(self._fig)
