#!/usr/bin/env python

"""rtplots.py: Calculate all the functions of utility plots"""

__author__ = "Chakraborty, S."
__copyright__ = "Chakraborty, S."
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "chakras4@erau.edu"
__status__ = "Research"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def setup(
    size=15,
    style_names=("science", "grid", "no-latex"),
    fallback_style="default",
    font_family="sans-serif",
    sans_serif_fonts=("Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"),
    text_usetex=False,
    figure_dpi=150,
    savefig_dpi=300,
    axes_linewidth=0.9,
    lines_linewidth=1.6,
    axes_grid=False,
    grid_alpha=0.0,
    grid_linestyle="-",
    grid_linewidth=0.0,
    axes_grid_which="major",
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style_api = getattr(plt, "style", None)
    can_use_style = style_api is not None and hasattr(style_api, "use")

    # Prefer SciencePlots style when available, but do not hard-fail.
    try:
        import scienceplots  # noqa: F401

        if can_use_style:
            style_api.use(list(style_names))
    except Exception:
        if can_use_style:
            style_api.use(fallback_style)

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.sans-serif"] = list(sans_serif_fonts)
    plt.rcParams["text.usetex"] = text_usetex
    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams["savefig.dpi"] = savefig_dpi
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["lines.linewidth"] = lines_linewidth
    plt.rcParams["axes.grid"] = axes_grid
    plt.rcParams["grid.alpha"] = grid_alpha
    plt.rcParams["grid.linestyle"] = grid_linestyle
    plt.rcParams["grid.linewidth"] = grid_linewidth
    plt.rcParams["axes.grid.which"] = axes_grid_which
    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
    logger.debug("plottrace.setup applied with size={}", size)
    return


class PlotRays(object):
    def __init__(
        self,
        nrows=1,
        ncols=1,
        ylim=[],
        xlim=[],
        oth=True,
        figsize=(5, 5),
        Re_km=6371.0,
        font_size=15,
        ylabel_loc=(-200, 200),
        xlabel_loc=(500, -50),
        figure_dpi=300,
        savefig_facecolor=(1, 1, 1, 1),
        savefig_bbox="tight",
        style_kwargs=None,
        default_xlim=(-300, 300),
        default_ylim=(-100, 800),
        default_yticks=(0, 200, 400, 600, 800),
        arc_samples=181,
        arc_line_color="k",
        arc_line_width=1.0,
        axis_facecolor="0.98",
        ground_fill_color="gray",
        ground_fill_alpha=0.5,
        ground_fill_bottom_km=-800.0,
        label_fontweight="bold",
        curved_tick_target=7,
        curved_tick_length_km=12.0,
        curved_tick_label_gap_km=18.0,
        curved_tick_linewidth=0.8,
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.xlim = xlim
        self.ylim = ylim
        self.axnum = 0
        self.fig = plt.figure(
            figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=figure_dpi
        )
        self.oth = oth
        self.Re = Re_km
        self.font_size = font_size
        self.xlabel_loc = xlabel_loc
        self.ylabel_loc = ylabel_loc
        self.savefig_facecolor = savefig_facecolor
        self.savefig_bbox = savefig_bbox
        self.default_xlim = list(default_xlim)
        self.default_ylim = list(default_ylim)
        self.default_yticks = list(default_yticks)
        self.arc_samples = int(arc_samples)
        self.arc_line_color = arc_line_color
        self.arc_line_width = float(arc_line_width)
        self.axis_facecolor = axis_facecolor
        self.ground_fill_color = ground_fill_color
        self.ground_fill_alpha = float(ground_fill_alpha)
        self.ground_fill_bottom_km = float(ground_fill_bottom_km)
        self.label_fontweight = label_fontweight
        self.curved_tick_target = int(curved_tick_target)
        self.curved_tick_length_km = float(curved_tick_length_km)
        self.curved_tick_label_gap_km = float(curved_tick_label_gap_km)
        self.curved_tick_linewidth = float(curved_tick_linewidth)
        setup(font_size, **(style_kwargs or {}))
        logger.info(
            "PlotRays initialized: nrows={}, ncols={}, figsize={}",
            nrows,
            ncols,
            figsize,
        )
        return

    @staticmethod
    def _nice_tick_step(span_km: float, target_ticks: int = 7) -> float:
        span = max(float(span_km), 1e-9)
        raw = span / max(int(target_ticks), 2)
        p10 = 10.0 ** np.floor(np.log10(raw))
        for m in (1.0, 2.0, 5.0, 10.0):
            step = m * p10
            if step >= raw:
                return float(step)
        return float(10.0 * p10)

    def _ground_tick_values(self, xmin: float, xmax: float, target_ticks: int = 7):
        step = self._nice_tick_step(xmax - xmin, target_ticks=target_ticks)
        t0 = np.ceil(xmin / step) * step
        t1 = np.floor(xmax / step) * step
        ticks = np.arange(t0, t1 + 0.5 * step, step, dtype=float)
        if xmin <= 0.0 <= xmax and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))
        return ticks

    @staticmethod
    def _format_tick_label(value: float, step: float) -> str:
        # Choose decimals from spacing; supports degree-like axes naturally.
        s = abs(float(step))
        if s >= 1.0:
            dec = 0
        elif s >= 0.1:
            dec = 1
        elif s >= 0.01:
            dec = 2
        elif s >= 0.001:
            dec = 3
        else:
            dec = 4
        v = float(value)
        if abs(v) < 0.5 * (10.0 ** (-dec)):
            v = 0.0
        return f"{v:.{dec}f}"

    def _draw_curved_ground_ticks(self, ax, xmin: float, xmax: float):
        ticks = self._ground_tick_values(
            xmin=xmin, xmax=xmax, target_ticks=self.curved_tick_target
        )
        if ticks.size == 0:
            return

        # Hide default straight x-axis ticks/labels when arc-ticks are used.
        if hasattr(ax, "set_xticks"):
            ax.set_xticks([])
        if hasattr(ax, "set_xticklabels"):
            ax.set_xticklabels([])

        span = max(abs(float(xmax) - float(xmin)), 1.0)
        eps = max(1e-3, 1e-4 * span)
        tick_len = self.curved_tick_length_km
        label_gap = self.curved_tick_label_gap_km

        for xt in ticks:
            yg = float(self.get_arc_heights(0.0, float(xt)))
            y1 = float(self.get_arc_heights(0.0, float(xt) - eps))
            y2 = float(self.get_arc_heights(0.0, float(xt) + eps))
            dydx = (y2 - y1) / (2.0 * eps)

            # Unit normal to the local tangent of y(x)
            nx, ny = -dydx, 1.0
            nn = max((nx * nx + ny * ny) ** 0.5, 1e-12)
            nx, ny = nx / nn, ny / nn

            dx = 0.5 * tick_len * nx
            dy = 0.5 * tick_len * ny
            ax.plot(
                [xt - dx, xt + dx],
                [yg - dy, yg + dy],
                color="k",
                lw=self.curved_tick_linewidth,
                zorder=5,
            )

            angle = float(np.rad2deg(np.arctan(dydx)))
            lx = xt - label_gap * nx
            ly = yg - label_gap * ny
            ax.text(
                lx,
                ly,
                f"{xt:.0f}",
                ha="center",
                va="top",
                fontsize=max(self.font_size - 2, 8),
                rotation=angle,
                rotation_mode="anchor",
                zorder=6,
            )

    def save(self, filepath):
        self.fig.savefig(
            filepath,
            bbox_inches=self.savefig_bbox,
            facecolor=self.savefig_facecolor,
        )
        logger.info("Plot saved: {}", filepath)
        return

    def close(self):
        self.fig.clf()
        plt.close()
        logger.debug("PlotRays figure closed")
        return

    def set_param_lims(
        self, pf_lim=(1, 9), edens_lim=(1e10, 1e12), ref_indx_lim=(0.8, 1.0)
    ):
        self.pf_lim = pf_lim
        self.edens_lim = edens_lim
        self.ref_indx_lim = ref_indx_lim
        return

    def get_parameter(self, kind):
        import matplotlib.colors as colors

        if kind == "pf":
            o, cmap, label, norm = (
                getattr(self, kind),
                "PuOr",
                r"$f_0$ [MHz]",
                colors.Normalize(self.pf_lim[0], self.pf_lim[1]),
            )
        if kind == "edens":
            o, cmap, label, norm = (
                getattr(self, kind),
                "cool",
                r"$N_e$ [$cm^{-3}$]",
                colors.LogNorm(self.edens_lim[0], self.edens_lim[1]),
            )
        if kind == "ref_indx":
            o, cmap, label, norm = (
                getattr(self, kind),
                "cool",
                r"$\eta$",
                colors.Normalize(self.ref_indx_lim[0], self.ref_indx_lim[1]),
            )
        return o, cmap, label, norm

    def get_arc_heights(self, height, dist):
        darc = dist / self.Re
        true_height = self.Re + height
        height = true_height * np.cos(darc) - self.Re
        return height

    def create_figure_pane(self, xlabel=r"Ground range, km", ylabel=r"Height, km"):
        self.axnum += 1
        fignum = 100 * self.nrows + 10 * self.ncols + self.axnum
        ax = self.fig.add_subplot(fignum)
        # Create Arc
        if self.oth:
            theta = np.deg2rad(np.linspace(-180, 180, self.arc_samples))
            x, y = self.Re * np.cos(theta), self.Re * np.sin(theta) - self.Re
            ax.plot(x, y, ls="-", color=self.arc_line_color, lw=self.arc_line_width)
            ax.text(
                self.ylabel_loc[0],
                self.ylabel_loc[1],
                ylabel,
                ha="left",
                va="center",
                fontdict={"size": self.font_size, "fontweight": self.label_fontweight},
                rotation=90,
            )
            ax.text(
                self.xlabel_loc[0],
                self.xlabel_loc[1],
                xlabel,
                ha="center",
                va="top",
                fontdict={"size": self.font_size, "fontweight": self.label_fontweight},
            )
            ax.set_facecolor(self.axis_facecolor)
            ax.fill_between(
                x,
                self.ground_fill_bottom_km * np.ones_like(y),
                y,
                color=self.ground_fill_color,
                alpha=self.ground_fill_alpha,
            )
        else:
            ax.set_ylabel(
                ylabel,
                fontdict={"size": self.font_size, "fontweight": self.label_fontweight},
            )
            ax.set_xlabel(
                xlabel,
                fontdict={"size": self.font_size, "fontweight": self.label_fontweight},
            )
        ax.set_xlim(self.xlim if len(self.xlim) == 2 else self.default_xlim)
        ax.set_ylim(self.ylim if len(self.ylim) == 2 else self.default_ylim)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=self.font_size)
        ax.set_yticks(self.default_yticks)
        if self.oth:
            xlim = self.xlim if len(self.xlim) == 2 else self.default_xlim
            self._draw_curved_ground_ticks(
                ax=ax, xmin=float(xlim[0]), xmax=float(xlim[1])
            )
        return ax

    def lay_rays(
        self,
        outputs=[],
        kind="edens",
        lcolor="k",
        lw=0.3,
        ls="-",
        param_alpha=1,
        tag_distance: float = -1,
        ax=None,
        xlabel=r"Ground range, km",
        ylabel=r"Height, km",
        date=None,
        stitle=None,
        text="(A)",
        ped_angles=[],
        add_cbar=True,
        param_zorder=2,
        ray_zorder=3,
        cbar_pad=0.025,
        cbar_width=0.015,
        cbar_height_scale=0.6,
        cbar_cmap="plasma",
    ):
        ax = ax if ax else self.create_figure_pane(xlabel, ylabel)
        o, cmap, label, norm = self.get_parameter(kind)

        im = ax.pcolormesh(
            self.X,
            self.Z,
            o,
            norm=norm,
            cmap=cmap,
            alpha=param_alpha,
            zorder=param_zorder,
        )
        ax.set_xlim(self.xlim if len(self.xlim) == 2 else self.default_xlim)
        ax.set_ylim(self.ylim if len(self.ylim) == 2 else self.default_ylim)
        ax.grid(False)
        if add_cbar:
            pos = ax.get_position()
            cpos = [
                pos.x1 + cbar_pad,
                pos.y0 + 0.05,
                cbar_width,
                pos.height * cbar_height_scale,
            ]
            cax = self.fig.add_axes(cpos)
            cbax = self.fig.colorbar(
                im, cax, spacing="uniform", orientation="vertical", cmap=cbar_cmap
            )
            _ = cbax.set_label(label, fontsize=self.font_size)
            cbax.ax.tick_params(axis="both", labelsize=self.font_size)

        for o in outputs:
            x_km, y_km = o.x_km, o.y_km
            if self.oth:
                y_km = self.get_arc_heights(y_km, x_km)
            col, width = lcolor, lw
            if o.el0_deg in ped_angles:
                col, width = "darkgreen", lw * 2
            ax.plot(x_km, y_km, c=col, zorder=ray_zorder, ls=ls, lw=width)
        if text:
            ax.text(0.05, 0.95, text, ha="left", va="top", transform=ax.transAxes)
        return ax

    def set_density(self, X, Z, Ne, pf=None):
        self.X, self.Z, self.edens = X, Z, Ne
        self.pf = pf
        if self.oth:
            self.Z = self.get_arc_heights(self.Z, self.X)
        logger.info(
            "PlotRays density set: grid_shape={}, pf_set={}",
            np.shape(Ne),
            pf is not None,
        )
        return


class PlotRays3D(object):
    """
    Two-panel 3D face viewer with 2D-style arc rendering.
    Left panel: side face (east-west, lon-derived range vs height)
    Right panel: front face (north-south, lat-derived range vs height)
    """

    def __init__(
        self,
        oth=True,
        figsize=(6.5, 4.5),
        Re_km=6371.0,
        font_size=13,
        pf_lim=(1, 9),
        edens_lim=(1e3, 1e6),
        ref_indx_lim=(0.8, 1.0),
        hide_negative_yticks=True,
        figure_dpi=300,
        style_kwargs=None,
        axis_facecolor="0.98",
        ground_fill_color="gray",
        ground_fill_alpha=0.5,
        ground_fill_bottom_km=-800.0,
        arc_line_color="k",
        arc_line_width=1.0,
        arc_samples=512,
        label_fontweight="bold",
        top_facecolor="0.98",
        top_aspect="auto",
        curved_tick_target=7,
        curved_tick_linewidth=0.8,
        curved_tick_length_scale=0.015,
        curved_tick_gap_scale=0.03,
        curved_tick_min_length_km=6.0,
        curved_tick_min_gap_km=10.0,
        ray_under_color="white",
        ray_over_color="k",
        ray_under_width_scale=2.2,
        ray_over_width=0.8,
        ray_alpha=0.95,
        savefig_facecolor=(1, 1, 1, 1),
    ):
        self.fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=figure_dpi)
        self.oth = oth
        self.Re = Re_km
        self.font_size = font_size
        self.pf_lim = pf_lim
        self.edens_lim = edens_lim
        self.ref_indx_lim = ref_indx_lim
        self.hide_negative_yticks = hide_negative_yticks
        self._base_figsize = tuple(figsize)
        self.axis_facecolor = axis_facecolor
        self.ground_fill_color = ground_fill_color
        self.ground_fill_alpha = float(ground_fill_alpha)
        self.ground_fill_bottom_km = float(ground_fill_bottom_km)
        self.arc_line_color = arc_line_color
        self.arc_line_width = float(arc_line_width)
        self.arc_samples = int(arc_samples)
        self.label_fontweight = label_fontweight
        self.top_facecolor = top_facecolor
        self.top_aspect = top_aspect
        self.curved_tick_target = int(curved_tick_target)
        self.curved_tick_linewidth = float(curved_tick_linewidth)
        self.curved_tick_length_scale = float(curved_tick_length_scale)
        self.curved_tick_gap_scale = float(curved_tick_gap_scale)
        self.curved_tick_min_length_km = float(curved_tick_min_length_km)
        self.curved_tick_min_gap_km = float(curved_tick_min_gap_km)
        self.ray_under_color = ray_under_color
        self.ray_over_color = ray_over_color
        self.ray_under_width_scale = float(ray_under_width_scale)
        self.ray_over_width = float(ray_over_width)
        self.ray_alpha = float(ray_alpha)
        self.savefig_facecolor = savefig_facecolor
        setup(font_size, **(style_kwargs or {}))
        return

    def set_param_lims(
        self, pf_lim=(1, 9), edens_lim=(1e3, 1e6), ref_indx_lim=(0.8, 1.0)
    ):
        self.pf_lim = pf_lim
        self.edens_lim = edens_lim
        self.ref_indx_lim = ref_indx_lim
        return

    @staticmethod
    def _nice_tick_step(span_km: float, target_ticks: int = 7) -> float:
        span = max(float(span_km), 1e-9)
        raw = span / max(int(target_ticks), 2)
        p10 = 10.0 ** np.floor(np.log10(raw))
        for m in (1.0, 2.0, 5.0, 10.0):
            step = m * p10
            if step >= raw:
                return float(step)
        return float(10.0 * p10)

    def _ground_tick_values(self, xmin: float, xmax: float, target_ticks: int = 7):
        step = self._nice_tick_step(xmax - xmin, target_ticks=target_ticks)
        t0 = np.ceil(xmin / step) * step
        t1 = np.floor(xmax / step) * step
        ticks = np.arange(t0, t1 + 0.5 * step, step, dtype=float)
        if xmin <= 0.0 <= xmax and not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))
        return ticks, float(step)

    @staticmethod
    def _format_tick_label(value: float, step: float) -> str:
        s = abs(float(step))
        if s >= 1.0:
            dec = 0
        elif s >= 0.1:
            dec = 1
        elif s >= 0.01:
            dec = 2
        elif s >= 0.001:
            dec = 3
        else:
            dec = 4
        v = float(value)
        if abs(v) < 0.5 * (10.0 ** (-dec)):
            v = 0.0
        return f"{v:.{dec}f}"

    def _draw_curved_ground_ticks(
        self,
        ax,
        xmin: float,
        xmax: float,
        x_scale_km: float,
        x_center: float,
    ):
        ticks, step = self._ground_tick_values(
            xmin=xmin, xmax=xmax, target_ticks=self.curved_tick_target
        )
        if ticks.size == 0:
            return
        if hasattr(ax, "set_xticks"):
            ax.set_xticks([])
        if hasattr(ax, "set_xticklabels"):
            ax.set_xticklabels([])

        yspan = (
            float(ax.get_ylim()[1] - ax.get_ylim()[0])
            if hasattr(ax, "get_ylim")
            else 800.0
        )
        tick_len_y = max(
            self.curved_tick_min_length_km, self.curved_tick_length_scale * yspan
        )
        label_gap_y = max(
            self.curved_tick_min_gap_km, self.curved_tick_gap_scale * yspan
        )
        s = float(x_scale_km)
        c = float(x_center)

        for xt in ticks:
            d0 = (float(xt) - c) * s
            yg = float(self.get_arc_heights(0.0, d0))
            # For degree axes (lon/lat), keep ticks/labels in pure y-offset space
            # to avoid unit-mixing that can explode figure bbox.
            ax.plot(
                [xt, xt],
                [yg - 0.5 * tick_len_y, yg + 0.5 * tick_len_y],
                color="k",
                lw=self.curved_tick_linewidth,
                zorder=5,
                clip_on=True,
            )
            ax.text(
                xt,
                yg - label_gap_y,
                self._format_tick_label(float(xt), step),
                ha="center",
                va="top",
                fontsize=max(self.font_size - 2, 8),
                zorder=6,
                clip_on=True,
            )

    def get_parameter(self, param_face, kind):
        import matplotlib.colors as colors

        if kind == "pf":
            cmap, label, norm = (
                "PuOr",
                r"$f_0$ [MHz]",
                colors.Normalize(self.pf_lim[0], self.pf_lim[1]),
            )
        elif kind == "ref_indx":
            cmap, label, norm = (
                "cool",
                r"$\eta$",
                colors.Normalize(self.ref_indx_lim[0], self.ref_indx_lim[1]),
            )
        else:
            cmap, label, norm = (
                "cool",
                r"$N_e$ [$cm^{-3}$]",
                colors.LogNorm(self.edens_lim[0], self.edens_lim[1]),
            )
        return param_face, cmap, label, norm

    def get_arc_heights(self, height, dist):
        darc = dist / self.Re
        true_height = self.Re + height
        return true_height * np.cos(darc) - self.Re

    def _create_axis(
        self,
        idx: int,
        xlabel: str,
        ylabel: str,
        xlim,
        ylim,
        x_scale_km=1.0,
        x_center=0.0,
        ncols=2,
    ):
        ax = self.fig.add_subplot(1, int(ncols), idx)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if self.oth:
            x = np.linspace(float(xlim[0]), float(xlim[1]), self.arc_samples)
            y = self.get_arc_heights(
                np.zeros_like(x),
                (x - float(x_center)) * float(x_scale_km),
            )
            ax.plot(x, y, ls="-", color=self.arc_line_color, lw=self.arc_line_width)
            ax.set_facecolor(self.axis_facecolor)
            ax.fill_between(
                x,
                self.ground_fill_bottom_km * np.ones_like(y),
                y,
                color=self.ground_fill_color,
                alpha=self.ground_fill_alpha,
            )
        ax.set_xlabel(
            xlabel,
            fontdict={"size": self.font_size, "fontweight": self.label_fontweight},
        )
        ax.set_ylabel(
            ylabel,
            fontdict={"size": self.font_size, "fontweight": self.label_fontweight},
        )
        ax.tick_params(axis="both", labelsize=self.font_size)
        ax.grid(False)
        if self.oth:
            self._draw_curved_ground_ticks(
                ax=ax,
                xmin=float(xlim[0]),
                xmax=float(xlim[1]),
                x_scale_km=float(x_scale_km),
                x_center=float(x_center),
            )
        if self.hide_negative_yticks:
            yt = ax.get_yticks()
            ax.set_yticks(yt[yt >= 0])
        return ax

    def _plot_face(
        self,
        ax,
        xvec,
        heights,
        param_face,
        title,
        kind,
        x_scale_km=1.0,
        x_center=0.0,
        curve_density=True,
    ):
        X, Z = np.meshgrid(xvec, heights)
        if self.oth and curve_density:
            Z = self.get_arc_heights(Z, (X - float(x_center)) * float(x_scale_km))
        p, cmap, _, norm = self.get_parameter(param_face, kind)
        im = ax.pcolormesh(
            X,
            Z,
            p,
            norm=norm,
            cmap=cmap,
            alpha=0.9,
            zorder=2,
            shading="auto",
        )
        ax.set_title(title, fontsize=self.font_size)
        return im

    def _plot_rays_on_face(
        self,
        ax,
        x_series,
        h_series,
        lw=0.8,
        x_scale_km=1.0,
        x_center=0.0,
        curve_rays=True,
    ):
        for x, h in zip(x_series, h_series):
            if x.size == 0 or h.size == 0:
                continue
            n = min(x.size, h.size)
            x = x[:n]
            h = h[:n]
            ok = np.isfinite(x) & np.isfinite(h)
            if not np.any(ok):
                continue
            y = (
                self.get_arc_heights(
                    h[ok], (x[ok] - float(x_center)) * float(x_scale_km)
                )
                if (self.oth and curve_rays)
                else h[ok]
            )
            ax.plot(
                x[ok],
                y,
                c=self.ray_under_color,
                lw=lw * self.ray_under_width_scale,
                zorder=8,
                alpha=self.ray_alpha,
            )
            ax.plot(
                x[ok],
                y,
                c=self.ray_over_color,
                lw=lw,
                zorder=9,
                alpha=self.ray_alpha,
            )

    def plot_faces(
        self,
        ne_side,
        ne_front,
        x_side,
        x_front,
        heights,
        ray_side_x,
        ray_front_x,
        ray_h,
        kind="edens",
        xlim_side=None,
        xlim_front=None,
        ylim=None,
        xlabel_side="Cross-range (km)",
        xlabel_front="Cross-range (km)",
        x_scale_side_km=1.0,
        x_scale_front_km=1.0,
        x_center_side=0.0,
        x_center_front=0.0,
        hide_right_xaxis=False,
        hide_right_yaxis=True,
        cbar_fraction=0.012,
        cbar_pad=0.012,
        cbar_shrink=0.72,
        curve_density=True,
        curve_rays=True,
        show_top_view=False,
        top_xlabel="Longitude (deg)",
        top_ylabel="Latitude (deg)",
        ray_top_x=None,
        ray_top_y=None,
        cbar_after_panel=2,
        panel_wspace=None,
    ):
        if ylim is None:
            ylim = [float(heights.min()) - 50.0, float(heights.max())]
        if xlim_side is None:
            xlim_side = [float(np.min(x_side)), float(np.max(x_side))]
        if xlim_front is None:
            xlim_front = [float(np.min(x_front)), float(np.max(x_front))]

        ncols = 3 if bool(show_top_view) else 2
        if hasattr(self.fig, "set_size_inches"):
            self.fig.set_size_inches(
                self._base_figsize[0] * ncols,
                self._base_figsize[1],
                forward=True,
            )
        if hasattr(self.fig, "subplots_adjust"):
            if panel_wspace is not None:
                self.fig.subplots_adjust(wspace=float(panel_wspace))
            elif ncols == 3:
                self.fig.subplots_adjust(wspace=0.28)

        ax0 = self._create_axis(
            idx=1,
            xlabel=xlabel_side,
            ylabel="Height (km)",
            xlim=xlim_side,
            ylim=ylim,
            x_scale_km=x_scale_side_km,
            x_center=x_center_side,
            ncols=ncols,
        )
        ax1 = self._create_axis(
            idx=2,
            xlabel=xlabel_front,
            ylabel="Height (km)",
            xlim=xlim_front,
            ylim=ylim,
            x_scale_km=x_scale_front_km,
            x_center=x_center_front,
            ncols=ncols,
        )
        if ncols == 3:
            ax2 = self.fig.add_subplot(1, 3, 3)
            ax2.set_xlabel(
                top_xlabel, fontdict={"size": self.font_size, "fontweight": "bold"}
            )
            ax2.set_ylabel(
                top_ylabel, fontdict={"size": self.font_size, "fontweight": "bold"}
            )
            ax2.tick_params(axis="both", labelsize=self.font_size)
            ax2.grid(False)
            ax2.set_facecolor(self.top_facecolor)
            if hasattr(ax2, "set_aspect"):
                ax2.set_aspect(self.top_aspect, adjustable="box")

        im = self._plot_face(
            ax0,
            x_side,
            heights,
            ne_side,
            "Side Face",
            kind=kind,
            x_scale_km=x_scale_side_km,
            x_center=x_center_side,
            curve_density=curve_density,
        )
        self._plot_face(
            ax1,
            x_front,
            heights,
            ne_front,
            "Front Face",
            kind=kind,
            x_scale_km=x_scale_front_km,
            x_center=x_center_front,
            curve_density=curve_density,
        )
        self._plot_rays_on_face(
            ax0,
            ray_side_x,
            ray_h,
            x_scale_km=x_scale_side_km,
            x_center=x_center_side,
            curve_rays=curve_rays,
        )
        self._plot_rays_on_face(
            ax1,
            ray_front_x,
            ray_h,
            x_scale_km=x_scale_front_km,
            x_center=x_center_front,
            curve_rays=curve_rays,
        )
        if hide_right_xaxis:
            ax1.set_xlabel("")
            ax1.tick_params(axis="x", labelbottom=False)
        if hide_right_yaxis:
            ax1.set_ylabel("")
            ax1.tick_params(axis="y", labelleft=False)

        if ncols == 3:
            top_x = ray_top_x if ray_top_x is not None else ray_side_x
            top_y = ray_top_y if ray_top_y is not None else ray_front_x
            tx_all, ty_all = [], []
            for tx, ty in zip(top_x, top_y):
                tx = np.asarray(tx, dtype=float).ravel()
                ty = np.asarray(ty, dtype=float).ravel()
                if tx.size == 0 or ty.size == 0:
                    continue
                n = min(tx.size, ty.size)
                tx = tx[:n]
                ty = ty[:n]
                ok = np.isfinite(tx) & np.isfinite(ty)
                if not np.any(ok):
                    continue
                tx_all.append(tx[ok])
                ty_all.append(ty[ok])
                ax2.plot(
                    tx[ok],
                    ty[ok],
                    c=self.ray_under_color,
                    lw=self.ray_over_width * self.ray_under_width_scale,
                    zorder=8,
                    alpha=self.ray_alpha,
                )
                ax2.plot(
                    tx[ok],
                    ty[ok],
                    c=self.ray_over_color,
                    lw=self.ray_over_width,
                    zorder=9,
                    alpha=self.ray_alpha,
                )
            if len(tx_all) > 0:
                xcat = np.concatenate(tx_all)
                ycat = np.concatenate(ty_all)
                if xcat.size > 0 and ycat.size > 0:
                    xpad = max(
                        0.1, 0.05 * max(1e-6, float(np.max(xcat) - np.min(xcat)))
                    )
                    ypad = max(
                        0.1, 0.05 * max(1e-6, float(np.max(ycat) - np.min(ycat)))
                    )
                    ax2.set_xlim(float(np.min(xcat)) - xpad, float(np.max(xcat)) + xpad)
                    ax2.set_ylim(float(np.min(ycat)) - ypad, float(np.max(ycat)) + ypad)
            ax2.set_title("Top View", fontsize=self.font_size)

        _, _, label, _ = self.get_parameter(ne_side, kind)
        cbar_axes = [ax0, ax1]
        if ncols == 3 and int(cbar_after_panel) >= 3:
            cbar_axes = [ax0, ax1, ax2]
        cbar = self.fig.colorbar(
            im,
            ax=cbar_axes,
            fraction=cbar_fraction,
            pad=cbar_pad,
            shrink=cbar_shrink,
            aspect=28,
        )
        cbar.set_label(label, fontsize=self.font_size)
        cbar.ax.tick_params(axis="both", labelsize=self.font_size)
        return

    def save(self, filepath):
        # Keep fixed canvas size for 3D face plots; tight bbox can explode if
        # any annotation extends outside axes.
        self.fig.savefig(filepath, facecolor=self.savefig_facecolor)
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return


class PlotRays3DRouteFaces(PlotRays3D):
    """
    Route-aligned 3D face viewer.

    Panels:
    - Left: along-track distance (km) vs height
    - Middle: bearing-spread (cross-track, km) vs height
    - Right: top view (along-track vs bearing-spread)
    """

    @staticmethod
    def _enu_from_latlon(lat, lon, origin_lat, origin_lon):
        km_per_deg_lat = 111.32
        km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(float(origin_lat))))
        east_km = (np.asarray(lon, dtype=float) - float(origin_lon)) * km_per_deg_lon
        north_km = (np.asarray(lat, dtype=float) - float(origin_lat)) * km_per_deg_lat
        return east_km, north_km

    @staticmethod
    def _along_cross_from_enu(east_km, north_km, bearing_deg):
        br = np.deg2rad(float(bearing_deg))
        along = east_km * np.sin(br) + north_km * np.cos(br)
        cross = east_km * np.cos(br) - north_km * np.sin(br)
        return along, cross

    @staticmethod
    def _latlon_from_along_cross(
        along_km, cross_km, origin_lat, origin_lon, bearing_deg
    ):
        br = np.deg2rad(float(bearing_deg))
        east_km = np.asarray(along_km, dtype=float) * np.sin(br) + np.asarray(
            cross_km, dtype=float
        ) * np.cos(br)
        north_km = np.asarray(along_km, dtype=float) * np.cos(br) - np.asarray(
            cross_km, dtype=float
        ) * np.sin(br)
        km_per_deg_lat = 111.32
        km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(float(origin_lat))))
        lat = float(origin_lat) + north_km / km_per_deg_lat
        lon = float(origin_lon) + east_km / km_per_deg_lon
        return lat, lon

    def _sample_face(
        self,
        ne_grid: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        heights: np.ndarray,
        x_axis: np.ndarray,
        x_kind: str,
        x_fixed: float,
        origin_lat: float,
        origin_lon: float,
        bearing_deg: float,
    ) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (
                np.asarray(lats, dtype=float),
                np.asarray(lons, dtype=float),
                np.asarray(heights, dtype=float),
            ),
            np.asarray(ne_grid, dtype=float),
            bounds_error=False,
            fill_value=np.nan,
        )
        h = np.asarray(heights, dtype=float)
        x = np.asarray(x_axis, dtype=float)
        X, H = np.meshgrid(x, h, indexing="xy")
        if x_kind == "along":
            along = X
            cross = np.full_like(X, float(x_fixed))
        else:
            along = np.full_like(X, float(x_fixed))
            cross = X
        lat, lon = self._latlon_from_along_cross(
            along_km=along,
            cross_km=cross,
            origin_lat=float(origin_lat),
            origin_lon=float(origin_lon),
            bearing_deg=float(bearing_deg),
        )
        pts = np.column_stack([lat.ravel(), lon.ravel(), H.ravel()])
        face = interp(pts).reshape(H.shape)
        return np.clip(face, 1.0, None)

    def plot_route_faces(
        self,
        ne_grid: np.ndarray,
        ray_path_data,
        lats: np.ndarray,
        lons: np.ndarray,
        heights: np.ndarray,
        origin_lat: float,
        origin_lon: float,
        bearing_deg: float,
        along_ref_km: float | None = None,
        kind: str = "edens",
    ):
        paths = ray_path_data if isinstance(ray_path_data, list) else [ray_path_data]
        ray_along, ray_cross, ray_h = [], [], []
        all_along = []
        all_cross = []

        for rp in paths:
            lat = np.asarray(getattr(rp, "lat", []), dtype=float).ravel()
            lon = np.asarray(getattr(rp, "lon", []), dtype=float).ravel()
            h = np.asarray(getattr(rp, "height", []), dtype=float).ravel()
            if lat.size == 0 or lon.size == 0 or h.size == 0:
                continue
            east, north = self._enu_from_latlon(
                lat=lat,
                lon=lon,
                origin_lat=float(origin_lat),
                origin_lon=float(origin_lon),
            )
            along, cross = self._along_cross_from_enu(
                east_km=east, north_km=north, bearing_deg=float(bearing_deg)
            )
            ok = np.isfinite(along) & np.isfinite(cross) & np.isfinite(h)
            if not np.any(ok):
                continue
            along = along[ok]
            cross = cross[ok]
            h = h[ok]
            ray_along.append(along)
            ray_cross.append(cross)
            ray_h.append(h)
            all_along.append(along)
            all_cross.append(cross)

        if len(all_along) == 0:
            raise ValueError("No valid ray path points to plot in route-aligned frame")

        along_all = np.concatenate(all_along)
        cross_all = np.concatenate(all_cross)
        along_min = float(np.nanmin(along_all))
        along_max = float(np.nanmax(along_all))
        cross_min = float(np.nanmin(cross_all))
        cross_max = float(np.nanmax(cross_all))
        pad_a = max(5.0, 0.05 * max(1e-6, along_max - along_min))
        pad_c = max(5.0, 0.08 * max(1e-6, cross_max - cross_min))
        along_axis = np.linspace(
            along_min - pad_a, along_max + pad_a, max(64, int(np.asarray(lons).size))
        )
        cross_axis = np.linspace(
            cross_min - pad_c, cross_max + pad_c, max(64, int(np.asarray(lats).size))
        )

        if along_ref_km is None:
            along_ref_km = float(np.nanpercentile(along_all, 70.0))

        ne_along = self._sample_face(
            ne_grid=ne_grid,
            lats=lats,
            lons=lons,
            heights=heights,
            x_axis=along_axis,
            x_kind="along",
            x_fixed=0.0,
            origin_lat=float(origin_lat),
            origin_lon=float(origin_lon),
            bearing_deg=float(bearing_deg),
        )
        ne_cross = self._sample_face(
            ne_grid=ne_grid,
            lats=lats,
            lons=lons,
            heights=heights,
            x_axis=cross_axis,
            x_kind="cross",
            x_fixed=float(along_ref_km),
            origin_lat=float(origin_lat),
            origin_lon=float(origin_lon),
            bearing_deg=float(bearing_deg),
        )

        self.plot_faces(
            ne_side=ne_along,
            ne_front=ne_cross,
            x_side=along_axis,
            x_front=cross_axis,
            heights=np.asarray(heights, dtype=float),
            ray_side_x=ray_along,
            ray_front_x=ray_cross,
            ray_h=ray_h,
            kind=kind,
            xlim_side=[float(np.min(along_axis)), float(np.max(along_axis))],
            xlim_front=[float(np.min(cross_axis)), float(np.max(cross_axis))],
            ylim=[-300.0, 600.0],
            xlabel_side="Along-track (km)",
            xlabel_front="Bearing spread (km)",
            x_scale_side_km=1.0,
            x_scale_front_km=1.0,
            x_center_side=0.0,
            x_center_front=0.0,
            curve_density=True,
            curve_rays=True,
            show_top_view=True,
            top_xlabel="Along-track (km)",
            top_ylabel="Bearing spread (km)",
            ray_top_x=ray_along,
            ray_top_y=ray_cross,
        )
        return


class MatlabGeoPlot3D(object):
    """
    MATLAB `geoplot3` ray plotting wrapper.
    Starts MATLAB Engine only when available and requested.
    """

    def __init__(self, eng=None):
        self.eng = eng
        self._owns_engine = False
        self.available = False
        self.reason = ""
        self._matlab = None
        self.has_display = False
        self.can_geoplot3 = False
        self.can_plot3 = False
        self.last_mode = None
        self.last_used_topography = False

        try:
            import matlab
            import matlab.engine

            self._matlab = matlab
            if self.eng is None:
                self.eng = matlab.engine.start_matlab()
                self._owns_engine = True
            self.eng.eval(
                """
                has_geoplot3 = (exist('geoplot3','builtin') > 0) || ...
                               (exist('geoplot3','file') > 0) || ...
                               (exist('geoplot3') > 0);
                has_geoglobe = (exist('geoglobe','builtin') > 0) || ...
                               (exist('geoglobe','file') > 0) || ...
                               (exist('geoglobe') > 0);
                has_map_toolbox = license('test','map_toolbox');
                has_display = usejava('desktop') && feature('ShowFigureWindows');
                """,
                nargout=0,
            )
            has_geoplot3 = bool(float(self.eng.workspace["has_geoplot3"]))
            has_geoglobe = bool(float(self.eng.workspace["has_geoglobe"]))
            has_map_toolbox = bool(float(self.eng.workspace["has_map_toolbox"]))
            self.has_display = bool(float(self.eng.workspace["has_display"]))
            self.can_geoplot3 = (
                has_geoplot3 and has_geoglobe and has_map_toolbox and self.has_display
            )
            self.can_plot3 = True  # base MATLAB plot3 fallback path
            self.available = self.can_geoplot3 or self.can_plot3
            if self.can_geoplot3:
                self.reason = ""
            elif not self.has_display:
                self.reason = (
                    "Display unavailable; using headless plot3(ECEF) fallback."
                )
            elif not has_map_toolbox:
                self.reason = "Mapping Toolbox unavailable; using plot3(ECEF) fallback."
            elif not has_geoglobe or not has_geoplot3:
                self.reason = (
                    "geoplot3/geoglobe unavailable; using plot3(ECEF) fallback."
                )
            else:
                self.reason = "Using plot3(ECEF) fallback."
        except Exception as exc:
            self.reason = f"MATLAB Engine unavailable: {exc}"
            self.available = False
            if self._owns_engine and self.eng is not None:
                try:
                    self.eng.quit()
                except Exception:
                    pass
                self.eng = None
                self._owns_engine = False

    def close(self):
        if self._owns_engine and self.eng is not None:
            try:
                self.eng.quit()
            finally:
                self.eng = None
                self._owns_engine = False
        return

    def _iter_path_vectors(self, ray_path_data):
        paths = ray_path_data if isinstance(ray_path_data, list) else [ray_path_data]
        for rp in paths:
            lat = np.asarray(getattr(rp, "lat", []), dtype=float).ravel()
            lon = np.asarray(getattr(rp, "lon", []), dtype=float).ravel()
            h_km = np.asarray(getattr(rp, "height", []), dtype=float).ravel()
            n = min(lat.size, lon.size, h_km.size)
            if n < 2:
                continue
            lat = lat[:n]
            lon = lon[:n]
            h_m = 1000.0 * h_km[:n]
            ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h_m)
            if np.count_nonzero(ok) < 2:
                continue
            yield lat[ok], lon[ok], h_m[ok]

    def plot_rays(
        self,
        ray_path_data,
        out_file: str | Path | None = None,
        title: str = "PHaRLAP 3D Rays (geoplot3)",
        line_width: float = 1.2,
        figure_visible: bool = False,
        basemap: str = "streets-light",
        zoom_to_rays: bool = False,
        zoom_pad_deg: float = 0.5,
        terrain_exaggeration: float = 8.0,
        cam_lat: float | None = None,
        cam_lon: float | None = None,
        cam_alt_m: float | None = None,
        cam_pitch_deg: float | None = -30.0,
        cam_heading_deg: float | None = 40.0,
    ):
        if not self.available or self.eng is None:
            raise RuntimeError(
                self.reason or "MATLAB geoplot3 plotting is unavailable."
            )

        path_vectors = list(self._iter_path_vectors(ray_path_data))
        if len(path_vectors) == 0:
            raise RuntimeError("No valid ray points available for 3D plotting.")

        self.eng.workspace["fig_visible"] = "on" if figure_visible else "off"
        self.eng.workspace["plot_title"] = str(title)
        self.eng.workspace["plot_basemap"] = str(basemap)
        self.eng.workspace["zoom_to_rays"] = bool(zoom_to_rays)
        self.eng.workspace["zoom_pad_deg"] = float(zoom_pad_deg)
        self.eng.workspace["terrain_exaggeration"] = float(terrain_exaggeration)
        lat_all = np.concatenate([v[0] for v in path_vectors])
        lon_all = np.concatenate([v[1] for v in path_vectors])
        h_all_m = np.concatenate([v[2] for v in path_vectors])
        self.eng.workspace["lat_min"] = float(np.nanmin(lat_all))
        self.eng.workspace["lat_max"] = float(np.nanmax(lat_all))
        self.eng.workspace["lon_min"] = float(np.nanmin(lon_all))
        self.eng.workspace["lon_max"] = float(np.nanmax(lon_all))
        # Camera defaults center on ray envelope if not explicitly provided.
        self.eng.workspace["cam_lat"] = (
            float(cam_lat) if cam_lat is not None else float(np.nanmean(lat_all))
        )
        self.eng.workspace["cam_lon"] = (
            float(cam_lon) if cam_lon is not None else float(np.nanmean(lon_all))
        )
        self.eng.workspace["cam_alt_m"] = (
            float(cam_alt_m)
            if cam_alt_m is not None
            else float(max(9e3, np.nanmax(h_all_m) + 4e6))
        )
        self.eng.workspace["cam_pitch_deg"] = float(cam_pitch_deg)
        self.eng.workspace["cam_heading_deg"] = float(cam_heading_deg)
        self.eng.workspace["lw"] = float(line_width)
        if self.can_geoplot3:
            self.last_mode = "geoplot3"
            self.last_used_topography = basemap.lower() in {
                "topographic",
                "topographic-alt",
            }
            self.eng.eval(
                """
                fig = uifigure('Color','w','Visible',fig_visible);
                gl = geoglobe(fig);
                try
                    geobasemap(gl, plot_basemap);
                catch
                    % Keep plotting even if basemap is unavailable.
                end
                """,
                nargout=0,
            )
            for i, (lat, lon, h_m) in enumerate(path_vectors):
                self.eng.workspace[f"lat_{i}"] = self._matlab.double(lat.tolist())
                self.eng.workspace[f"lon_{i}"] = self._matlab.double(lon.tolist())
                self.eng.workspace[f"h_{i}"] = self._matlab.double(h_m.tolist())
                self.eng.eval(
                    f"geoplot3(gl, lat_{i}, lon_{i}, h_{i}, 'k-', 'LineWidth', lw);",
                    nargout=0,
                )
            self.eng.eval(
                """
                title(gl, plot_title);
                gl.FontSize = 12;
                % Camera controls similar to direct MATLAB usage:
                % campos(g,lat,lon,alt); campitch(g,pitch); camheading(g,heading)
                try
                    campos(gl, cam_lat, cam_lon, cam_alt_m);
                    campitch(gl, cam_pitch_deg);
                    camheading(gl, cam_heading_deg);
                catch
                    % Keep default top-down if camera ops are unavailable.
                end
                if zoom_to_rays
                    try
                        geolimits(gl, [lat_min-zoom_pad_deg, lat_max+zoom_pad_deg], ...
                                     [lon_min-zoom_pad_deg, lon_max+zoom_pad_deg]);
                    catch
                        % Keep full extent if zooming fails on this release.
                    end
                end
                """,
                nargout=0,
            )
            if out_file is not None:
                out_path = Path(out_file).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                self.eng.workspace["out_file"] = str(out_path)
                self.eng.eval(
                    """
                    drawnow;
                    try
                        exportapp(fig, out_file);
                    catch
                        exportgraphics(fig, out_file, 'Resolution', 220);
                    end
                    """,
                    nargout=0,
                )
            self.eng.eval("close(fig);", nargout=0)
            return

        # Headless fallback: plot3 in ECEF with invisible figure.
        self.last_mode = "plot3_ecef"
        self.last_used_topography = False
        self.eng.eval(
            """
            fig = figure('Color','w','Visible','off');
            ax = axes(fig);
            hold(ax, 'on');
            grid(ax, 'on');
            xlabel(ax,'X (km)'); ylabel(ax,'Y (km)'); zlabel(ax,'Z (km)');
            axis(ax,'equal');
            view(ax, 35, 25);
            Re_km = 6371.0;
            try
                topo_loaded = false;
                topo_candidates = {
                    fullfile(matlabroot, 'toolbox', 'map', 'mapdata', 'topo.mat'), ...
                    fullfile(matlabroot, 'toolbox', 'local', 'topo.mat'), ...
                    'topo.mat'
                };
                for kk = 1:numel(topo_candidates)
                    ftopo = topo_candidates{kk};
                    if exist(ftopo, 'file') == 2
                        S = load(ftopo);
                        if isfield(S, 'topo')
                            topo = double(S.topo);  % meters
                            topo_loaded = true;
                            break;
                        end
                    end
                end
                if ~topo_loaded
                    error('topo.mat not found');
                end
                latv = linspace(90, -90, size(topo, 1));       % deg
                lonv = linspace(-180, 180, size(topo, 2));     % deg
                [LON, LAT] = meshgrid(lonv, latv);
                topo_km = (topo ./ 1000.0) .* terrain_exaggeration;
                R = Re_km + topo_km;
                Xs = R .* cosd(LAT) .* cosd(LON);
                Ys = R .* cosd(LAT) .* sind(LON);
                Zs = R .* sind(LAT);
                surf(ax, Xs, Ys, Zs, topo, 'EdgeColor','none', 'FaceAlpha', 1.0);
                colormap(ax, terrain(256));
                clim(ax, [-8000 8000]);
                used_topo = true;
            catch
                % Fallback if topo dataset is unavailable.
                [sx, sy, sz] = sphere(120);
                surf(ax, Re_km*sx, Re_km*sy, Re_km*sz, ...
                    'FaceColor',[0.86 0.90 0.98], 'EdgeColor','none', 'FaceAlpha',0.7);
                used_topo = false;
            end
            light(ax);
            camlight(ax,'headlight');
            title(ax, plot_title);
            """,
            nargout=0,
        )
        try:
            self.last_used_topography = bool(float(self.eng.workspace["used_topo"]))
        except Exception:
            self.last_used_topography = False
        xyz_all = []
        for i, (lat, lon, h_m) in enumerate(path_vectors):
            lat_r = np.deg2rad(lat)
            lon_r = np.deg2rad(lon)
            r_km = 6371.0 + (h_m / 1000.0)
            x = r_km * np.cos(lat_r) * np.cos(lon_r)
            y = r_km * np.cos(lat_r) * np.sin(lon_r)
            z = r_km * np.sin(lat_r)
            self.eng.workspace[f"x_{i}"] = self._matlab.double(x.tolist())
            self.eng.workspace[f"y_{i}"] = self._matlab.double(y.tolist())
            self.eng.workspace[f"z_{i}"] = self._matlab.double(z.tolist())
            xyz_all.append(np.c_[x, y, z])
            self.eng.eval(
                f"plot3(ax, x_{i}, y_{i}, z_{i}, 'k-', 'LineWidth', lw);",
                nargout=0,
            )
        if zoom_to_rays and len(xyz_all) > 0:
            xyz = np.vstack(xyz_all)
            xmin, ymin, zmin = np.nanmin(xyz, axis=0)
            xmax, ymax, zmax = np.nanmax(xyz, axis=0)
            dx = max(1.0, xmax - xmin)
            dy = max(1.0, ymax - ymin)
            dz = max(1.0, zmax - zmin)
            padx, pady, padz = 0.15 * dx, 0.15 * dy, 0.2 * dz
            self.eng.workspace["xlim_v"] = self._matlab.double(
                [float(xmin - padx), float(xmax + padx)]
            )
            self.eng.workspace["ylim_v"] = self._matlab.double(
                [float(ymin - pady), float(ymax + pady)]
            )
            self.eng.workspace["zlim_v"] = self._matlab.double(
                [float(zmin - padz), float(zmax + padz)]
            )
            self.eng.eval(
                "xlim(ax, xlim_v); ylim(ax, ylim_v); zlim(ax, zlim_v);",
                nargout=0,
            )
        if out_file is not None:
            out_path = Path(out_file).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.eng.workspace["out_file"] = str(out_path)
            self.eng.eval(
                "exportgraphics(fig, out_file, 'Resolution', 220);",
                nargout=0,
            )
        self.eng.eval("close(fig);", nargout=0)
        return
