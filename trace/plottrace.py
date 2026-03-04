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


def setup(size=15):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    plt.rcParams["text.usetex"] = False
    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
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
        xlabel_loc=(0, -50),
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.xlim = xlim
        self.ylim = ylim
        self.axnum = 0
        self.fig = plt.figure(figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=300)
        self.oth = oth
        self.Re = Re_km
        self.font_size = font_size
        self.xlabel_loc = xlabel_loc
        self.ylabel_loc = ylabel_loc
        setup(font_size)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
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
            theta = np.deg2rad(np.linspace(-180, 180, 181))
            x, y = self.Re * np.cos(theta), self.Re * np.sin(theta) - self.Re
            ax.plot(x, y, ls="-", color="k", lw=1)
            ax.text(
                self.ylabel_loc[0],
                self.ylabel_loc[1],
                ylabel,
                ha="left",
                va="center",
                fontdict={"size": self.font_size, "fontweight": "bold"},
                rotation=90,
            )
            ax.text(
                self.xlabel_loc[0],
                self.xlabel_loc[1],
                xlabel,
                ha="center",
                va="top",
                fontdict={"size": self.font_size, "fontweight": "bold"},
            )
            ax.set_facecolor("0.98")
            ax.fill_between(x, -800 * np.ones_like(y), y, color="gray", alpha=0.5)
        else:
            ax.set_ylabel(
                ylabel, fontdict={"size": self.font_size, "fontweight": "bold"}
            )
            ax.set_xlabel(
                xlabel, fontdict={"size": self.font_size, "fontweight": "bold"}
            )
        ax.set_xlim(self.xlim if len(self.xlim) == 2 else [-300, 300])
        ax.set_ylim(self.ylim if len(self.ylim) == 2 else [-100, 800])
        ax.tick_params(axis="both", labelsize=self.font_size)
        ax.set_yticks([0, 200, 400, 600, 800])
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
        ax.set_xlim(self.xlim if len(self.xlim) == 2 else [-300, 300])
        ax.set_ylim(self.ylim if len(self.ylim) == 2 else [-100, 800])
        if add_cbar:
            pos = ax.get_position()
            cpos = [
                pos.x1 + 0.025,
                pos.y0 + 0.05,
                0.015,
                pos.height * 0.6,
            ]
            cax = self.fig.add_axes(cpos)
            cbax = self.fig.colorbar(
                im, cax, spacing="uniform", orientation="vertical", cmap="plasma"
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
    ):
        self.fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=300)
        self.oth = oth
        self.Re = Re_km
        self.font_size = font_size
        self.pf_lim = pf_lim
        self.edens_lim = edens_lim
        self.ref_indx_lim = ref_indx_lim
        self.hide_negative_yticks = hide_negative_yticks
        setup(font_size)
        return

    def set_param_lims(
        self, pf_lim=(1, 9), edens_lim=(1e3, 1e6), ref_indx_lim=(0.8, 1.0)
    ):
        self.pf_lim = pf_lim
        self.edens_lim = edens_lim
        self.ref_indx_lim = ref_indx_lim
        return

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
    ):
        ax = self.fig.add_subplot(1, 2, idx)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if self.oth:
            x = np.linspace(float(xlim[0]), float(xlim[1]), 512)
            y = self.get_arc_heights(
                np.zeros_like(x),
                (x - float(x_center)) * float(x_scale_km),
            )
            ax.plot(x, y, ls="-", color="k", lw=1)
            ax.set_facecolor("0.98")
            ax.fill_between(x, -800 * np.ones_like(y), y, color="gray", alpha=0.5)
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size, "fontweight": "bold"})
        ax.set_ylabel(ylabel, fontdict={"size": self.font_size, "fontweight": "bold"})
        ax.tick_params(axis="both", labelsize=self.font_size)
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
    ):
        X, Z = np.meshgrid(xvec, heights)
        if self.oth:
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
        )
        ax.set_title(title, fontsize=self.font_size)
        return im

    def _plot_rays_on_face(
        self, ax, x_series, h_series, lw=0.8, x_scale_km=1.0, x_center=0.0
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
                if self.oth
                else h[ok]
            )
            ax.plot(x[ok], y, c="white", lw=lw * 2.2, zorder=8, alpha=0.95)
            ax.plot(x[ok], y, c="k", lw=lw, zorder=9, alpha=0.95)

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
    ):
        if ylim is None:
            ylim = [float(heights.min()) - 50.0, float(heights.max())]
        if xlim_side is None:
            xlim_side = [float(np.min(x_side)), float(np.max(x_side))]
        if xlim_front is None:
            xlim_front = [float(np.min(x_front)), float(np.max(x_front))]

        ax0 = self._create_axis(
            idx=1,
            xlabel=xlabel_side,
            ylabel="Height (km)",
            xlim=xlim_side,
            ylim=ylim,
            x_scale_km=x_scale_side_km,
            x_center=x_center_side,
        )
        ax1 = self._create_axis(
            idx=2,
            xlabel=xlabel_front,
            ylabel="Height (km)",
            xlim=xlim_front,
            ylim=ylim,
            x_scale_km=x_scale_front_km,
            x_center=x_center_front,
        )

        im = self._plot_face(
            ax0,
            x_side,
            heights,
            ne_side,
            "Side Face",
            kind=kind,
            x_scale_km=x_scale_side_km,
            x_center=x_center_side,
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
        )
        self._plot_rays_on_face(
            ax0,
            ray_side_x,
            ray_h,
            x_scale_km=x_scale_side_km,
            x_center=x_center_side,
        )
        self._plot_rays_on_face(
            ax1,
            ray_front_x,
            ray_h,
            x_scale_km=x_scale_front_km,
            x_center=x_center_front,
        )
        if hide_right_xaxis:
            ax1.set_xlabel("")
            ax1.tick_params(axis="x", labelbottom=False)
        if hide_right_yaxis:
            ax1.set_ylabel("")
            ax1.tick_params(axis="y", labelleft=False)

        _, _, label, _ = self.get_parameter(ne_side, kind)
        cbar = self.fig.colorbar(
            im,
            ax=[ax0, ax1],
            fraction=cbar_fraction,
            pad=cbar_pad,
            shrink=cbar_shrink,
            aspect=28,
        )
        cbar.set_label(label, fontsize=self.font_size)
        cbar.ax.tick_params(axis="both", labelsize=self.font_size)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
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
        self.eng.workspace["lat_min"] = float(np.nanmin(lat_all))
        self.eng.workspace["lat_max"] = float(np.nanmax(lat_all))
        self.eng.workspace["lon_min"] = float(np.nanmin(lon_all))
        self.eng.workspace["lon_max"] = float(np.nanmax(lon_all))
        self.eng.workspace["lw"] = float(line_width)
        if self.can_geoplot3:
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
                S = load('topo.mat');
                topo = double(S.topo);  % meters
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
            catch
                % Fallback if topo dataset is unavailable.
                [sx, sy, sz] = sphere(120);
                surf(ax, Re_km*sx, Re_km*sy, Re_km*sz, ...
                    'FaceColor',[0.86 0.90 0.98], 'EdgeColor','none', 'FaceAlpha',0.7);
            end
            light(ax);
            camlight(ax,'headlight');
            title(ax, plot_title);
            """,
            nargout=0,
        )
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
