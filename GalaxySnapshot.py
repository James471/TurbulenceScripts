import os
import yt
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from prettytable import PrettyTable
from util import create_movie

class GalaxySnapshot:
    SCALE_HEIGHT_FACTOR = 50
    def __init__(self, path, outdir=None):
        self.path = path
        self.outdir = outdir
        if outdir is None:
            self.outdir = os.path.join(os.path.dirname(path), "info_plots")
        self.ds = yt.load(path, max_level_convention="ramses", max_level=100)
        self.ad = self.ds.all_data()
        os.makedirs(self.outdir, exist_ok=True)

    def do_everything(self, mass_fraction=0.9,
                      wnm_criteria=[f'obj["temperature_over_mu"] < {8e3/1.27}', f'obj["temperature_over_mu"] > {5e3/1.27}'],
                      cnm_criteria=[f'obj["temperature_over_mu"] < {2e3}', f'obj["temperature_over_mu"] > {1e2}'],
                      max_zoom=50, 
                      diff_zoom=2,
                      framerate=5,
                      no_com=False):
        dx_min = self.get_smallest_dx()
        max_level = self.get_max_level()
        self.get_total_gas_mass()
        self.get_total_particle_mass()
        self.get_total_star_mass()
        self.get_total_DM_mass()
        self.get_total_star_count()
        self.get_total_DM_count()
        self.get_particle_types()
        self.get_gas_COM()
        self.get_star_COM()
        self.get_star_gas_COM()
        self.get_gas_bulk_velocity()
        self.get_star_bulk_velocity()
        self.get_star_gas_bulk_velocity()
        self.create_domain_maps()
        if not no_com:
            self.create_COM_plots(max_zoom=max_zoom, diff_zoom=diff_zoom, framerate=framerate)
        R, _ = self.plot_gas_disc(mass_fraction)
        self.create_phase_plots(mass_fraction)
        wnm_info = self.plot_phase_projection(wnm_criteria, name="WNM")
        cnm_info = self.plot_phase_projection(cnm_criteria, name="CNM")

        table = PrettyTable()
        table.field_names = ["Quantity", "Value"]
        table.add_row(["Max Level", max_level])
        table.add_row(["Smallest dx [pc]", f"{dx_min.to('pc').value:.2e}"])
        table.add_row(["Gas Mass [Msun]", f"{self.gas_mass:.2e}"])
        table.add_row(["Particle Mass [Msun]", f"{self.particle_mass:.2e}"])
        table.add_row(["Star Mass [Msun]", f"{self.star_mass:.2e}"])
        table.add_row(["DM Mass [Msun]", f"{self.DM_mass:.2e}"])
        table.add_row(["Number of Stars", f"{self.n_star}"])
        table.add_row(["Number of DM Particles", f"{self.n_DM}"])
        table.add_row(["Star COM [kpc]", np.array2string(self.com_star.to('kpc').value, precision=2)])
        table.add_row(["Gas COM [kpc]", np.array2string(self.com_gas.to('kpc').value, precision=2)])
        table.add_row(["Star+Gas COM [kpc]", np.array2string(self.com_star_gas.to('kpc').value, precision=2)])
        table.add_row(["Star Bulk Velocity [km/s]", np.array2string(self.star_bulk_velocity.to('km/s').value, precision=2)])
        table.add_row(["Gas Bulk Velocity [km/s]", np.array2string(self.gas_bulk_velocity.to('km/s').value, precision=2)])
        table.add_row(["Star+Gas Bulk Velocity [km/s]", np.array2string(self.bulk_velocity.to('km/s').value, precision=2)])
        table.add_row(["Gas Disc Radius [kpc]", f"{R.to('kpc').value:.2f}"])
        table.add_row(["Scale Height [pc]", f"{self.z0.value:.2f} ± {self.sigma_z0.value:.2f}"])
        table.add_row(["Galaxy Volume [kpc^3]", f"{wnm_info['galaxy_volume'].to('kpc**3').value:.2f}"])
        table.add_row(["Number of Galaxy Cells", f"{wnm_info['galaxy_resolution_elements']}"])
        print(table)

        table_wnm = PrettyTable()
        table_wnm.field_names = ["WNM Quantity", "Value"]
        wnm_temp_min = float(wnm_criteria[1].split('>')[1].strip())
        wnm_temp_max = float(wnm_criteria[0].split('<')[1].strip())
        table_wnm.add_row(["WNM Temperature Range [K]", f"{wnm_temp_min:.0f} - {wnm_temp_max:.0f} (K)"])
        table_wnm.add_row(["Volume [kpc^3]", f"{wnm_info['WNM_volume'].to('kpc**3').value:.2f}"])
        table_wnm.add_row(["Number Cells", f"{wnm_info['WNM_resolution_elements']}"])
        table_wnm.add_row(["Fraction (Volume / Galaxy Volume)", f"{wnm_info['WNM_volume_fraction'].value:.4f}"])
        table_wnm.add_row(["Fraction (Resolution Elements / Galaxy Resolution Elements)", f"{wnm_info['WNM_resolution_fraction']:.4f}"])
        for level in range(self.get_max_level() + 1):
            cell_length = dx_min * (2 ** (max_level-level))
            table_wnm.add_row([f"Fraction of Cells at Level {level} (dx={cell_length.to('pc'):.2f})", f"{wnm_info['WNM_fraction_per_level'][level]:.4f}"])
        print(table_wnm)

        table_cnm = PrettyTable()
        table_cnm.field_names = ["CNM Quantity", "Value"]
        cnm_temp_min = float(cnm_criteria[1].split('>')[1].strip())
        cnm_temp_max = float(cnm_criteria[0].split('<')[1].strip())
        table_cnm.add_row(["CNM Temperature Range [K]", f"{cnm_temp_min:.0f} - {cnm_temp_max:.0f} (K)"])
        table_cnm.add_row(["Volume [kpc^3]", f"{cnm_info['CNM_volume'].to('kpc**3').value:.2f}"])
        table_cnm.add_row(["Number Cells", f"{cnm_info['CNM_resolution_elements']}"])
        table_cnm.add_row(["Fraction (Volume / Galaxy Volume)", f"{cnm_info['CNM_volume_fraction'].value:.4f}"])
        table_cnm.add_row(["Fraction (Resolution Elements / Galaxy Resolution Elements)", f"{cnm_info['CNM_resolution_fraction']:.4f}"])
        for level in range(self.get_max_level() + 1):
            cell_length = dx_min * (2 ** (max_level-level))
            table_cnm.add_row([f"Fraction of Cells at Level {level} (dx={cell_length.to('pc'):.2f})", f"{cnm_info['CNM_fraction_per_level'][level]:.4f}"])
        print(table_cnm)

        table_path = os.path.join(self.outdir, "galaxy_info_table.txt")
        table_wnm_path = os.path.join(self.outdir, "wnm_info_table.txt")
        table_cnm_path = os.path.join(self.outdir, "cnm_info_table.txt")
        with open(table_path, "w") as f:
            f.write(table.get_string())
        with open(table_wnm_path, "w") as f:
            f.write(table_wnm.get_string())
        with open(table_cnm_path, "w") as f:
            f.write(table_cnm.get_string())

    def get_max_level(self):
        if hasattr(self, "max_level"):
            return self.max_level
        self.max_level = self.ds.index.max_level
        return self.max_level
    
    def get_smallest_dx(self):
        if hasattr(self, "smallest_dx"):
            return self.smallest_dx
        self.smallest_dx = self.ds.index.get_smallest_dx()
        return self.smallest_dx

    def get_total_gas_mass(self):
        if hasattr(self, "gas_mass"):
            return self.gas_mass
        self.gas_mass = self.ad.quantities.total_quantity("cell_mass").in_units("Msun")
        return self.gas_mass

    def get_total_particle_mass(self):
        if hasattr(self, "particle_mass"):
            return self.particle_mass
        self.particle_mass = self.ad.quantities.total_quantity("particle_mass").in_units("Msun")
        return self.particle_mass

    def get_total_star_mass(self):
        if hasattr(self, "star_mass"):
            return self.star_mass
        self.star_mass = self.ad["star", "particle_mass"].sum().in_units("Msun")
        return self.star_mass

    def get_total_DM_mass(self):
        if hasattr(self, "DM_mass"):
            return self.DM_mass
        self.DM_mass = self.ad["DM", "particle_mass"].sum().in_units("Msun")
        return self.DM_mass

    def get_total_star_count(self):
        if hasattr(self, "n_star"):
            return self.n_star
        self.n_star = self.ad["star", "particle_mass"].size
        return self.n_star

    def get_total_DM_count(self):
        if hasattr(self, "n_DM"):
            return self.n_DM
        self.n_DM = self.ad["DM", "particle_mass"].size
        return self.n_DM

    def get_particle_types(self):
        if hasattr(self, "particle_types"):
            return self.particle_types
        self.particle_types = self.ds.particle_types
        return self.particle_types
    
    def get_gas_COM(self):
        if hasattr(self, "com_gas"):
            return self.com_gas
        self.com_gas = self.ad.quantities.center_of_mass(use_gas=True, use_particles=False).to("kpc")
        return self.com_gas

    def get_star_COM(self):
        if hasattr(self, "com_star"):
            return self.com_star
        self.com_star = self.ad.quantities.center_of_mass(use_gas=False, use_particles=True, particle_type="star").to("kpc")
        return self.com_star
    
    def get_star_gas_COM(self):
        if hasattr(self, "com_star_gas"):
            return self.com_star_gas
        self.com_star_gas = self.ad.quantities.center_of_mass(use_gas=True, use_particles=True, particle_type="star").to("kpc")
        return self.com_star_gas
    
    def get_gas_bulk_velocity(self):
        if hasattr(self, "gas_bulk_velocity"):
            return self.gas_bulk_velocity
        self.gas_bulk_velocity = self.ad.quantities.bulk_velocity(use_gas=True, use_particles=False).to("km/s")
        return self.gas_bulk_velocity
    
    def get_star_bulk_velocity(self):
        if hasattr(self, "star_bulk_velocity"):
            return self.star_bulk_velocity
        self.star_bulk_velocity = self.ad.quantities.bulk_velocity(use_gas=False, use_particles=True, particle_type="star").to("km/s")
        return self.star_bulk_velocity
    
    def get_star_gas_bulk_velocity(self):
        if hasattr(self, "bulk_velocity"):
            return self.bulk_velocity
        self.bulk_velocity = self.ad.quantities.bulk_velocity(use_gas=True, use_particles=True, particle_type="star").to("km/s")
        return self.bulk_velocity
    
    def get_scale_height(self, n_bins=64):
        if hasattr(self, "z0_positive"):
            return self.z0_positive, self.sigma_z0_positive, self.z0_negative, self.sigma_z0_negative

        def log_scale_height_model(z, rho0, z0):
            return np.log(rho0) - np.abs(z) / z0
        
        com = self.get_gas_COM()
        com_z = com[2]
        
        ad = self.ad.cut_region(
            [f"obj['z'].to('code_length') > {(com_z.to('code_length') - self.ds.quan(1, 'kpc').to('code_length')).value}",
             f"obj['z'].to('code_length') < {(com_z.to('code_length') + self.ds.quan(1, 'kpc').to('code_length')).value}"]
        )
        
        prof = yt.create_profile(ad, "z", ("gas", "density"), n_bins=n_bins, weight_field=("gas", "density"), logs={"z": False})
        z_cord = prof.x
        rho = prof.field_data["gas", "density"]

        z_mid = com_z
        vertical_distance = z_cord - z_mid
        nan_mask = np.isfinite((np.log(rho)))
        rho_clean = rho[nan_mask]
        vertical_distance_clean = vertical_distance[nan_mask]
        popt, cor = curve_fit(log_scale_height_model, vertical_distance_clean.to('kpc').value, np.log(rho_clean), p0=[rho_clean.max().value, 0.1])

        z0, sigma_z0 = popt[1], np.sqrt(cor[1][1])

        pos_sch = z0 + z_mid.to("kpc").value
        neg_sch = -z0 + z_mid.to("kpc").value

        pl.plot(z_cord.to("kpc").value, rho, label="Density")
        density_fit = np.exp(log_scale_height_model(vertical_distance.to('kpc').value, *popt))
        pl.plot(z_cord.to("kpc").value, density_fit, label="Fit", color="black", linestyle="--")
        pl.axvline(pos_sch, color="green", linestyle="--")
        pl.axvspan(pos_sch - sigma_z0, pos_sch + sigma_z0, color="green", alpha=0.2)
        pl.axvline(neg_sch, color="green", linestyle="--")
        pl.axvspan(neg_sch - sigma_z0, neg_sch + sigma_z0, color="green", alpha=0.2)
        pl.axvline(z_mid.to("kpc").value, color="red", linestyle="--", label="COM")
        pl.yscale("log")
        pl.xlabel("z (kpc)")
        pl.ylabel(r"Density $(g/cm^3)$")
        pl.title("Scale Height")
        pl.legend()
        pl.savefig(os.path.join(self.outdir, "scale_height.pdf"))
        pl.close()

        self.z0 = self.ds.quan(popt[1], "kpc").to('pc')
        self.sigma_z0 = self.ds.quan(sigma_z0, "pc")

        return self.z0, self.sigma_z0

    def annotate_COMs(self, plot):
        com_star = self.get_star_COM()
        com_gas  = self.get_gas_COM()
        com_star_gas = self.get_star_gas_COM()
        plot.annotate_marker(com_star, marker="x", coord_system='data', s=15)
        plot.annotate_marker(com_gas, marker="x", coord_system='data', s=15)
        plot.annotate_marker(com_star_gas, marker="x", coord_system='data', s=15)
        plot.annotate_text(com_star, "Star COM", coord_system='data', text_args={"color": "black", "size": 15})
        plot.annotate_text(com_gas, "Gas COM", coord_system='data', text_args={"color": "black", "size": 15})
        plot.annotate_text(com_star_gas, "Combined COM", coord_system='data', text_args={"color": "black", "size": 15})

    def create_gas_plot(self, dataset, proj_dir, savefile, zoom=1, annotate_COM=False, unit=None):
        if proj_dir not in ["x", "y", "z"]:
            raise NotImplementedError("Projection direction must be 'x', 'y', or 'z'")
        plot = yt.ProjectionPlot(self.ds, proj_dir, dataset, weight_field="ones")
        if unit is not None:
            plot.set_unit(dataset, unit)
        plot.zoom(zoom)
        if annotate_COM:
            self.annotate_COMs(plot)
        plot.save(savefile)
        return plot

    def create_star_plot(self, dir_a, dir_b, savefile, zoom=1, annotate_COM=False, weight_field=None, cbar_label=None):
        plot = yt.ParticlePlot(self.ds, ("star", f"particle_position_{dir_a}"), f"particle_position_{dir_b}", ("star", "particle_mass"), weight_field=weight_field)
        plot.set_unit(('star', 'particle_mass'), 'Msun')
        if annotate_COM:
            self.annotate_COMs(plot)
        if cbar_label is not None:
            plot.set_colorbar_label(('star', 'particle_mass'), cbar_label)
        plot.zoom(zoom)
        plot.save(savefile)
        return plot
    
    def create_COM_plots(self, max_zoom=50, diff_zoom=2, framerate=5):
        zoom_levels = [1]+list(np.arange(2, max_zoom+1, diff_zoom))
        os.makedirs(os.path.join(self.outdir, "COM_plots"), exist_ok=True)

        for i, zoom in enumerate(zoom_levels):
            for dir in ['x', 'y', 'z']:
                savefile = os.path.join(self.outdir, "COM_plots", f"gas_COM_{dir}_{i}.png")
                self.create_gas_plot(("gas", "density"), dir, savefile, zoom, True)
            for dir_a, dir_b in [['x', 'y'], ['y', 'z'], ['z', 'x']]:
                savefile = os.path.join(self.outdir, "COM_plots", f"star_COM_{dir_a}{dir_b}_{i}.png")
                self.create_star_plot(dir_a, dir_b, savefile, zoom, True)

        create_movie(os.path.join(self.outdir, "COM_plots"), "star_COM_xy_(\d+).png", self.outdir, "star_xy_COM.mp4", framerate=framerate)
        create_movie(os.path.join(self.outdir, "COM_plots"), "star_COM_yz_(\d+).png", self.outdir, "star_yz_COM.mp4", framerate=framerate)
        create_movie(os.path.join(self.outdir, "COM_plots"), "star_COM_zx_(\d+).png", self.outdir, "star_zx_COM.mp4", framerate=framerate)

        create_movie(os.path.join(self.outdir, "COM_plots"), "gas_COM_x_(\d+).png", self.outdir, "gas_x_COM.mp4", framerate=framerate)
        create_movie(os.path.join(self.outdir, "COM_plots"), "gas_COM_y_(\d+).png", self.outdir, "gas_y_COM.mp4", framerate=framerate)
        create_movie(os.path.join(self.outdir, "COM_plots"), "gas_COM_z_(\d+).png", self.outdir, "gas_z_COM.mp4", framerate=framerate)
        
        os.system(f"rm -r {os.path.join(self.outdir, 'COM_plots')}")

    def create_domain_maps(self):
        for dir in ["x", "y", "z"]:
            savefile_temp = os.path.join(self.outdir, f"{dir}_temperature.pdf")
            savefile_dens = os.path.join(self.outdir, f"{dir}_density.pdf")
            self.create_gas_plot(("gas", "density"), dir, savefile_dens)
            self.create_gas_plot(("gas", "temperature_over_mu"), dir, savefile_temp)
        for dir_a, dir_b in [["x", "y"], ["y", "z"], ["z", "x"]]:
            savefile_star = os.path.join(self.outdir, f"{dir_a}{dir_b}_star.pdf")
            savefile_star_avg = os.path.join(self.outdir, f"{dir_a}{dir_b}_star_avg.pdf")
            self.create_star_plot(dir_a, dir_b, savefile_star)
            self.create_star_plot(dir_a, dir_b, savefile_star_avg, weight_field=('star', 'particle_ones'))
        
            savefile_dm = os.path.join(self.outdir, f"{dir_a}{dir_b}_dm.pdf")
            dm_plt = yt.ParticlePlot(self.ds, ('DM', f'particle_position_{dir_a}'), ('DM', f'particle_position_{dir_b}'), ('DM', 'particle_mass'))
            dm_plt.set_unit(('DM', 'particle_mass'), "Msun")
            dm_plt.save(savefile_dm)

            savefile_dm_avg = os.path.join(self.outdir, f"{dir_a}{dir_b}_dm_avg.pdf")
            dm_plt_avg = dm_plt = yt.ParticlePlot(self.ds, ('DM', f'particle_position_{dir_a}'), ('DM', f'particle_position_{dir_b}'), ('DM', 'particle_mass'), weight_field=('DM', 'particle_ones'))
            dm_plt_avg.set_unit(('DM', 'particle_mass'), "Msun")
            dm_plt_avg.save(savefile_dm_avg)
        
    def locate_gas_disc(self, mass_fraction=0.9):
        if hasattr(self, "z0"):
            z0 = self.z0
        else:
            z0, _ = self.get_scale_height()
        ad = self.ad
        com = self.get_gas_COM()
        x = ad["x"] - com[0]
        y = ad["y"] - com[1]
        z = ad["z"] - com[2]
        in_disc = (np.abs(z) < z0)
        x_in = x[in_disc]
        y_in = y[in_disc]

        masses = ad["cell_mass"][in_disc]
        R = np.sqrt(x_in**2 + y_in**2)
        inds = np.argsort(R)
        R_sorted = R[inds]
        masses_sorted = masses[inds]
        cum_mass = np.cumsum(masses_sorted)
        total_mass = cum_mass[-1]
        i = np.searchsorted(cum_mass, mass_fraction * total_mass)
        R = R_sorted[i]

        return R.to("kpc"), z0

    def plot_gas_disc(self, mass_fraction=0.9):
        R, z0 = self.locate_gas_disc(mass_fraction)
        p = yt.ProjectionPlot(self.ds, 'z', 'density', weight_field="ones")
        com_gas = self.get_gas_COM()
        p.annotate_marker(com_gas, marker="x", coord_system='data', s=15)
        p.annotate_sphere(com_gas, R, coord_system='data')
        p.annotate_marker(com_gas, marker="x", coord_system='data', s=15)
        p.annotate_text(com_gas, "Gas COM", coord_system='data', text_args={"color": "black", "size": 15})
        p.save(os.path.join(self.outdir, "gas_disc.pdf"))

        return R.to("kpc"), z0.to("kpc")
    
    def get_disc_criteria(self, mass_fraction=0.9):
        com_gas = self.get_gas_COM()
        R, z0 = self.locate_gas_disc(mass_fraction)
        x_com = com_gas[0].to("kpc").value
        y_com = com_gas[1].to("kpc").value
        return [f'obj["z"].to("kpc").value < {(GalaxySnapshot.SCALE_HEIGHT_FACTOR * z0 + com_gas[2]).to("kpc").value}',
                f'obj["z"].to("kpc").value > {(com_gas[2] - GalaxySnapshot.SCALE_HEIGHT_FACTOR * z0).to("kpc").value}',
                f'((obj["x"].to("kpc").value - {x_com})**2 + (obj["y"].to("kpc").value - {y_com})**2) < {R.to("kpc").value**2}']

    def create_phase_plots(self, mass_fraction=0.9):
        p_complete = yt.PhasePlot(self.ds, ("gas", "density"), ("gas", "temperature_over_mu"), ("gas", "mass"))
        p_complete.set_unit(("gas", "mass"), "Msun")
        p_complete.annotate_title("Complete Phase Plot")
        p_complete.save(os.path.join(self.outdir, "phase_plot.pdf"))

        disc_criteria = self.get_disc_criteria(mass_fraction)
        ad_disc = self.ad.cut_region(disc_criteria)
        p_disc = yt.PhasePlot(ad_disc, ("gas", "density"), ("gas", "temperature_over_mu"), ("gas", "mass"))
        p_disc.set_unit(("gas", "mass"), "Msun")
        p_disc.annotate_title("Disc Phase Plot")
        p_disc.save(os.path.join(self.outdir, "phase_plot_disc.pdf"))

    def get_ad_galaxy_phase(self, disc_phase_criteria):
        ad_phase = self.ad.cut_region(disc_phase_criteria)
        return ad_phase

    def get_ad_galaxy(self, disc_criteria):
        ad_galaxy = self.ad.cut_region(disc_criteria)
        return ad_galaxy

    def get_phase_info(self, phase_criteria, disc_criteria, name="phase"):
        disc_phase_criteria = disc_criteria + phase_criteria
        ad_galaxy = self.get_ad_galaxy(disc_criteria)
        ad_galaxy_phase = self.get_ad_galaxy_phase(disc_phase_criteria)
        galaxy_volume = ad_galaxy["cell_volume"].sum().to("kpc**3")
        galaxy_resolution_elements = ad_galaxy["cell_volume"].size
        phase_volume = ad_galaxy_phase["cell_volume"].sum().to("kpc**3")
        phase_resolution_elements = ad_galaxy_phase["cell_volume"].size
        phase_volume_fraction = phase_volume / galaxy_volume
        phase_resolution_fraction = phase_resolution_elements / galaxy_resolution_elements
        info = {
            "galaxy_volume": galaxy_volume,
            "galaxy_resolution_elements": galaxy_resolution_elements,
            f"{name}_volume": phase_volume,
            f"{name}_resolution_elements": phase_resolution_elements,
            f"{name}_volume_fraction": phase_volume_fraction,
            f"{name}_resolution_fraction": phase_resolution_fraction,
            f"{name}_fraction_per_level": {}
        }
        for level in range(self.get_max_level() + 1):
            num_cells = (ad_galaxy_phase[("index", "grid_level")] == level).sum()
            info[f"{name}_fraction_per_level"][level] = num_cells / phase_resolution_elements
        return info

    def get_phase_fraction(self, phase_criteria, name="phase", mass_fraction=0.9):
        disc_criteria = self.get_disc_criteria(mass_fraction)
        phase_info = self.get_phase_info(phase_criteria, disc_criteria, name)
        ad_galaxy_phase = self.get_ad_galaxy_phase(disc_criteria + phase_criteria)
        return phase_info, ad_galaxy_phase

    def plot_phase_projection(self, phase_criteria, name="phase", mass_fraction=0.9):
        phase_info, ad_galaxy_phase = self.get_phase_fraction(phase_criteria, name, mass_fraction)
        ad_full_phase = self.ad.cut_region(phase_criteria)

        x_galaxy = ad_galaxy_phase["x"]
        y_galaxy = ad_galaxy_phase["y"]
        z_galaxy = ad_galaxy_phase["z"]
        width_galaxy = ((x_galaxy.max() - x_galaxy.min()).to("code_length"), (y_galaxy.max() - y_galaxy.min()).to("code_length"), (z_galaxy.max() - z_galaxy.min()).to("code_length"))
        p_x = yt.ProjectionPlot(self.ds, 'z', ('gas', 'density'), data_source=ad_galaxy_phase, width=width_galaxy, weight_field="ones")
        p_y = yt.ProjectionPlot(self.ds, 'x', ('gas', 'density'), data_source=ad_galaxy_phase, width=width_galaxy, weight_field="ones")
        p_z = yt.ProjectionPlot(self.ds, 'y', ('gas', 'density'), data_source=ad_galaxy_phase, width=width_galaxy, weight_field="ones")
        p_x.set_axes_unit('kpc')
        p_y.set_axes_unit('kpc')
        p_z.set_axes_unit('kpc')
        p_x.save(os.path.join(self.outdir, f"{name}_projection_x_sch_{GalaxySnapshot.SCALE_HEIGHT_FACTOR}.pdf"))
        p_y.save(os.path.join(self.outdir, f"{name}_projection_y_sch_{GalaxySnapshot.SCALE_HEIGHT_FACTOR}.pdf"))
        p_z.save(os.path.join(self.outdir, f"{name}_projection_z_sch_{GalaxySnapshot.SCALE_HEIGHT_FACTOR}.pdf"))

        x_full = ad_full_phase["x"]
        y_full = ad_full_phase["y"]
        z_full = ad_full_phase["z"]
        width_full = ((x_full.max() - x_full.min()).to("code_length"), (y_full.max() - y_full.min()).to("code_length"), (z_full.max() - z_full.min()).to("code_length"))
        p_x_full = yt.ProjectionPlot(self.ds, 'z', ('gas', 'density'), data_source=ad_full_phase, width=width_full, weight_field="ones")
        p_y_full = yt.ProjectionPlot(self.ds, 'x', ('gas', 'density'), data_source=ad_full_phase, width=width_full, weight_field="ones")
        p_z_full = yt.ProjectionPlot(self.ds, 'y', ('gas', 'density'), data_source=ad_full_phase, width=width_full, weight_field="ones")
        p_x_full.set_axes_unit('kpc')
        p_y_full.set_axes_unit('kpc')
        p_z_full.set_axes_unit('kpc')
        p_x_full.save(os.path.join(self.outdir, f"{name}_projection_x_full.pdf"))
        p_y_full.save(os.path.join(self.outdir, f"{name}_projection_y_full.pdf"))
        p_z_full.save(os.path.join(self.outdir, f"{name}_projection_z_full.pdf"))

        return phase_info