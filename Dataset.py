import pickle

class Dataset:
    def __init__(self, turbulence_object):
        self.x = turbulence_object.x
        self.y = turbulence_object.y
        self.z = turbulence_object.z
        self.dx = turbulence_object.dx
        self.dy = turbulence_object.dy
        self.dz = turbulence_object.dz
        self.volume = turbulence_object.volume
        self.num_cells = turbulence_object.num_cells
        self.positions = turbulence_object.positions
        self.tree = turbulence_object.tree
        self.turb_fwhm_factor = turbulence_object.turb_fwhm_factor
        self.dens_disp = turbulence_object.dens_disp
        self.rms_turb_mach = turbulence_object.rms_turb_mach
        self.b = turbulence_object.b
        self.mean_cell_length_monitor = turbulence_object.mean_cell_length_monitor
        self.std_cell_length_monitor = turbulence_object.std_cell_length_monitor
        self.turb_kernel_num_cells_monitor = turbulence_object.turb_kernel_num_cells_monitor
        self.turb_kernel_radius_monitor = turbulence_object.turb_kernel_radius_monitor
        self.center_weight_x_monitor = turbulence_object.center_weight_x_monitor
        self.center_weight_y_monitor = turbulence_object.center_weight_y_monitor
        self.center_weight_z_monitor = turbulence_object.center_weight_z_monitor
        self.length_unit = turbulence_object.length_unit.__str__()
        self.rho_unit = turbulence_object.rho_unit.__str__()
        self.vel_unit = turbulence_object.vel_unit.__str__()

    def store(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {path}")