import numpy as np
from scipy.spatial import KDTree
from util import numba_unstructured_gaussian_kernel, numba_weighted_std, numba_get_rho_by_rho0, numba_get_mach
from mpi4py import MPI
from tqdm import tqdm

import time

class Turbulence:
    RHO_INDEX = 0
    LN_RHO_INDEX = 1
    VEL_X_INDEX = 2
    VEL_Y_INDEX = 3
    VEL_Z_INDEX = 4
    CS_INDEX = 5
    DATASET_SIZE = CS_INDEX + 1

    TURB_SCRATCH_RHO_INDEX = 0
    TURB_SCRATCH_LN_RHO_INDEX = 1
    TURB_SCRATCH_VEL_X_INDEX = 2
    TURB_SCRATCH_VEL_Y_INDEX = 3
    TURB_SCRATCH_VEL_Z_INDEX = 4
    TURB_SCRATCH_MACH_INDEX = 5
    TURB_SCRATCH_SIZE = TURB_SCRATCH_MACH_INDEX + 1

    MIN_CELLS_PER_TURB_KERNEL = 30

    fill_smooth_scratch_time = 0
    turb_sc_calc_time = 0
    fill_gaussian_kernel_scratch_2_time = 0
    query_ball_time = 0
    size_calc_time = 0

    def update_load_balanced_order(self):
        if self.comm is None or self.size == 1:
            self.load_balanced_order = np.arange(self._ad["x"].shape[0])
            return
        x = self._ad["x"].to(self.length_unit).value
        y = self._ad["y"].to(self.length_unit).value
        z = self._ad["z"].to(self.length_unit).value
        dx = self._ad["dx"].to(self.length_unit).value
        dy = self._ad["dy"].to(self.length_unit).value
        dz = self._ad["dz"].to(self.length_unit).value
        positions = np.array([x, y, z]).T
        tree = KDTree(positions)
        neighbor_lists = tree.query_ball_point(positions, 1.3 * self.turb_fwhm_factor * np.sqrt(dx**2 + dy**2 + dz**2))
        neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_lists])
        items = list(enumerate(neighbor_counts))
        items.sort(key=lambda x: -x[1])
        N = len(neighbor_counts)
        P = self.size
        bins = [[] for _ in range(P)]
        loads = np.zeros(P, dtype=np.int32)
        for (idx, count) in items:
            i = np.argmin(loads)
            bins[i].append(idx)
            loads[i] += count
        self.load_balanced_order = np.concatenate(bins)

    def get_data(self, key, unit):
        if not hasattr(self, "load_balanced_order"):
            self.update_load_balanced_order()
        return self._ad[key].to(unit).value[self.load_balanced_order]

    def __init__(self, ad, turb_fwhm_factor=5):
        '''
        ad: yt cut_region() object containing the data for the region/phase of interest.
        turb_fwhm_factor: FWHM of the roving kernel is turb_fwhm_factor * sqrt(dx^2 + dy^2 + dz^2).
        '''
        self._ad = ad
        try:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        except NameError:
            self.comm = None
            self.rank = 0
            self.size = 1

        self.length_unit = self._ad.ds.quan(1, "pc")
        self.rho_unit = self._ad.ds.quan(1, "g/cm**3")
        self.vel_unit = self._ad.ds.quan(1, "km/s")

        self.turb_fwhm_factor = turb_fwhm_factor

        self.x = self.get_data("x", self.length_unit)
        self.y = self.get_data("y", self.length_unit)
        self.z = self.get_data("z", self.length_unit)
        self.dx = self.get_data("dx", self.length_unit)
        self.dy = self.get_data("dy", self.length_unit)
        self.dz = self.get_data("dz", self.length_unit)
        self.volume = self.get_data("cell_volume", self.length_unit**3)

        self.num_cells = self.x.shape[0]

        self.dataset = np.zeros((Turbulence.DATASET_SIZE, self.num_cells))
        self.dataset[Turbulence.RHO_INDEX] = self.get_data("density", self.rho_unit)
        self.dataset[Turbulence.LN_RHO_INDEX] = np.log(self.dataset[Turbulence.RHO_INDEX])
        self.dataset[Turbulence.VEL_X_INDEX] = self.get_data("velocity_x", self.vel_unit)
        self.dataset[Turbulence.VEL_Y_INDEX] = self.get_data("velocity_y", self.vel_unit)
        self.dataset[Turbulence.VEL_Z_INDEX] = self.get_data("velocity_z", self.vel_unit)
        self.dataset[Turbulence.CS_INDEX] = self.get_data("sound_speed", self.vel_unit)

        self.positions = np.array([self.x, self.y, self.z]).T
        self.tree = KDTree(self.positions)

        self.dens_disp = np.zeros(self.positions.shape[0])
        self.rms_turb_mach = np.zeros(self.positions.shape[0])
        self.b = np.zeros(self.positions.shape[0])

        self.turb_scratch = np.zeros((Turbulence.TURB_SCRATCH_SIZE, self.num_cells))
        self.turb_scratch[Turbulence.TURB_SCRATCH_RHO_INDEX] = np.zeros(self.num_cells)
        self.turb_scratch[Turbulence.TURB_SCRATCH_LN_RHO_INDEX] = np.zeros(self.num_cells)
        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_X_INDEX] = np.zeros(self.num_cells)
        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Y_INDEX] = np.zeros(self.num_cells)
        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Z_INDEX] = np.zeros(self.num_cells)
        self.turb_scratch[Turbulence.TURB_SCRATCH_MACH_INDEX] = np.zeros(self.num_cells)

        self.kernel_scratch = np.zeros(self.positions.shape[0])
        self.distance_scratch = np.zeros((self.positions.shape[0], 3))

        self.mean_cell_length_monitor = np.zeros(self.num_cells)
        self.std_cell_length_monitor = np.zeros(self.num_cells)
        self.turb_kernel_num_cells_monitor = np.zeros(self.num_cells)
        self.turb_kernel_radius_monitor = np.zeros(self.num_cells)
        self.center_weight_x_monitor = np.zeros(self.num_cells)
        self.center_weight_y_monitor = np.zeros(self.num_cells)
        self.center_weight_z_monitor = np.zeros(self.num_cells)

        self.x_scratch = np.zeros(self.num_cells)
        self.y_scratch = np.zeros(self.num_cells)
        self.z_scratch = np.zeros(self.num_cells)
        self.dx_scratch = np.zeros(self.num_cells)

    def fill_gaussian_kernel_scratch(self, center_index, kernel_indices, end, fwhm):
        numba_unstructured_gaussian_kernel(self.positions[kernel_indices], self.positions[center_index], 
                                           self.volume[kernel_indices], fwhm, self.kernel_scratch[:end])

    def fill_smooth_scratch(self, turb_kernel_indices, fwhm):
        start_t = time.time()
        r = 1.3 * fwhm
        all_smooth_kernel_indices = self.tree.query_ball_point(self.positions[turb_kernel_indices], r, return_sorted=False)
        end_t = time.time()
        Turbulence.query_ball_time += end_t - start_t
        for i in range(len(turb_kernel_indices)):
            start_t = time.time()
            center_index = turb_kernel_indices[i] 
            smooth_kernel_indices = all_smooth_kernel_indices[i]  
            end = len(smooth_kernel_indices)
            end_t = time.time()
            Turbulence.size_calc_time += end_t - start_t

            start_t = time.time()
            self.fill_gaussian_kernel_scratch(center_index, smooth_kernel_indices, end, fwhm)
            end_t = time.time()
            Turbulence.fill_gaussian_kernel_scratch_2_time += end_t - start_t

            start_t = time.time()
            self.turb_scratch[Turbulence.LN_RHO_INDEX][i], self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_X_INDEX][i], \
                self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Y_INDEX][i], self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Z_INDEX][i] = \
                    self.dataset[Turbulence.LN_RHO_INDEX:Turbulence.VEL_Z_INDEX+1, smooth_kernel_indices].dot(self.kernel_scratch[:end])
            end_t = time.time()
            Turbulence.turb_sc_calc_time += end_t - start_t

    def fill_turb_scratch(self, kernel_indices, end):
        dataset_slice = np.s_[Turbulence.LN_RHO_INDEX:Turbulence.VEL_Z_INDEX+1]
        np.subtract(self.dataset[dataset_slice, kernel_indices], self.turb_scratch[dataset_slice, :end], 
                    out=self.turb_scratch[dataset_slice, :end])

    def get_weighted_std(self, end, dataset):
        return numba_weighted_std(dataset[:end], self.kernel_scratch[:end])

    def get_turbulence_params(self, index):
        '''
        Returns density dispersion, RMS Mach number, and b parameter for a given cell.
        index: index of the cell in the tree object
        '''
        cell_location = self.positions[index]
        dx, dy, dz = self.dx[index], self.dy[index], self.dz[index]
        turb_fwhm = self.turb_fwhm_factor * np.sqrt(dx**2 + dy**2 + dz**2)
        turb_r = 1.3 * turb_fwhm
        turb_kernel_indices = self.tree.query_ball_point(cell_location, turb_r)
        end = len(turb_kernel_indices)

        if end > Turbulence.MIN_CELLS_PER_TURB_KERNEL:
            self.x_scratch[:end] = self.x[turb_kernel_indices]
            self.y_scratch[:end] = self.y[turb_kernel_indices]
            self.z_scratch[:end] = self.z[turb_kernel_indices]
            self.dx_scratch[:end] = self.dx[turb_kernel_indices]

            self.turb_kernel_num_cells_monitor[index] = end
            self.turb_kernel_radius_monitor[index] = turb_r
            self.mean_cell_length_monitor[index] = np.mean(self.dx_scratch[:end])
            self.std_cell_length_monitor[index] = np.std(self.dx_scratch[:end])

            st = time.time()
            self.fill_smooth_scratch(turb_kernel_indices, turb_fwhm / 2)
            et = time.time()
            Turbulence.fill_smooth_scratch_time += et - st

            self.fill_turb_scratch(turb_kernel_indices, end)
            
            sl = np.s_[0:end]
            numba_get_rho_by_rho0(self.turb_scratch[Turbulence.LN_RHO_INDEX, sl], self.turb_scratch[Turbulence.TURB_SCRATCH_RHO_INDEX, sl])
            numba_get_mach(self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_X_INDEX, sl], self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Y_INDEX, sl], 
                        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Z_INDEX, sl], self.dataset[Turbulence.CS_INDEX, turb_kernel_indices], 
                        self.turb_scratch[Turbulence.TURB_SCRATCH_MACH_INDEX, sl])        

            self.fill_gaussian_kernel_scratch(index, turb_kernel_indices, end, turb_fwhm)

            self.center_weight_x_monitor[index] = self.x_scratch[:end].dot(self.kernel_scratch[:end])
            self.center_weight_y_monitor[index] = self.y_scratch[:end].dot(self.kernel_scratch[:end])
            self.center_weight_z_monitor[index] = self.z_scratch[:end].dot(self.kernel_scratch[:end])

            dens_disp = self.get_weighted_std(end, self.turb_scratch[Turbulence.TURB_SCRATCH_RHO_INDEX])
            rms_turb_mach = self.get_weighted_std(end, self.turb_scratch[Turbulence.TURB_SCRATCH_MACH_INDEX])

            b = dens_disp / rms_turb_mach

        else:
            dens_disp = np.nan
            rms_turb_mach = np.nan
            b = np.nan
            self.turb_kernel_num_cells_monitor[index] = end
            self.turb_kernel_radius_monitor[index] = np.nan
            self.mean_cell_length_monitor[index] = np.nan
            self.std_cell_length_monitor[index] = np.nan
            self.center_weight_x_monitor[index] = np.nan
            self.center_weight_y_monitor[index] = np.nan
            self.center_weight_z_monitor[index] = np.nan

        return dens_disp, rms_turb_mach, b
    
    def fill_turbulence_maps(self):
        num_cells = self.positions.shape[0]
        cells_per_proc = num_cells // self.size
        start = self.rank * cells_per_proc
        end = (self.rank + 1) * cells_per_proc if self.rank < self.size - 1 else num_cells
        print(f"Rank {self.rank} processing cells {start} to {end}")
        start_t = time.time()
        for i in tqdm(range(start, end), desc=f"Rank {self.rank}", position=self.rank):
            self.dens_disp[i], self.rms_turb_mach[i], self.b[i] = self.get_turbulence_params(i)
        end_t = time.time()
        print(f"Rank {self.rank} finished processing cells {start} to {end} in {(end_t - start_t)/60:.2f} minutes")

        print(f"Rank {self.rank} smooth scratch fill time: {Turbulence.fill_smooth_scratch_time:.2f} seconds")
        print(f"Rank {self.rank} turbulence scratch calculation time: {Turbulence.turb_sc_calc_time:.2f} seconds")
        print(f"Rank {self.rank} fill gaussian kernel scratch 2 time: {Turbulence.fill_gaussian_kernel_scratch_2_time:.2f} seconds")
        print(f"Rank {self.rank} query ball time: {Turbulence.query_ball_time:.2f} seconds")
        print(f"Rank {self.rank} size calculation time: {Turbulence.size_calc_time:.2f} seconds")

        if self.comm is not None and self.size > 1:
            if self.rank == 0:
                for i in range(1, self.size):
                    sl = np.s_[i * cells_per_proc: (i + 1) * cells_per_proc] if i < self.size - 1 else np.s_[i * cells_per_proc:]
                    self.dens_disp[sl] = self.comm.recv(source=i, tag=0)
                    self.rms_turb_mach[sl] = self.comm.recv(source=i, tag=1)
                    self.b[sl] = self.comm.recv(source=i, tag=2)
                    self.turb_kernel_num_cells_monitor[sl] = self.comm.recv(source=i, tag=3)
                    self.turb_kernel_radius_monitor[sl] = self.comm.recv(source=i, tag=4)
                    self.mean_cell_length_monitor[sl] = self.comm.recv(source=i, tag=5)
                    self.std_cell_length_monitor[sl] = self.comm.recv(source=i, tag=6)
                    self.center_weight_x_monitor[sl] = self.comm.recv(source=i, tag=7)
                    self.center_weight_y_monitor[sl] = self.comm.recv(source=i, tag=8)
                    self.center_weight_z_monitor[sl] = self.comm.recv(source=i, tag=9)
            else:
                self.comm.send(self.dens_disp[start:end], dest=0, tag=0)
                self.comm.send(self.rms_turb_mach[start:end], dest=0, tag=1)
                self.comm.send(self.b[start:end], dest=0, tag=2)
                self.comm.send(self.turb_kernel_num_cells_monitor[start:end], dest=0, tag=3)
                self.comm.send(self.turb_kernel_radius_monitor[start:end], dest=0, tag=4)
                self.comm.send(self.mean_cell_length_monitor[start:end], dest=0, tag=5)
                self.comm.send(self.std_cell_length_monitor[start:end], dest=0, tag=6)
                self.comm.send(self.center_weight_x_monitor[start:end], dest=0, tag=7)
                self.comm.send(self.center_weight_y_monitor[start:end], dest=0, tag=8)
                self.comm.send(self.center_weight_z_monitor[start:end], dest=0, tag=9)