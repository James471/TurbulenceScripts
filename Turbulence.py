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

    X_INDEX = 0
    Y_INDEX = 1
    Z_INDEX = 2
    DX_INDEX = 3
    DY_INDEX = 4
    DZ_INDEX = 5
    VOLUME_INDEX = 6
    SHARED_LOC_SIZE = VOLUME_INDEX + 1

    TURB_SCRATCH_RHO_INDEX = 0
    TURB_SCRATCH_LN_RHO_INDEX = 1
    TURB_SCRATCH_VEL_X_INDEX = 2
    TURB_SCRATCH_VEL_Y_INDEX = 3
    TURB_SCRATCH_VEL_Z_INDEX = 4
    TURB_SCRATCH_MACH_INDEX = 5
    TURB_SCRATCH_SIZE = TURB_SCRATCH_MACH_INDEX + 1

    MIN_CELLS_PER_TURB_KERNEL = 30

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

    def update_max_scratch_size(self):
        self.scratch_size = 0
        for i in range(self.start, self.end):
            dx, dy, dz = self.dx[i], self.dy[i], self.dz[i]
            turb_fwhm = self.turb_fwhm_factor * np.sqrt(dx**2 + dy**2 + dz**2)
            turb_r = 1.3 * turb_fwhm
            turb_kernel_indices = self.tree.query_ball_point(self.positions[i], turb_r)
            end = len(turb_kernel_indices)
            if end > self.scratch_size:
                self.scratch_size = end
        self.scratch_size *= 10  # Fingers crossed
        self.scratch_size = min(self.scratch_size, self._ad["x"].shape[0])

    def __init__(self, ad, turb_fwhm_factor=5):
        '''
        ad: yt cut_region() object containing the data for the region/phase of interest.
        turb_fwhm_factor: FWHM of the roving kernel is turb_fwhm_factor * sqrt(dx^2 + dy^2 + dz^2).
        '''
        self._ad = ad
        self.length_unit = self._ad.ds.quan(1, "pc")
        self.rho_unit = self._ad.ds.quan(1, "g/cm**3")
        self.vel_unit = self._ad.ds.quan(1, "km/s")
        self.turb_fwhm_factor = turb_fwhm_factor
        try:
            self.comm = MPI.COMM_WORLD
            self.shmcomm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            N = self.get_data("x", self.length_unit).shape[0]
            itemsize = np.dtype(np.float64).itemsize
            win_dataset = MPI.Win.Allocate_shared(self.DATASET_SIZE * N * itemsize if self.rank == 0 else 0, itemsize, MPI.INFO_NULL, self.shmcomm)
            win_loc = MPI.Win.Allocate_shared(self.SHARED_LOC_SIZE * N * itemsize if self.rank == 0 else 0, itemsize, MPI.INFO_NULL, self.shmcomm)
            win_pos = MPI.Win.Allocate_shared(3 * N * itemsize if self.rank == 0 else 0, itemsize, MPI.INFO_NULL, self.shmcomm)
            self.shmcomm.Barrier()
            buf_dataset, _ = win_dataset.Shared_query(0)
            buf_loc, _ = win_loc.Shared_query(0)
            buf_pos, _ = win_pos.Shared_query(0)
            shared_dataset = np.ndarray((self.DATASET_SIZE, N), dtype=np.float64, buffer=buf_dataset)
            shared_loc = np.ndarray((self.SHARED_LOC_SIZE, N), dtype=np.float64, buffer=buf_loc)
            shared_pos = np.ndarray((N , 3), dtype=np.float64, buffer=buf_pos)

        except NameError:
            self.comm = None
            self.shmcomm = None
            self.rank = 0
            self.size = 1
            N = self._ad["x"].shape[0]
            shared_dataset = np.zeros((self.DATASET_SIZE, N), dtype=np.float64)
            shared_loc = np.zeros((self.SHARED_LOC_SIZE, N), dtype=np.float64)
            shared_pos = np.zeros((N, 3), dtype=np.float64)

        if self.rank == 0:
            shared_loc[Turbulence.X_INDEX] = self.get_data("x", self.length_unit)
            shared_loc[Turbulence.Y_INDEX] = self.get_data("y", self.length_unit)
            shared_loc[Turbulence.Z_INDEX] = self.get_data("z", self.length_unit)
            shared_loc[Turbulence.DX_INDEX] = self.get_data("dx", self.length_unit)
            shared_loc[Turbulence.DY_INDEX] = self.get_data("dy", self.length_unit)
            shared_loc[Turbulence.DZ_INDEX] = self.get_data("dz", self.length_unit)
            shared_loc[Turbulence.VOLUME_INDEX] = self.get_data("cell_volume", self.length_unit**3)

            shared_dataset[Turbulence.RHO_INDEX] = self.get_data("density", self.rho_unit)
            shared_dataset[Turbulence.LN_RHO_INDEX] = np.log(shared_dataset[Turbulence.RHO_INDEX])
            shared_dataset[Turbulence.VEL_X_INDEX] = self.get_data("velocity_x", self.vel_unit)
            shared_dataset[Turbulence.VEL_Y_INDEX] = self.get_data("velocity_y", self.vel_unit)
            shared_dataset[Turbulence.VEL_Z_INDEX] = self.get_data("velocity_z", self.vel_unit)
            shared_dataset[Turbulence.CS_INDEX] = self.get_data("sound_speed", self.vel_unit)

            shared_pos[:, 0] = shared_loc[Turbulence.X_INDEX]
            shared_pos[:, 1] = shared_loc[Turbulence.Y_INDEX]
            shared_pos[:, 2] = shared_loc[Turbulence.Z_INDEX]

        self.shmcomm.Barrier()

        self.x = shared_loc[Turbulence.X_INDEX]
        self.y = shared_loc[Turbulence.Y_INDEX]
        self.z = shared_loc[Turbulence.Z_INDEX]
        self.dx = shared_loc[Turbulence.DX_INDEX]
        self.dy = shared_loc[Turbulence.DY_INDEX]
        self.dz = shared_loc[Turbulence.DZ_INDEX]
        self.volume = shared_loc[Turbulence.VOLUME_INDEX]

        self.positions = shared_pos
        self.tree = KDTree(self.positions)

        self.dataset = shared_dataset

        self.num_cells = self.x.shape[0]
        self.cells_per_proc = self.num_cells // self.size
        self.start = self.rank * self.cells_per_proc
        self.end = (self.rank + 1) * self.cells_per_proc if self.rank < self.size - 1 else self.num_cells
        self.num_cells_processing = self.end - self.start

        self.update_max_scratch_size()

        self.dens_disp = np.zeros(self.num_cells_processing)
        self.rms_turb_mach = np.zeros(self.num_cells_processing)
        self.b = np.zeros(self.num_cells_processing)

        self.turb_scratch = np.zeros((Turbulence.TURB_SCRATCH_SIZE, self.scratch_size))
        self.turb_scratch[Turbulence.TURB_SCRATCH_RHO_INDEX] = np.zeros(self.scratch_size)
        self.turb_scratch[Turbulence.TURB_SCRATCH_LN_RHO_INDEX] = np.zeros(self.scratch_size)
        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_X_INDEX] = np.zeros(self.scratch_size)
        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Y_INDEX] = np.zeros(self.scratch_size)
        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Z_INDEX] = np.zeros(self.scratch_size)
        self.turb_scratch[Turbulence.TURB_SCRATCH_MACH_INDEX] = np.zeros(self.scratch_size)

        self.kernel_scratch = np.zeros(self.scratch_size)
        self.distance_scratch = np.zeros((self.scratch_size, 3))

        self.mean_cell_length_monitor = np.zeros(self.num_cells_processing)
        self.std_cell_length_monitor = np.zeros(self.num_cells_processing)
        self.turb_kernel_num_cells_monitor = np.zeros(self.num_cells_processing)
        self.turb_kernel_radius_monitor = np.zeros(self.num_cells_processing)
        self.center_weight_x_monitor = np.zeros(self.num_cells_processing)
        self.center_weight_y_monitor = np.zeros(self.num_cells_processing)
        self.center_weight_z_monitor = np.zeros(self.num_cells_processing)

        self.x_scratch = np.zeros(self.scratch_size)
        self.y_scratch = np.zeros(self.scratch_size)
        self.z_scratch = np.zeros(self.scratch_size)
        self.dx_scratch = np.zeros(self.scratch_size)

        if self.rank == 0:
            self.complete_dens_disp = np.zeros(self.num_cells)
            self.complete_rms_turb_mach = np.zeros(self.num_cells)
            self.complete_b = np.zeros(self.num_cells)
            self.complete_turb_kernel_num_cells_monitor = np.zeros(self.num_cells)
            self.complete_turb_kernel_radius_monitor = np.zeros(self.num_cells)
            self.complete_mean_cell_length_monitor = np.zeros(self.num_cells)
            self.complete_std_cell_length_monitor = np.zeros(self.num_cells)
            self.complete_center_weight_x_monitor = np.zeros(self.num_cells)
            self.complete_center_weight_y_monitor = np.zeros(self.num_cells)
            self.complete_center_weight_z_monitor = np.zeros(self.num_cells)
        else:
            self.complete_dens_disp = None
            self.complete_rms_turb_mach = None
            self.complete_b = None
            self.complete_turb_kernel_num_cells_monitor = None
            self.complete_turb_kernel_radius_monitor = None
            self.complete_mean_cell_length_monitor = None
            self.complete_std_cell_length_monitor = None
            self.complete_center_weight_x_monitor = None
            self.complete_center_weight_y_monitor = None
            self.complete_center_weight_z_monitor = None

    def fill_gaussian_kernel_scratch(self, center_index, kernel_indices, end, fwhm):
        numba_unstructured_gaussian_kernel(self.positions[kernel_indices], self.positions[center_index], 
                                           self.volume[kernel_indices], fwhm, self.kernel_scratch[:end])

    def fill_smooth_scratch(self, turb_kernel_indices, fwhm):
        r = 1.3 * fwhm
        all_smooth_kernel_indices = self.tree.query_ball_point(self.positions[turb_kernel_indices], r, return_sorted=False)
        ln_rho_to_vel_z_slice = np.s_[Turbulence.LN_RHO_INDEX:Turbulence.VEL_Z_INDEX+1]
        for i in range(len(turb_kernel_indices)):
            center_index = turb_kernel_indices[i] 
            smooth_kernel_indices = all_smooth_kernel_indices[i]
            end = len(smooth_kernel_indices)

            self.fill_gaussian_kernel_scratch(center_index, smooth_kernel_indices, end, fwhm)

            self.turb_scratch[ln_rho_to_vel_z_slice, i] = np.dot(self.dataset[ln_rho_to_vel_z_slice, smooth_kernel_indices], 
                                                                self.kernel_scratch[:end])

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

            self.turb_kernel_num_cells_monitor[index - self.start] = end
            self.turb_kernel_radius_monitor[index - self.start] = turb_r
            self.mean_cell_length_monitor[index - self.start] = np.mean(self.dx_scratch[:end])
            self.std_cell_length_monitor[index - self.start] = np.std(self.dx_scratch[:end])

            self.fill_smooth_scratch(turb_kernel_indices, turb_fwhm / 2)

            self.fill_turb_scratch(turb_kernel_indices, end)
            
            sl = np.s_[0:end]
            numba_get_rho_by_rho0(self.turb_scratch[Turbulence.LN_RHO_INDEX, sl], self.turb_scratch[Turbulence.TURB_SCRATCH_RHO_INDEX, sl])
            numba_get_mach(self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_X_INDEX, sl], self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Y_INDEX, sl], 
                        self.turb_scratch[Turbulence.TURB_SCRATCH_VEL_Z_INDEX, sl], self.dataset[Turbulence.CS_INDEX, turb_kernel_indices], 
                        self.turb_scratch[Turbulence.TURB_SCRATCH_MACH_INDEX, sl])        

            self.fill_gaussian_kernel_scratch(index, turb_kernel_indices, end, turb_fwhm)

            self.center_weight_x_monitor[index - self.start] = self.x_scratch[:end].dot(self.kernel_scratch[:end])
            self.center_weight_y_monitor[index - self.start] = self.y_scratch[:end].dot(self.kernel_scratch[:end])
            self.center_weight_z_monitor[index - self.start] = self.z_scratch[:end].dot(self.kernel_scratch[:end])

            dens_disp = self.get_weighted_std(end, self.turb_scratch[Turbulence.TURB_SCRATCH_RHO_INDEX])
            rms_turb_mach = self.get_weighted_std(end, self.turb_scratch[Turbulence.TURB_SCRATCH_MACH_INDEX])

            b = dens_disp / rms_turb_mach

        else:
            dens_disp = np.nan
            rms_turb_mach = np.nan
            b = np.nan
            self.turb_kernel_num_cells_monitor[index - self.start] = end
            self.turb_kernel_radius_monitor[index - self.start] = np.nan
            self.mean_cell_length_monitor[index - self.start] = np.nan
            self.std_cell_length_monitor[index - self.start] = np.nan
            self.center_weight_x_monitor[index - self.start] = np.nan
            self.center_weight_y_monitor[index - self.start] = np.nan
            self.center_weight_z_monitor[index - self.start] = np.nan

        return dens_disp, rms_turb_mach, b
    
    def fill_turbulence_maps(self):
        cells_per_proc = self.cells_per_proc
        start = self.start
        end = self.end
        print(f"Rank {self.rank} processing cells {start} to {end}")
        start_time = time.time()
        for i in tqdm(range(start, end), desc=f"Rank {self.rank}", position=self.rank):
            local_index = i - start
            self.dens_disp[local_index], self.rms_turb_mach[local_index], self.b[local_index] = self.get_turbulence_params(i)
        end_time = time.time()
        print(f"Rank {self.rank} finished processing cells {start} to {end} in {(end_time - start_time)/60:.2f} minutes")
        self.shmcomm.Free()

        if self.comm is not None and self.size > 1:
            if self.rank == 0:
                self.complete_dens_disp[start:end] = self.dens_disp[:]
                self.complete_rms_turb_mach[start:end] = self.rms_turb_mach[:]
                self.complete_b[start:end] = self.b[:]
                self.complete_turb_kernel_num_cells_monitor[start:end] = self.turb_kernel_num_cells_monitor[:]
                self.complete_turb_kernel_radius_monitor[start:end] = self.turb_kernel_radius_monitor[:]
                self.complete_mean_cell_length_monitor[start:end] = self.mean_cell_length_monitor[:]
                self.complete_std_cell_length_monitor[start:end] = self.std_cell_length_monitor[:]
                self.complete_center_weight_x_monitor[start:end] = self.center_weight_x_monitor[:]
                self.complete_center_weight_y_monitor[start:end] = self.center_weight_y_monitor[:]
                self.complete_center_weight_z_monitor[start:end] = self.center_weight_z_monitor[:]
                for i in range(1, self.size):
                    sl = np.s_[i * cells_per_proc: (i + 1) * cells_per_proc] if i < self.size - 1 else np.s_[i * cells_per_proc:]
                    self.complete_dens_disp[sl] = self.comm.recv(source=i, tag=0)
                    self.complete_rms_turb_mach[sl] = self.comm.recv(source=i, tag=1)
                    self.complete_b[sl] = self.comm.recv(source=i, tag=2)
                    self.complete_turb_kernel_num_cells_monitor[sl] = self.comm.recv(source=i, tag=3)
                    self.complete_turb_kernel_radius_monitor[sl] = self.comm.recv(source=i, tag=4)
                    self.complete_mean_cell_length_monitor[sl] = self.comm.recv(source=i, tag=5)
                    self.complete_std_cell_length_monitor[sl] = self.comm.recv(source=i, tag=6)
                    self.complete_center_weight_x_monitor[sl] = self.comm.recv(source=i, tag=7)
                    self.complete_center_weight_y_monitor[sl] = self.comm.recv(source=i, tag=8)
                    self.complete_center_weight_z_monitor[sl] = self.comm.recv(source=i, tag=9)
            else:
                self.comm.send(self.dens_disp[:], dest=0, tag=0)
                self.comm.send(self.rms_turb_mach[:], dest=0, tag=1)
                self.comm.send(self.b[:], dest=0, tag=2)
                self.comm.send(self.turb_kernel_num_cells_monitor[:], dest=0, tag=3)
                self.comm.send(self.turb_kernel_radius_monitor[:], dest=0, tag=4)
                self.comm.send(self.mean_cell_length_monitor[:], dest=0, tag=5)
                self.comm.send(self.std_cell_length_monitor[:], dest=0, tag=6)
                self.comm.send(self.center_weight_x_monitor[:], dest=0, tag=7)
                self.comm.send(self.center_weight_y_monitor[:], dest=0, tag=8)
                self.comm.send(self.center_weight_z_monitor[:], dest=0, tag=9)
        else:
            self.complete_dens_disp = self.dens_disp
            self.complete_rms_turb_mach = self.rms_turb_mach
            self.complete_b = self.b
            self.complete_turb_kernel_num_cells_monitor = self.turb_kernel_num_cells_monitor
            self.complete_turb_kernel_radius_monitor = self.turb_kernel_radius_monitor
            self.complete_mean_cell_length_monitor = self.mean_cell_length_monitor
            self.complete_std_cell_length_monitor = self.std_cell_length_monitor
            self.complete_center_weight_x_monitor = self.center_weight_x_monitor
            self.complete_center_weight_y_monitor = self.center_weight_y_monitor
            self.complete_center_weight_z_monitor = self.center_weight_z_monitor