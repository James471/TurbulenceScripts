#!python3

from GalaxySnapshot import GalaxySnapshot
from Turbulence import Turbulence
from Dataset import Dataset
import design
import pickle
import os
from mpi4py import MPI

print("Loading GalaxySnapshot...")
DATADIR = "../data/output_00202/"
galaxy = GalaxySnapshot(DATADIR)
_, ad = galaxy.get_phase_fraction([f"obj['temperature_over_mu'] < {8e3/1.27}", f"obj['temperature_over_mu'] > {5e3/1.27}"], name="WNM")
turb = Turbulence(ad, turb_fwhm_factor=5)
print("Filling turbulence map for WNM...")
turb.fill_turbulence_maps()

comm = MPI.COMM_WORLD
if comm.Get_rank() == 0:
    print("Creating Dataset...")
    dataset = Dataset(turb)
    print("Storing Dataset...")
    dataset.store(os.path.join(DATADIR, "wnm_turb_dataset.pkl"))