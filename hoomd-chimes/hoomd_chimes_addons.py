import numpy as np
import hoomd
from hoomd.md.force import Custom
from hoomd.logging import log
import sys

class ChIMES_manybody(Custom):
    def __init__(
        self,
        sim,
        chimes_api_path: str,
        chimes_wrapper_path: str,
        param_file_path: str,
        N_particles: float,
        allow_replicates: bool = True
    ):
        super().__init__()
        
        if sim.device.communicator.num_ranks > 1:
            raise NotImplementedError("Parallel execution not implemented")
        
        sys.path.append(chimes_api_path)
        import chimescalc_serial_py

        chimescalc_serial_py.chimes_wrapper = chimescalc_serial_py.init_chimes_wrapper(
            chimes_wrapper_path + "/libchimescalc_dl.so"
        )
        chimescalc_serial_py.set_chimes(allow_replicates)
        chimescalc_serial_py.init_chimes(param_file_path, 0)
        self._calc_chimes = chimescalc_serial_py
        self._current_timestep = None
        self._zero_force_array = np.zeros(N_particles).astype(float).tolist()
        self._zero_stress_array = np.zeros(9).astype(float).tolist()

    def set_forces(self, timestep):
        if self._current_timestep != timestep:
            self._current_timestep = timestep

            with self._state.cpu_local_snapshot as snapshot:
                pos = np.asarray(snapshot.particles.position[:])
                box = np.asarray([snapshot.global_box.Lx, snapshot.global_box.Ly, snapshot.global_box.Lz])
                N = pos.shape[0]

                xcrd = pos[:, 0].tolist()
                ycrd = pos[:, 1].tolist()
                zcrd = pos[:, 2].tolist()
                volume = box[0] * box[1] * box[2]

                fx, fy, fz, stress, energy = self._calc_chimes.calculate_chimes(
                    natoms=N,
                    xcrd=xcrd,
                    ycrd=ycrd,
                    zcrd=zcrd,
                    atmtyps=["1"] * N,
                    cell_a=[box[0], 0.0, 0.0],
                    cell_b=[0.0, box[1], 0.0],
                    cell_c=[0.0, 0.0, box[2]],
                    energy=0.0,
                    fx=self._zero_force_array,
                    fy=self._zero_force_array,
                    fz=self._zero_force_array,
                    stress=self._zero_stress_array
                )
                self._current_total_energy = energy
                self._force = np.column_stack((fx[:], fy[:], fz[:]))
                self._current_total_virial = np.array(
                    [stress[0], stress[1], stress[2], stress[4], stress[5], stress[8]]
                ) * volume  # chimes return total stress tensor and hoomd requires irreducible virial element only
                # convert total stress to total virial via multiplying the toal stress with total volume
            

        with self.cpu_local_force_arrays as arrays:
            arrays.potential_energy[0] = self._current_total_energy  # it will cause the incorrect energy on each particle but wouldn't affect simulation.
            arrays.force[:] = self._force
            arrays.virial[0, :] = self._current_total_virial  # assign the total virial to first particle to integrate the volume.
            # it will cause the incorrect virial on each particle but wouldn't affect simulation.
