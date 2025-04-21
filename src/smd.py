import numpy as np
import hoomd
from hoomd.md.force import Custom
from hoomd.logging import log


SMALL = np.finfo(np.float64).eps

class SMD_constV_couple(Custom):
    def __init__(
        self,
        sim: hoomd.simulation,
        group1_idx: np.array,
        group2_idx: np.array,
        dt: float,
        pulling_velocity: float,
        spring_constant: float,
        pulling_direction: np.array=None,
        R0=0.
    ):
        r"""Constant velocity SMD with coupling mode
        
        Args: 
            sim: hoomd.Simulation object
            group1_idx: an array of indexes of particle group 1
            group2_idx: an array of indexes of particle group 2
            dt: timestep
            pulling_velocity: SMD pulling velocity
            spring_constant: spring constant of the harmonic guiding potential for SMD pullings
            pulling_direction: the direction of SMD pullings, with negative (-) being forward pulling and positive (+) being reverse pulling  
            R0: distance deviation from the tether point based on the harmonic guiding potential

        """
        super().__init__()
        self._group1_idx = np.asarray(group1_idx)
        self._group2_idx = np.asarray(group2_idx)
        with sim._state.cpu_local_snapshot as snapshot:
            group1_realidx = snapshot.particles.rtag[self._group1_idx]
            group2_realidx = snapshot.particles.rtag[self._group2_idx]

            group1_mass = snapshot.particles.mass[group1_realidx]
            group2_mass = snapshot.particles.mass[group2_realidx]

            group1_pos = snapshot.particles.position[group1_realidx, :]
            group2_pos = snapshot.particles.position[group2_realidx, :]

            self._group1_cm = np.sum(group1_pos * group1_mass[:, None], axis=0) / group1_mass.sum()
            self._group2_cm = np.sum(group2_pos * group2_mass[:, None], axis=0) / group2_mass.sum()

            if pulling_direction is None:
                # determine direction automatically
                # point from group 1 to group 2
                r21 = self._group2_cm - self._group1_cm
                self._spring_length = np.linalg.norm(r21)
                self._distance = np.linalg.norm(r21)
                self._pulling_direction = r21 / np.linalg.norm(r21)
                self._auto = True
            else:
                r21 = self._group2_cm - self._group1_cm
                self._spring_length = np.linalg.norm(r21)
                self._distance = np.linalg.norm(r21)
                temp = np.asarray(pulling_direction)
                self._pulling_direction = temp / np.linalg.norm(temp)
                self._auto = False


        self._dt = dt
        self._velocity = pulling_velocity
        self._k = spring_constant
        self._R0 = R0
        
        self._current_timestep = None
        self._total_force = None
        self._group1_force = None
        self._group2_force = None
        self._pmf = 0.

    def set_forces(self, timestep):
        if self._current_timestep != timestep:
            self._current_timestep = timestep

            self._integrate_spring_forward()

            with self._state.cpu_local_snapshot as snapshot:
                group1_realidx = snapshot.particles.rtag[self._group1_idx]
                group2_realidx = snapshot.particles.rtag[self._group2_idx]

                group1_mass = snapshot.particles.mass[group1_realidx]
                group2_mass = snapshot.particles.mass[group2_realidx]

                group1_pos = snapshot.particles.position[group1_realidx, :]
                group2_pos = snapshot.particles.position[group2_realidx, :]

                self._group1_cm = np.sum(group1_pos * group1_mass[:, None], axis=0) / group1_mass.sum()
                self._group2_cm = np.sum(group2_pos * group2_mass[:, None], axis=0) / group2_mass.sum()

                current_r21 = self._group2_cm - self._group1_cm
                current_d0 = np.linalg.norm(current_r21)
                self._distance = current_d0
                if self._auto:
                    # determine direction automatically
                    # point from group 1 to group 2
                    self._pulling_direction = current_r21 / current_d0
                dr = current_r21 - self._spring_length * self._pulling_direction
                dd = np.linalg.norm(dr) - self._R0

                self._total_force = self._k * dd * dr / (dd + self._R0 + SMALL)

                group1_mass = snapshot.particles.mass[group1_realidx]
                group1_massfraction = group1_mass / group1_mass.sum()
                group2_mass = snapshot.particles.mass[group2_realidx]
                group2_massfraction = group2_mass / group2_mass.sum()

                self._group1_force = self._total_force * group1_massfraction[:, None]
                self._group2_force = -self._total_force * group2_massfraction[:, None]
            self._pmf += np.sum(self._total_force * self._pulling_direction) * np.abs(self._velocity) * self._dt
            
        with self._state.cpu_local_snapshot as snapshot:
            group1_realidx = snapshot.particles.rtag[self._group1_idx]
            group2_realidx = snapshot.particles.rtag[self._group2_idx]
        with self.cpu_local_force_arrays as arrays:
            arrays.force[group1_realidx, :] = self._group1_force
            arrays.force[group2_realidx, :] = self._group2_force

    def _integrate_spring_forward(self):
        self._spring_length += self._velocity * self._dt

    @log(category="scalar", is_property=False, requires_run=True)
    def t(self):
        if self._current_timestep is None:
            return 0
        return self._current_timestep

    @log(category="scalar", is_property=False, requires_run=True)
    def spring_length(self):
        return self._spring_length
        
    @log(category="scalar", is_property=False, requires_run=True)
    def force_x(self):
        return self._total_force[0]

    @log(category="scalar", is_property=False, requires_run=True)
    def force_y(self):
        return self._total_force[1]

    @log(category="scalar", is_property=False, requires_run=True)
    def force_z(self):
        return self._total_force[2]

    @log(category="scalar", is_property=False, requires_run=True)
    def total_force(self):
        return (self._total_force * self._pulling_direction).sum()

    @log(category="scalar", is_property=False, requires_run=True)
    def d(self):
        return self._distance
    
    @log(category="scalar", is_property=False, requires_run=True)
    def work(self):
        return self._pmf

    def get_writer(self, log_fn, dump_period, mode="w"):
        logger = hoomd.logging.Logger(categories=["scalar"])
        logger.add(
            self, 
            quantities=[
                "t", 
                "force_x", 
                "force_y", 
                "force_z",
                "total_force",
                "spring_length",
                "d", 
                "work"
            ],
            user_name="smd"
        )
        return hoomd.write.Table(
            output=open(log_fn, mode=mode, newline="\n"),
            trigger=hoomd.trigger.Periodic(dump_period),
            logger=logger,
        )

    def restart_sim(self, restart_fn):
        data = np.genfromtxt(restart_fn, skip_header=1)[-1, :]
        spring_length = data[5]
        pmf = data[-1]
        
        self._spring_length = spring_length
        self._pmf = pmf
        return


class SMD_constV_threebody(Custom):
    def __init__(
            self,
            sim: hoomd.simulation,
            group1_idx: np.array,
            group2_idx: np.array,
            group3_idx: np.array,
            dt: float,
            pulling_velocity: float,
            spring_constant_fix: float,
            spring_constant_pull: float,
            r_c: np.array = None,
            r_fix: np.array = None,
            pulling_direction: np.array = None,
            R0=0.
    ):
        r"""Constant velocity SMD with coupling mode
        
        Args: 
            sim: hoomd.Simulation object
            group1_idx: an array of indexes of particle group 1
            group2_idx: an array of indexes of particle group 2
            group3_idx: an array of indexes of particle group 3
            dt: timestep
            pulling_velocity: SMD pulling velocity
            spring_constant_fix: spring constant of the harmonic guiding potential that fixes particle group 1 and 2
            spring_constant_pull: spring constant of the harmonic guiding potential that pulls particle group 3
            r_c: the anchor point that the particle group 3 is pulled towards for work calculation. Default value is the midpoint between particle group 1 and 2 
            r_fix: anchor points that fix the positions of particle group 1 and 2 
            pulling_direction: the direction of SMD pullings, with negative (-) being forward pulling and positive (+) being reverse pulling  
            R0: distance deviation from the tether point based on the harmonic guiding potential

        """
        super().__init__()
        self._group1_idx = np.asarray(group1_idx)
        self._group2_idx = np.asarray(group2_idx)
        self._group3_idx = np.asarray(group3_idx)
        with sim._state.cpu_local_snapshot as snapshot:
            group1_realidx = snapshot.particles.rtag[self._group1_idx]
            group2_realidx = snapshot.particles.rtag[self._group2_idx]
            group3_realidx = snapshot.particles.rtag[self._group3_idx]

            group1_mass = snapshot.particles.mass[group1_realidx]
            group2_mass = snapshot.particles.mass[group2_realidx]
            group3_mass = snapshot.particles.mass[group3_realidx]

            group1_pos = snapshot.particles.position[group1_realidx, :]
            group2_pos = snapshot.particles.position[group2_realidx, :]
            group3_pos = snapshot.particles.position[group3_realidx, :]

            self._group1_cm = np.sum(group1_pos * group1_mass[:, None], axis=0) / group1_mass.sum()
            self._group2_cm = np.sum(group2_pos * group2_mass[:, None], axis=0) / group2_mass.sum()
            self._group3_cm = np.sum(group3_pos * group3_mass[:, None], axis=0) / group3_mass.sum()

            self._r1 = self._group1_cm
            self._r2 = self._group2_cm
            self._rc = (self._group1_cm + self._group2_cm) / 2
            if r_c is not None:
                self._rc = np.asarray(r_c)
            if r_fix is not None:
                r_fix = np.asarray(r_fix)
                assert r_fix.shape[0] == 2
                self._r1 = r_fix[0]
                self._r2 = r_fix[1]
            if pulling_direction is None:
                # determine direction automatically
                # point from C point to group 3
                r3c = self._group3_cm - self._rc
                self._spring_length = np.linalg.norm(r3c)
                self._spring_length0 = np.linalg.norm(r3c)
                self._pulling_direction = r3c / self._spring_length0
                self._auto = True
            else:
                r3c = self._group3_cm - self._rc
                self._spring_length = np.linalg.norm(r3c)
                self._spring_length0 = np.linalg.norm(r3c)
                temp = np.asarray(pulling_direction )
                self._pulling_direction = temp / np.linalg.norm(temp)
                self._auto = False

        self._dt = dt
        self._velocity = pulling_velocity
        self._k_fix = spring_constant_fix
        self._k_pull = spring_constant_pull
        self._R0 = R0

        self._current_timestep = None
        self._group1_force = None
        self._group2_force = None
        self._group3_force = None
        self._total_pulling_force = None

        self._dd1_devi = None
        self._dd2_devi = None
        self._d13 = None
        self._d23 = None
        self._d3c = None
        self._pmf = 0.

    def set_forces(self, timestep):
        if self._current_timestep != timestep:
            self._current_timestep = timestep

            self._integrate_spring_forward()

            with self._state.cpu_local_snapshot as snapshot:
                group1_realidx = snapshot.particles.rtag[self._group1_idx]
                group2_realidx = snapshot.particles.rtag[self._group2_idx]
                group3_realidx = snapshot.particles.rtag[self._group3_idx]

                group1_mass = snapshot.particles.mass[group1_realidx]
                group2_mass = snapshot.particles.mass[group2_realidx]
                group3_mass = snapshot.particles.mass[group3_realidx]

                group1_pos = snapshot.particles.position[group1_realidx, :]
                group2_pos = snapshot.particles.position[group2_realidx, :]
                group3_pos = snapshot.particles.position[group3_realidx, :]

                self._group1_cm = np.sum(group1_pos * group1_mass[:, None], axis=0) / group1_mass.sum()
                self._group2_cm = np.sum(group2_pos * group2_mass[:, None], axis=0) / group2_mass.sum()
                self._group3_cm = np.sum(group3_pos * group3_mass[:, None], axis=0) / group3_mass.sum()
                
                # Evaluate pulling constrain between C point and group3
                current_r3c = self._group3_cm - self._rc
                current_d3c = np.linalg.norm(current_r3c)
                self._d3c = current_d3c
                if self._auto:
                    # determine direction automatically
                    # point from C point to group 3
                    self._pulling_direction = current_r3c / current_d3c

                dr3c = current_r3c - self._spring_length * self._pulling_direction
                dd3c = np.linalg.norm(dr3c) - self._R0
                self._total_pulling_force = self._k_pull * dd3c * dr3c / (dd3c + self._R0 + SMALL)

                # Evaluate the constrain on group1
                dr1 = self._group1_cm - self._r1
                dd1 = np.linalg.norm(dr1)
                self._dd1_devi = dd1
                self._total_fix_force1 = self._k_fix * dd1 * dr1 / (dd1 + SMALL)

                # Evaluate the constrain on group2
                dr2 = self._group2_cm - self._r2
                dd2 = np.linalg.norm(dr2)
                self._dd2_devi = dd2
                self._total_fix_force2 = self._k_fix * dd2 * dr2 / (dd2 + SMALL)
                
                # calculate restoring force
                group1_massfraction = group1_mass / group1_mass.sum()
                group2_massfraction = group2_mass / group2_mass.sum()
                group3_massfraction = group3_mass / group3_mass.sum()

                self._group1_force = -self._total_fix_force1 * group1_massfraction[:, None]
                self._group2_force = -self._total_fix_force2 * group2_massfraction[:, None]
                self._group3_force = -self._total_pulling_force * group3_massfraction[:, None]
                
                # save the other information
                self._d13 = np.linalg.norm(self._group1_cm - self._group3_cm)
                self._d23 = np.linalg.norm(self._group2_cm - self._group3_cm)

            # evaluate pmf
            self._pmf += np.sum(
                self._total_pulling_force * self._pulling_direction
            ) * np.abs(self._velocity) * self._dt

        with self._state.cpu_local_snapshot as snapshot:
            group1_realidx = snapshot.particles.rtag[self._group1_idx]
            group2_realidx = snapshot.particles.rtag[self._group2_idx]
            group3_realidx = snapshot.particles.rtag[self._group3_idx]
        with self.cpu_local_force_arrays as arrays:
            arrays.force[group1_realidx, :] = self._group1_force
            arrays.force[group2_realidx, :] = self._group2_force
            arrays.force[group3_realidx, :] = self._group3_force

    def _integrate_spring_forward(self):
        self._spring_length += self._velocity * self._dt

    @log(category="scalar", is_property=False, requires_run=True)
    def t(self):
        if self._current_timestep is None:
            return 0
        return self._current_timestep

    @log(category="scalar", is_property=False, requires_run=True)
    def spring_length(self):
        return self._spring_length

    @log(category="scalar", is_property=False, requires_run=True)
    def r1_deviation(self):
        return self._dd1_devi

    @log(category="scalar", is_property=False, requires_run=True)
    def r2_deviation(self):
        return self._dd2_devi
    
    @log(category="scalar", is_property=False, requires_run=True)
    def pull_force_x(self):
        return self._total_pulling_force[0]

    @log(category="scalar", is_property=False, requires_run=True)
    def pull_force_y(self):
        return self._total_pulling_force[1]

    @log(category="scalar", is_property=False, requires_run=True)
    def pull_force_z(self):
        return self._total_pulling_force[2]

    @log(category="scalar", is_property=False, requires_run=True)
    def total_pull_force(self):
        return (self._total_pulling_force * self._pulling_direction).sum()
    
    @log(category="scalar", is_property=False, requires_run=True)
    def d3c(self):
        return self._d3c
    
    @log(category="scalar", is_property=False, requires_run=True)
    def d13(self):
        return self._d13
    
    @log(category="scalar", is_property=False, requires_run=True)
    def d23(self):
        return self._d23
    
    @log(category="scalar", is_property=False, requires_run=True)
    def work(self):
        return self._pmf

    def get_writer(self, log_fn, dump_period, mode="w"):
        logger = hoomd.logging.Logger(categories=["scalar"])
        logger.add(
            self,
            quantities=[
                "t",
                "r1_deviation",
                "r2_deviation",
                "pull_force_x",
                "pull_force_y",
                "pull_force_z",
                "total_pull_force",
                "spring_length",
                "d3c",
                "work"
            ],
            user_name="smd"
        )
        return hoomd.write.Table(
            output=open(log_fn, mode=mode, newline="\n"),
            trigger=hoomd.trigger.Periodic(dump_period),
            logger=logger,
        )

    def restart_sim(self, restart_fn):
        data = np.genfromtxt(restart_fn, skip_header=1)[-1, :]
        spring_length = data[2]
        pmf = data[-1]

        self._spring_length = spring_length
        self._pmf = pmf
        return
