import hoomd
import gsd.hoomd
import numpy as np
import freud
from hoomd_chimes_addons import ChIMES_manybody

TIME_CONVERSION = 0.01  # From 1 fs to 100fs (derived time unit)
# ChIMES related
chimes_api_path = "/home/shihkual/chimes_calculator-myfork/serial_interface/api"
chimes_wrapper_path = "/home/shihkual/chimes_calculator-myfork/build/"
param_file = "./A4_params.txt"

traj_fn = "traj.gsd"
thermo_fn = "log.txt"
dump_period = 1000
total_steps = 100_000
segment = 5000
seed = 43

# Initialize a fcc 4x4x4 supercell
N_repeats = 4
lattice_const = 120.0
perturb_pos_std = 0.0

# Thermodynamic condition
dt = 5.0            # fs
T = 300             # K
tau = 500.0         # fs
kb = 0.00831446262  # kJ/mol/K

kT = kb*T
dt *= TIME_CONVERSION
tau *= TIME_CONVERSION

sc = freud.data.UnitCell.fcc()
random_system = sc.generate_system(N_repeats, lattice_const, perturb_pos_std, seed=seed)

snapshot = gsd.hoomd.Frame()
snapshot.particles.N = random_system[1].shape[0]
snapshot.particles.position = random_system[1]
snapshot.particles.types = ['A']
snapshot.configuration.box = [random_system[0].Lx, random_system[0].Ly, random_system[0].Lz, 0.0, 0.0, 0.0]
snapshot.particles.mass = [67256.0] * random_system[1].shape[0]
snapshot.configuration.step = 0

with gsd.hoomd.open(name='lattice.gsd', mode='w') as f:
    f.append(snapshot)

device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=seed)
sim.timestep = 0
sim.create_state_from_gsd(filename='lattice.gsd', frame=-1)

forces = []

chimes = ChIMES_manybody(
    sim=sim,
    chimes_api_path=chimes_api_path,
    chimes_wrapper_path=chimes_wrapper_path,
    param_file_path=param_file,
    N_particles=random_system[1].shape[0]
)
forces.append(chimes)

ensemble = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(),
    thermostat=hoomd.md.methods.thermostats.MTTK(kT=kT, tau=tau)
)
integrator = hoomd.md.Integrator(
    dt=dt,
    methods=[ensemble],
    forces=forces,
    integrate_rotational_dof=False
)
sim.operations += integrator

gsd_writer = hoomd.write.GSD(
    filename=traj_fn,
    trigger=hoomd.trigger.Periodic(dump_period),
    dynamic=["property"],
    mode='wb'
)
sim.operations += gsd_writer

thermo_analyzer = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All()
)
sim.operations += thermo_analyzer

logger = hoomd.logging.Logger(categories=["scalar"])
logger.add(
    sim,
    quantities=["timestep"]
)
logger.add(
    thermo_analyzer,
    quantities=["kinetic_temperature", "pressure", "volume"]
)

thermo_writer =  hoomd.write.Table(
    output=open(thermo_fn, mode="w", newline="\n"),
    trigger=hoomd.trigger.Periodic(dump_period),
    logger=logger,
)
sim.operations += thermo_writer

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)
sim.run(0)
thermo_writer.write()

while sim.timestep < total_steps:
    sim.run(segment)
    print(sim.timestep, sim.tps)
    for writer in sim.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()
