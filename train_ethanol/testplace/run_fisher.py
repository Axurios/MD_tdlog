#!/usr/bin/env python
# coding: utf-8

#pip install ase
#pip install numpy==1.24.4
#pip install "numpy<2"
#pip install msgpack
#pip install msgpack_numpy



import io
import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
import ase.optimize as ase_opt
#import matplotlib.pyplot as plt
#import py3Dmol
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random 
import flax 
import os 
import numpy as np 
import functools
import optax
import urllib.request

from config_managers import H5Manager, XMLManager
#from analysis import TimeOfFailureAnalysis

print("\n=================== JAX DEVICE CHECK ===================")
print("Available devices:", jax.devices())
print("Default backend:", jax.default_backend())
print("========================================================\n")


run_results = {}
run_results_path = './run_result.xml'

hyperparams = {
  "features" : 32,
  "max_degree" : 1,
  "num_iterations" : 3,
  "num_basis_functions" : 32, #16,
  "cutoff" : 5.0,
  "run_num_train" : 900,
  "run_num_valid" : 100,
  "timestep_fs" : 1.0,
  "num_steps" : 400,
  "temperature" : 1000
}


base_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(base_dir, "hyperparams.xml")
# print(params_path)

xml_param = XMLManager(params_path, mode='reading')
# print(xml_param.parse_xml()) 
for key, value in xml_param.parse_xml().items():
    # print(f"{key}: {value}")
    if key in hyperparams:
        current_type = type(hyperparams[key])
        try:
            hyperparams[key] = current_type(value)
        except ValueError:
            print(f"Warning: Could not cast '{key}' to {current_type}. Skipping.")


for key, value in hyperparams.items():
    globals()[key] = value

run_results["hyperparams"] = hyperparams





def mean_squared_loss(energy_prediction, energy_target, forces_prediction, forces_target, forces_weight):
  energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target))
  forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target))
  return energy_loss + forces_weight * forces_loss

def mean_absolute_error(prediction, target):
  return jnp.mean(jnp.abs(prediction - target))



class MessagePassingModel(nn.Module):
  features: int = 32
  max_degree: int = 2
  num_iterations: int = 3
  num_basis_functions: int = 8
  cutoff: float = 5.0
  max_atomic_number: int = 118  # This is overkill for most applications.


  def energy(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
    
    # Ensure all inputs are JAX arrays and on the correct device
    atomic_numbers = jnp.asarray(atomic_numbers)
    positions = jnp.asarray(positions)
    dst_idx = jnp.asarray(dst_idx)
    src_idx = jnp.asarray(src_idx)
    batch_segments = jnp.asarray(batch_segments)
    #batch_size = jnp.asarray(batch_size)
    
    # 1. Calculate displacement vectors.
    positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
    positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
    displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

    # 2. Expand displacement vectors in basis functions.
    basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
      displacements,
      num=self.num_basis_functions,
      max_degree=self.max_degree,
      radial_fn=e3x.nn.reciprocal_bernstein,
      cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
    )

    # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
    x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(atomic_numbers)

    # 4. Perform iterations (message-passing + atom-wise refinement).
    for i in range(self.num_iterations):
      # Message-pass.
      if i == self.num_iterations-1:  # Final iteration.
        # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
        # features for efficiency reasons.
        y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
        # After the final message pass, we can safely throw away all non-scalar features.
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
      else:
        # In intermediate iterations, the message-pass should consider all possible coupling paths.
        y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
        y = e3x.nn.add(x, y)

        # Atom-wise refinement MLP.
        y = e3x.nn.Dense(self.features)(y)
        y = e3x.nn.silu(y)
        y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

        # Residual connection.
        x = e3x.nn.add(x, y)

    # 5. Predict atomic energies with an ordinary dense layer.
    element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
    atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros, name="theta")(x)  # (..., Natoms, 1, 1, 1)
    atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
    #atomic_energies += element_bias[atomic_numbers]
    atomic_energies += jnp.take(element_bias, atomic_numbers, axis=0)

    # 6. Sum atomic energies to obtain the total energy.
    energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)

    # To be able to efficiently compute forces, our model should return a single output (instead of one for each
    # molecule in the batch). Fortunately, since all atomic contributions only influence the energy in their own
    # batch segment, we can simply sum the energy of all molecules in the batch to obtain a single proxy output
    # to differentiate.
    return -jnp.sum(energy), energy  # Forces are the negative gradient, hence the minus sign.


  @nn.compact
  def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
    if batch_segments is None:
      batch_segments = jnp.zeros_like(atomic_numbers)
      batch_size = 1

    # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
    # jax.value_and_grad to create a function for predicting both energy and forces for us.
    energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
    (_, energy), forces = energy_and_forces(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)

    return energy, forces


@jax.jit
def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx, params):
    return MessagePassingModel.apply(params,
        atomic_numbers=jnp.asarray(atomic_numbers),
        positions=jnp.asarray(positions),
        dst_idx=jnp.asarray(dst_idx),
        src_idx=jnp.asarray(src_idx),
    )

"""
class MessagePassingCalculator(ase_calc.Calculator):
  implemented_properties = ["energy", "forces"]

  def calculate(self, atoms, properties, system_changes=ase.calculators.calculator.all_changes):
    ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

    # Compute dst/src index pairs
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))

    # Ensure everything is JAX arrays
    atomic_numbers = jnp.asarray(atoms.get_atomic_numbers())
    positions = jnp.asarray(atoms.get_positions())
    dst_idx = jnp.asarray(dst_idx)
    src_idx = jnp.asarray(src_idx)

    # Evaluate model
    energy, forces = evaluate_energies_and_forces(
      atomic_numbers=atomic_numbers,
      positions=positions,
      dst_idx=dst_idx,
      src_idx=src_idx
    )

    # Block until GPU calculation is finished and move to numpy for ASE
    energy_np = np.array(energy.block_until_ready()).item()
    forces_np = np.array(forces.block_until_ready())

    # Store in ASE format
    self.results['energy'] = energy_np * ase.units.kcal/ase.units.mol
    self.results['forces'] = forces_np * ase.units.kcal/ase.units.mol
"""



    
@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update', 'batch_size'))
def train_step(model_apply, optimizer_update, batch, batch_size, forces_weight, opt_state, params):
    # Force everything into JAX arrays (move to GPU if needed)
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    forces_weight = jnp.asarray(forces_weight)

    def loss_fn(params):
        energy, forces = model_apply(
            params,
            atomic_numbers=batch['atomic_numbers'],
            positions=batch['positions'],
            dst_idx=batch['dst_idx'],
            src_idx=batch['src_idx'],
            batch_segments=batch['batch_segments'],
            batch_size=batch_size
        )
        loss = mean_squared_loss(
            energy_prediction=energy,
            energy_target=batch['energy'],
            forces_prediction=forces,
            forces_target=batch['forces'],
            forces_weight=forces_weight
        )
        return loss, (energy, forces)
    
    (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    energy_mae = mean_absolute_error(energy, batch['energy'])
    forces_mae = mean_absolute_error(forces, batch['forces'])
    return params, opt_state, loss, energy_mae, forces_mae

@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size'))
def eval_step(model_apply, batch, batch_size, forces_weight, params):
    # Force everything to JAX arrays to ensure GPU
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    forces_weight = jnp.asarray(forces_weight)

    energy, forces = model_apply(
        params,
        atomic_numbers=batch['atomic_numbers'],
        positions=batch['positions'],
        dst_idx=batch['dst_idx'],
        src_idx=batch['src_idx'],
        batch_segments=batch['batch_segments'],
        batch_size=batch_size
    )
    loss = mean_squared_loss(
        energy_prediction=energy,
        energy_target=batch['energy'],
        forces_prediction=forces,
        forces_target=batch['forces'],
        forces_weight=forces_weight
    )
    energy_mae = mean_absolute_error(energy, batch['energy'])
    forces_mae = mean_absolute_error(forces, batch['forces'])
    return loss, energy_mae, forces_mae
    


# os.system('pwd')
# os.system('ls -ltr')





















#Load the data: # Download the dataset.
filename = "md17_ethanol.npz"
if not os.path.exists(filename):
  print(f"Downloading {filename} (this may take a while)...")
  urllib.request.urlretrieve(f"http://www.quantum-machine.org/gdml/data/npz/{filename}", filename)
  
def prepare_datasets(key, num_train, num_valid):
  # Load the dataset.
  dataset = np.load(filename)

  # Make sure that the dataset contains enough entries.
  num_data = len(dataset['E'])
  num_draw = num_train + num_valid
  if num_draw > num_data:
    raise RuntimeError(
      f'datasets only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}')

  # Randomly draw train and validation sets from dataset.
  choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
  train_choice = choice[:num_train]
  valid_choice = choice[num_train:]

  # Determine mean energy of the training set.
  mean_energy = np.mean(dataset['E'][train_choice])  # ~ -97000

  # Collect and return train and validation sets.
  train_data = dict(
    energy=jnp.asarray(dataset['E'][train_choice, 0] - mean_energy),
    forces=jnp.asarray(dataset['F'][train_choice]),
    atomic_numbers=jnp.asarray(dataset['z']),
    positions=jnp.asarray(dataset['R'][train_choice]),
  )
  valid_data = dict(
    energy=jnp.asarray(dataset['E'][valid_choice, 0] - mean_energy),
    forces=jnp.asarray(dataset['F'][valid_choice]),
    atomic_numbers=jnp.asarray(dataset['z']),
    positions=jnp.asarray(dataset['R'][valid_choice]),
  )
  return train_data, valid_data, mean_energy




# Create PRNGKeys.
key = jax.random.PRNGKey(0)
data_key, train_key = jax.random.split(key, 2)

# Draw training and validation sets.
train_data, valid_data, _ = prepare_datasets(data_key, num_train=run_num_train, num_valid=run_num_valid)
  



# import msgpack
import msgpack_numpy
# import pprint
# Use msgpack_numpy for proper NumPy serialization support
msgpack_numpy.patch()  # this allows msgpack to handle np arrays

# Re-initialize the model exactly as you did before training
message_passing_model = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
)

# Create a PRNGKey for initialization
key = random.PRNGKey(0)  # Use a key, the specific value is not crucial here

dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data['atomic_numbers']))
# SAFETY: convert everything to JAX arrays
atomic_numbers = jnp.asarray(train_data['atomic_numbers'])
positions = jnp.asarray(train_data['positions'][0])
dst_idx = jnp.asarray(dst_idx)
src_idx = jnp.asarray(src_idx)
# print("ok")

# Pack into a dictionary
dummy_input = dict(
    atomic_numbers=atomic_numbers,
    positions=positions,
    dst_idx=dst_idx,
    src_idx=src_idx,
)



dummy_params = message_passing_model.init(key, **dummy_input)
with open('before_model_params.bin', 'rb') as f:
     serialized_params = f.read()
default_params = flax.serialization.from_bytes(dummy_params, serialized_params)  
# print(params)  


with open('fisher_model_params.bin', 'rb') as f:
    fisher_serialized_params = f.read()
fisher_params = flax.serialization.from_bytes(dummy_params, fisher_serialized_params)  
# print(params)


with open('mixed_model_params.bin', 'rb') as f:
    mixed_serialized_params = f.read()
mixed_params = flax.serialization.from_bytes(dummy_params, mixed_serialized_params)  
# print(params)

params = fisher_params
@jax.jit
def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
    # print(params)
    return message_passing_model.apply(params,
        atomic_numbers=jnp.asarray(atomic_numbers),
        positions=jnp.asarray(positions),
        dst_idx=jnp.asarray(dst_idx),
        src_idx=jnp.asarray(src_idx),
    )

 

class MessagePassingCalculator(ase_calc.Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes=ase.calculators.calculator.all_changes):
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

        # Get connectivity
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))

        # Convert everything to JAX arrays
        atomic_numbers = jnp.asarray(atoms.get_atomic_numbers())
        positions = jnp.asarray(atoms.get_positions())
        dst_idx = jnp.asarray(dst_idx)
        src_idx = jnp.asarray(src_idx)

        # Evaluate energy and forces
        energy, forces = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=positions,
            dst_idx=dst_idx,
            src_idx=src_idx
        )

        # Convert back to numpy (because ASE expects NumPy)
        energy_np = np.array(energy.block_until_ready()).item()  # float
        forces_np = np.array(forces.block_until_ready())         # (Natoms, 3) array

        # Save in self.results! This is critical
        self.results = {
            "energy": energy_np * ase.units.kcal / ase.units.mol,
            "forces": forces_np * ase.units.kcal / ase.units.mol
        }








def run_md_simulation(params, tag):
    params = params
    atoms = ase.Atoms(train_data['atomic_numbers'], train_data['positions'][0])
    atoms.set_calculator(MessagePassingCalculator())

    # Structure optimization
    _ = ase_opt.BFGS(atoms).run(fmax=0.05)

    # Initial momenta
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    # Integrator
    integrator = VelocityVerlet(atoms, timestep=timestep_fs * ase.units.fs)

    # Storage
    frames = np.zeros((num_steps, len(atoms), 3))
    potential_energy = np.zeros(num_steps)
    kinetic_energy = np.zeros(num_steps)
    total_energy = np.zeros(num_steps)

    for i in range(num_steps):
        integrator.run(1)
        frames[i] = atoms.get_positions()
        potential_energy[i] = atoms.get_potential_energy()
        kinetic_energy[i] = atoms.get_kinetic_energy()
        total_energy[i] = atoms.get_total_energy()
        if i % 1000 == 0:
            print(f"[{tag}] step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}")

    # Export results
    tag_results = {
        "time": np.arange(num_steps)*timestep_fs,
        "frames": frames,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "total_energy": total_energy
    }
    return tag_results









"""
params = fisher_params
atoms = ase.Atoms(train_data['atomic_numbers'], train_data['positions'][0])
atoms.set_calculator(MessagePassingCalculator())

# Run structure optimization with BFGS.
_ = ase_opt.BFGS(atoms).run(fmax=0.05)

# Parameters.
temperature = 1000
# defined above
# timestep_fs = 1.0
# num_steps = 400


# Draw initial momenta.
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
Stationary(atoms)  # Remove center of mass translation.
ZeroRotation(atoms)  # Remove rotations.

# Initialize Velocity Verlet integrator.
integrator = VelocityVerlet(atoms, timestep=timestep_fs*ase.units.fs)

# Run molecular dynamics.
frames = np.zeros((num_steps, len(atoms), 3))
potential_energy = np.zeros((num_steps,))
kinetic_energy = np.zeros((num_steps,))
total_energy = np.zeros((num_steps,))
for i in range(num_steps):
  # Run 1 time step.
  integrator.run(1)
  # Save current frame and keep track of energies.
  frames[i] = atoms.get_positions()
  potential_energy[i] = atoms.get_potential_energy()
  kinetic_energy[i] = atoms.get_kinetic_energy()
  total_energy[i] = atoms.get_total_energy()
  # Occasionally print progress.
  if i % 100 == 0:
    print(f"step {i:5d} epot {potential_energy[i]: 5.3f} ekin {kinetic_energy[i]: 5.3f} etot {total_energy[i]: 5.3f}")
"""


# export the results :
#time = np.arange(num_steps) * timestep_fs
# run_results["time"] = time
# run_results["frames"] = frames
# run_results["potential_energy"] = potential_energy
# run_results["kinetic_energy"] = kinetic_energy
# run_results["total_energy"] = total_energy


run_results["fisher_results"] = run_md_simulation(fisher_params, tag="fisher")
#run_results["mixed_results"] = run_md_simulation(mixed_params, tag="mixed")
#run_results["default_results"] = run_md_simulation(default_params, tag="default")


xml_res = XMLManager(run_results_path, mode='writing')
xml_res.generate_xml(run_results)



#import matplotlib.pyplot as plt
#import numpy as np
#
## Safe matplotlib inline if needed
#try:
#    get_ipython().run_line_magic('matplotlib', 'inline')
#except NameError:
#    pass

## Create the plot
#plt.figure(figsize=(8, 6))
#plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#plt.xlabel('time [fs]')
#plt.ylabel('energy [eV]')
#time = np.arange(num_steps) * timestep_fs
#plt.plot(time, potential_energy, label='potential energy')
#plt.plot(time, kinetic_energy, label='kinetic energy')
#plt.plot(time, total_energy, label='total energy')
#plt.legend()
#plt.grid()
#
## Save the figure
#plt.savefig('energy_vs_time.png', dpi=300, bbox_inches='tight')  # high resolution and nice margins



