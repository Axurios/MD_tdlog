import os
import e3x
import optax
import warnings
import functools
import numpy as np
import urllib.request
import flax.linen as nn
import flax 
import jax
import jax.numpy as jnp
warnings.simplefilter(action='ignore', category=FutureWarning)
from config_managers import H5Manager, XMLManager
import copy

print("\n=================== JAX DEVICE CHECK ===================")
print("Available devices:", jax.devices())
print("Default backend:", jax.default_backend())
print("========================================================\n")


results = {}
results_path = './result.xml'

results["device"] = jax.devices()

## --- Environment setup ---
#new_paths = "/gpfslocalsup/pub/anaconda-py3/2023.09/condabin:/gpfslocalsys/cuda/12.2.0/samples:/gpfslocalsys/cuda/12.2.0/nvvm/bin:/gpfslocalsys/cuda/12.2.0/bin:/gpfslocalsup/spack_soft/environment-modules/4.3.1/gcc-4.8.5-ism7cdy4xverxywj27jvjstqwk5oxe2v/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/c3/bin:/usr/lpp/mmfs/bin:/sbin:/bin:/gpfslocalsys/slurm/current/bin:/gpfslocalsup/bin:/gpfslocalsys/bin"
#current_path = os.environ.get('PATH', '')
#os.environ['PATH'] = new_paths + ':' + current_path
#print("PATH:", os.environ['PATH'])

# --- Download dataset ---
filename = "md17_ethanol.npz"
if not os.path.exists(filename):
    print(f"Downloading {filename} (this may take a while)...")
    urllib.request.urlretrieve(f"http://www.quantum-machine.org/gdml/data/npz/{filename}", filename)

# --- Hyperparameters ---
hyperparams = {
    "features" : 32,
    "max_degree" : 2,
    "num_iterations" : 3,
    "num_basis_functions" : 32,
    "cutoff" : 3.0,

    "num_train" : 200,
    "num_valid" : 25,
    "num_epochs" : 20,  # short for testing; increase as needed
    "learning_rate" : 0.01,
    "forces_weight" : 1.0,
    "batch_size" : 20
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

results["hyperparameters"] = hyperparams



kb_in_eV_pre_K = 8.617333262145e-5  # Boltzmann constant in eV/K
beta = 1.0/(300.0 * kb_in_eV_pre_K)  # inverse thermal energy for calibration in 1/eV
kcal_to_ev = 0.0433641153088  # conversion factor from kcal/mol to eV

# prepare the dataset
# This function prepares the training and validation datasets.
# It randomly selects num_train + num_valid samples from the dataset and splits them into training and validation sets.
# The mean energy is computed from the training set and used to center the energies.
# The energies and forces are converted from kcal/mol to eV.
# The atomic numbers and positions are converted to JAX arrays.
def prepare_datasets(key, num_train, num_valid):
    kcal_to_ev = 0.0433641153088
    dataset = np.load(filename)  # still NumPy, that's fine

    num_data = len(dataset['E'])
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f'dataset only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}')

    choice = jax.random.choice(key, num_data, shape=(num_draw,), replace=False)
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]

    # Convert mean to JAX (if you want full device consistency)
    mean_energy = jnp.mean(jnp.asarray(dataset['E'][train_choice]))
    #mean_energy= mean_energy - mean_energy 
    train_data = dict(
        energy=jnp.asarray(dataset['E'][train_choice, 0] - mean_energy) * kcal_to_ev,
        forces=jnp.asarray(dataset['F'][train_choice]) * kcal_to_ev,
        atomic_numbers=jnp.asarray(dataset['z']),
        positions=jnp.asarray(dataset['R'][train_choice]),
    )
    valid_data = dict(
        energy=jnp.asarray(dataset['E'][valid_choice, 0] - mean_energy) * kcal_to_ev,
        forces=jnp.asarray(dataset['F'][valid_choice]) * kcal_to_ev,
        atomic_numbers=jnp.asarray(dataset['z']),
        positions=jnp.asarray(dataset['R'][valid_choice]),
    )
    return train_data, valid_data, mean_energy

def prepare_calibration_dataset(filename, mean_energy, num_calib=200):
    kcal_to_ev = 0.0433641153088  # conversion factor from kcal/mol to eV
    dataset = np.load(filename)
    
    num_data = len(dataset['E'])
    if num_calib > num_data:
     raise RuntimeError(
         f'dataset only contains {num_data} points, requested num_calib={num_calib}')
    
    calib_data = dict(
        energy=jnp.asarray(dataset['E'][-num_calib:, 0] - mean_energy)*kcal_to_ev,
        forces=jnp.asarray(dataset['F'][-num_calib:])*kcal_to_ev,
        atomic_numbers=jnp.asarray(dataset['z'][-num_calib:]),
        positions=jnp.asarray(dataset['R'][-num_calib:])
    )
    return calib_data




class MessagePassingModel(nn.Module):
    features: int = features
    max_degree: int = max_degree
    num_iterations: int = num_iterations
    num_basis_functions: int = num_basis_functions
    cutoff: float = cutoff
    max_atomic_number: int = 118

    def energy(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        # --- Ensure all inputs are JAX arrays ---
        atomic_numbers = jnp.asarray(atomic_numbers)
        positions = jnp.asarray(positions)
        dst_idx = jnp.asarray(dst_idx)
        src_idx = jnp.asarray(src_idx)
        batch_segments = jnp.asarray(batch_segments)
        #batch_size = jnp.asarray(batch_size)

        # 1. Compute displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst

        # 2. Expand in basis functions.
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )

        # 3. Embed atomic numbers.
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers)

        # 4. Message passing iterations.
        for i in range(self.num_iterations):
            if i == self.num_iterations - 1:
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                y = e3x.nn.add(x, y)
                y = e3x.nn.Dense(self.features)(y)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)
                x = e3x.nn.add(x, y)

        # 5. Predict atomic energies.
        #CPUvsGPU 
        element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number + 1,))
        #element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros, name="theta")(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))
        #atomic_energies += element_bias[atomic_numbers]
        atomic_energies += jnp.take(element_bias, atomic_numbers, axis=0)

        # 6. Sum to total energy using modern API.
        #energy = jax.lax.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)
        #energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)
        energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)


        return -jnp.sum(energy), energy
    # This method extracts the molecule-level descriptor.
    @nn.compact
    def extract_descriptor(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        
        # --- Ensure inputs are jnp arrays (optional but safe) ---
        atomic_numbers = jnp.asarray(atomic_numbers)
        positions = jnp.asarray(positions)
        dst_idx = jnp.asarray(dst_idx)
        src_idx = jnp.asarray(src_idx)
        batch_segments = jnp.asarray(batch_segments)
        batch_size = jnp.asarray(batch_size)


        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(atomic_numbers)
        for i in range(self.num_iterations):
            if i == self.num_iterations-1:
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                y = e3x.nn.add(x, y)
                y = e3x.nn.Dense(self.features)(y)
                y = e3x.nn.silu(y)
                y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)
                x = e3x.nn.add(x, y)
        # Aggregate atomic features to form a molecule-level descriptor.
        # CPU vs GPU 
        descriptor = jax.ops.segment_sum(x, segment_ids=batch_segments, num_segments=batch_size)
        #descriptor = jax.lax.segment_sum(x, segment_ids=batch_segments, num_segments=batch_size)
        return descriptor

    # New method: extract descriptor and its gradient with respect to positions.
    @nn.compact
    def extract_descriptor_and_gradient(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        # --- Cast everything safely to JAX arrays ---
        atomic_numbers = jnp.asarray(atomic_numbers)
        positions = jnp.asarray(positions)
        dst_idx = jnp.asarray(dst_idx)
        src_idx = jnp.asarray(src_idx)
        batch_segments = jnp.asarray(batch_segments)
        batch_size = jnp.asarray(batch_size)
    
        descriptor = self.extract_descriptor(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
        
        pos_shape = positions.shape
        pos_flat = positions.reshape(-1)
    
        variables = self.scope.variables()
    
        def desc_fn(pos_flat):
            pos_reshaped = pos_flat.reshape(pos_shape)
            mod = self
            desc = mod.apply(
                variables,
                atomic_numbers,
                pos_reshaped,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                method=mod.extract_descriptor
            )
            return jnp.ravel(desc[0])
    
        descriptor_dim = descriptor.shape[-1]
        G = jax.vmap(lambda i: jax.grad(lambda x: desc_fn(x)[i])(pos_flat))(jnp.arange(descriptor_dim))
    
        return descriptor, G

    # This method computes forces using the energy function.
    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
        # --- Always cast to JAX arrays to ensure GPU ---
        atomic_numbers = jnp.asarray(atomic_numbers)
        positions = jnp.asarray(positions)
        dst_idx = jnp.asarray(dst_idx)
        src_idx = jnp.asarray(src_idx)
    
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
        else:
            batch_segments = jnp.asarray(batch_segments)
            #batch_size = jnp.asarray(batch_size)
    
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
        (_, energy), forces = energy_and_forces(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
    
        return energy, forces

        # ////////////////////////////////////////////////////////////////////////////////////////////////////////
        # element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
        # energy += element_bias[atomic_numbers] 
        # return energy, forces

# ———————— vectorized per‐sample calibration snippet ————————

def one_calib(params, model, pos, force, atomic_numbers, beta):
    # build per‐sample segment indices
    num_atoms = pos.shape[0]
    batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

    # extract descriptor + gradient for this one snapshot
    descriptor, G = model.apply(
        params,
        atomic_numbers,
        pos,
        dst_idx,
        src_idx,
        batch_segments,
        1,
        method=model.extract_descriptor_and_gradient
    )

    f = -force.reshape(-1)
    T = beta**2 * (G @ G.T)
    c = beta**2 * (G @ f)
    return T, c

# vmap over all calibration samples in one go:
v_one = jax.vmap(
    one_calib,
    in_axes=(None, None, 0, 0, None, None),
    out_axes=(0, 0),
)

# --- Batch preparation, loss, and training functions ---
def prepare_batches(key, data, batch_size):
    # --- Force data to JAX arrays for full GPU safety ---
    data = {k: jnp.asarray(v) for k, v in data.items()}

    data_size = len(data['energy'])
    steps_per_epoch = data_size // batch_size

    perms = jax.random.permutation(key, data_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    num_atoms = len(data['atomic_numbers'])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    atomic_numbers = jnp.tile(data['atomic_numbers'], batch_size)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)

    return [
        dict(
            energy=data['energy'][perm],
            forces=data['forces'][perm].reshape(-1, 3),
            atomic_numbers=atomic_numbers,
            positions=data['positions'][perm].reshape(-1, 3),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
        )
        for perm in perms
    ]



def mean_squared_loss(energy_prediction, energy_target, forces_prediction, forces_target, forces_weight):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target))
    return energy_loss + forces_weight * forces_loss

def mean_absolute_error(prediction, target):
    return jnp.mean(jnp.abs(prediction - target))


@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update', 'batch_size'))
def train_step(model_apply, optimizer_update, batch, batch_size, forces_weight, opt_state, params):
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
        loss = mean_squared_loss(energy, batch['energy'], forces, batch['forces'], forces_weight)
        return loss, (energy, forces)
    (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    energy_mae = mean_absolute_error(energy, batch['energy'])
    forces_mae = mean_absolute_error(forces, batch['forces'])
    return params, opt_state, loss, energy_mae, forces_mae

@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update', 'batch_size'))
def train_step(model_apply, optimizer_update, batch, batch_size, forces_weight, opt_state, params):
    # Sécurise les données : tout en jnp
    batch = {k: jnp.asarray(v) for k, v in batch.items()}

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
        loss = mean_squared_loss(energy, batch['energy'], forces, batch['forces'], forces_weight)
        return loss, (energy, forces)

    (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    energy_mae = mean_absolute_error(energy, batch['energy'])
    forces_mae = mean_absolute_error(forces, batch['forces'])
    return params, opt_state, loss, energy_mae, forces_mae

@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size'))
def eval_step(model_apply, batch, batch_size, forces_weight, params):
    # Convertir batch en jnp si besoin
    batch = {k: jnp.asarray(v) for k, v in batch.items()}

    energy, forces = model_apply(
        params,
        atomic_numbers=batch['atomic_numbers'],
        positions=batch['positions'],
        dst_idx=batch['dst_idx'],
        src_idx=batch['src_idx'],
        batch_segments=batch['batch_segments'],
        batch_size=batch_size
    )

    loss = mean_squared_loss(energy, batch['energy'], forces, batch['forces'], forces_weight)
    energy_mae = mean_absolute_error(energy, batch['energy'])
    forces_mae = mean_absolute_error(forces, batch['forces'])

    return loss, energy_mae, forces_mae



def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, forces_weight, batch_size):
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)

    # Assure-toi que tout est bien sur GPU
    atomic_numbers = jnp.asarray(train_data['atomic_numbers'])
    positions = jnp.asarray(train_data['positions'][0])
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atomic_numbers))

    # Initialisation GPU-safe
    params = model.init(init_key,
                        atomic_numbers=atomic_numbers,
                        positions=positions,
                        dst_idx=dst_idx,
                        src_idx=src_idx)
    opt_state = optimizer.init(params)

    # Validation pré-préparée (optionnel mais ok)
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0

        for i, batch in enumerate(train_batches):
            params, opt_state, loss, energy_mae, forces_mae = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                forces_weight=forces_weight,
                opt_state=opt_state,
                params=params
            )
            train_loss += (loss - train_loss) / (i+1)
            train_energy_mae += (energy_mae - train_energy_mae) / (i+1)
            train_forces_mae += (forces_mae - train_forces_mae) / (i+1)
            
            
            

        valid_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0

        for i, batch in enumerate(valid_batches):
            loss, energy_mae, forces_mae = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                forces_weight=forces_weight,
                params=params
            )
            valid_loss += (loss - valid_loss) / (i+1)
            valid_energy_mae += (energy_mae - valid_energy_mae) / (i+1)
            valid_forces_mae += (forces_mae - valid_forces_mae) / (i+1)

        if epoch % 50 == 0:
            print(f"epoch: {epoch: 3d}    train loss: {train_loss:8.3f}   valid loss: {valid_loss:8.3f}")
            print(f"    energy mae: {train_energy_mae:8.3f}   valid energy mae: {valid_energy_mae:8.3f}")
            print(f"    forces mae: {train_forces_mae:8.3f}   valid forces mae: {valid_forces_mae:8.3f}")

    return params
#jax.jit_version def train_model(key, model, train_data, valid_data,
#jax.jit_version                 num_epochs, learning_rate, forces_weight, batch_size):
#jax.jit_version     key, init_key = jax.random.split(key)
#jax.jit_version     optimizer = optax.adam(learning_rate)
#jax.jit_version 
#jax.jit_version     # GPU‐safe init (same as before) …
#jax.jit_version     atomic_numbers = jnp.asarray(train_data['atomic_numbers'])
#jax.jit_version     positions       = jnp.asarray(train_data['positions'][0])
#jax.jit_version     dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atomic_numbers))
#jax.jit_version     params   = model.init(init_key, atomic_numbers, positions, dst_idx, src_idx)
#jax.jit_version     opt_state = optimizer.init(params)
#jax.jit_version 
#jax.jit_version     # Pre‐compute validation batches once (same as before) …
#jax.jit_version     key, shuffle_key = jax.random.split(key)
#jax.jit_version     valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)
#jax.jit_version 
#jax.jit_version     # ——— Define epoch_step inside so it “sees” model & optimizer ———
#jax.jit_version     def epoch_step(carry, batch):
#jax.jit_version         params, opt_state = carry
#jax.jit_version         params, opt_state, loss, _, _ = train_step(
#jax.jit_version             model_apply     = model.apply,
#jax.jit_version             optimizer_update= optimizer.update,
#jax.jit_version             batch           = batch,
#jax.jit_version             batch_size      = batch_size,
#jax.jit_version             forces_weight   = forces_weight,
#jax.jit_version             opt_state       = opt_state,
#jax.jit_version             params          = params
#jax.jit_version         )
#jax.jit_version         return (params, opt_state), loss
#jax.jit_version 
#jax.jit_version     # ——— jit‐compile the scan over your list of batches ———
#jax.jit_version     @jax.jit
#jax.jit_version     def run_epoch(params, opt_state, batches):
#jax.jit_version         (params, opt_state), losses = jax.lax.scan(
#jax.jit_version             epoch_step,
#jax.jit_version             (params, opt_state),
#jax.jit_version             batches
#jax.jit_version         )
#jax.jit_version         return params, opt_state, losses.mean()
#jax.jit_version 
#jax.jit_version     # ——— Main training loop ———
#jax.jit_version     for epoch in range(1, num_epochs + 1):
#jax.jit_version         key, shuffle_key = jax.random.split(key)
#jax.jit_version         train_batches = prepare_batches(shuffle_key, train_data, batch_size)
#jax.jit_version 
#jax.jit_version         # stack list-of-dicts → dict-of-arrays along leading axis
#jax.jit_version         train_batches = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *train_batches)
#jax.jit_version 
#jax.jit_version         # Run the entire epoch in one JIT’d scan
#jax.jit_version         params, opt_state, train_loss = run_epoch(params, opt_state, train_batches)
#jax.jit_version 
#jax.jit_version         # (leave your validation loop unchanged)
#jax.jit_version         valid_loss = valid_energy_mae = valid_forces_mae = 0.0
#jax.jit_version         for i, batch in enumerate(valid_batches):
#jax.jit_version             loss, e_mae, f_mae = eval_step(
#jax.jit_version                 model_apply     = model.apply,
#jax.jit_version                 batch           = batch,
#jax.jit_version                 batch_size      = batch_size,
#jax.jit_version                 forces_weight   = forces_weight,
#jax.jit_version                 params          = params
#jax.jit_version             )
#jax.jit_version             valid_loss       += (loss - valid_loss)/(i+1)
#jax.jit_version             valid_energy_mae += (e_mae - valid_energy_mae)/(i+1)
#jax.jit_version             valid_forces_mae += (f_mae - valid_forces_mae)/(i+1)
#jax.jit_version 
#jax.jit_version         if epoch % 10 == 0:
#jax.jit_version             print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  valid loss {valid_loss:.4f}")
#jax.jit_version 
#jax.jit_version     return params






def calibrate_model(params, model, dataset, beta=beta):
    """
    For each sample in the calibration dataset:
      1. Extract descriptor and its gradient using extract_descriptor_and_gradient.
      2. Compute T_sample = beta^2 * (G @ G.T) and c_sample = beta^2 * (G @ (-force)).
    Then average and solve T theta = c.
    """
    # Sécurise l'entrée : tout en jnp
    dataset = {k: jnp.asarray(v) for k, v in dataset.items()}

    T_accum = 0.0
    c_accum = 0.0
    num_samples = dataset['positions'].shape[0]

    for i in range(num_samples):
        pos = dataset['positions'][i]           # (num_atoms, 3)
        force = dataset['forces'][i]            # (num_atoms, 3)
        atomic_numbers = dataset['atomic_numbers']  # (num_atoms,)

        num_atoms = pos.shape[0]
        batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
        batch_size = 1
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

        descriptor, G = model.apply(
            params,
            atomic_numbers,
            pos,
            dst_idx,
            src_idx,
            batch_segments,
            batch_size,
            method=model.extract_descriptor_and_gradient
        )

        f = -force.reshape(-1)
        T_sample = beta**2 * (G @ G.T)
        c_sample = beta**2 * (G @ f)

        T_accum += T_sample
        c_accum += c_sample

    T_avg = T_accum / num_samples
    c_avg = c_accum / num_samples

    theta_star = jnp.linalg.solve(T_avg, c_avg)

    # print("Calibration complete: theta_star =", theta_star)
    return theta_star


def calibrated_energy(params, theta, model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
    # Sécurise les entrées en jnp
    atomic_numbers = jnp.asarray(atomic_numbers)
    positions = jnp.asarray(positions)
    dst_idx = jnp.asarray(dst_idx)
    src_idx = jnp.asarray(src_idx)
    batch_segments = jnp.asarray(batch_segments)
    theta = jnp.asarray(theta)

    # Get descriptor from model
    descriptor = model.apply(
        params,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments,
        batch_size,
        method=model.extract_descriptor
    )

    # Get element bias from params
    element_bias = params['params']['element_bias']

    # Compute energy from the descriptor and theta
    energy_from_descriptor = jnp.sum(descriptor * theta, axis=-1)

    # Add element bias contribution if available
    if element_bias is not None and atomic_numbers is not None:
        atomic_biases = element_bias[atomic_numbers]
        #energy_from_bias = jax.lax.segment_sum(atomic_biases, segment_ids=batch_segments, num_segments=batch_size)
        energy_from_bias = jax.ops.segment_sum(atomic_biases, segment_ids=batch_segments, num_segments=batch_size)
        return energy_from_descriptor + energy_from_bias

    return energy_from_descriptor




def calibrated_forces(params, theta, model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
    # Sécurise toutes les entrées
    atomic_numbers = jnp.asarray(atomic_numbers)
    positions = jnp.asarray(positions)
    dst_idx = jnp.asarray(dst_idx)
    src_idx = jnp.asarray(src_idx)
    batch_segments = jnp.asarray(batch_segments)
    theta = jnp.asarray(theta)

    def energy_fn(pos):
        return calibrated_energy(params, theta, model, atomic_numbers, pos, dst_idx, src_idx, batch_segments, batch_size).sum()

    forces = -jax.grad(energy_fn)(positions)
    return forces







# --- Main execution ---
# 1. Prepare dataset
data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)
train_data, valid_data, mean_energy = prepare_datasets(data_key, num_train=num_train, num_valid=num_valid)
# print(f"Mean energy (shift applied): {mean_energy:.6f} kcal/mol")

# 2. Initialize and train the message-passing model
message_passing_model = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
)

params = train_model(
    key=train_key,
    model=message_passing_model,
    train_data=train_data,
    valid_data=valid_data,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    forces_weight=forces_weight,
    batch_size=batch_size,
)

# 3. Save bare (uncalibrated) model parameters
with open("before_model_params.bin", "wb") as f:
    f.write(flax.serialization.to_bytes(params))
print("Bare model parameters saved to 'before_model_params.bin'.")

theta_basic = params["params"]["theta"]['kernel'].flatten()
results["theta_basic"] = theta_basic
results["params"] = params



# 4. Prepare calibration dataset
calib_data = prepare_calibration_dataset(filename, mean_energy=mean_energy, num_calib=100)
# Run calibration
#theta_star = calibrate_model(params, message_passing_model, calib_data, beta=beta)
# --- vectorized calibration call ---
T_samples, c_samples = v_one(
    params,
    message_passing_model,
    calib_data['positions'],      # shape [n_calib, natoms, 3]
    calib_data['forces'],         # shape [n_calib, natoms, 3]
    calib_data['atomic_numbers'], # shape [natoms]
    beta
)
# compute averages and solve
T_avg = jnp.mean(T_samples, axis=0)
c_avg = jnp.mean(c_samples, axis=0)
theta_star = jnp.linalg.solve(T_avg, c_avg)

results["theta_fisher"] = theta_star

# 1. Copy and inject theta_star into params
params_fisher = copy.deepcopy(params)
params_fisher["params"]["theta"]['kernel'] = theta_star.reshape(-1, 1)  # Ensure shape (N,1) for compatibility
with open("fisher_model_params.bin", "wb") as f:
    f.write(flax.serialization.to_bytes(params_fisher))
print("Fisher-calibrated model parameters saved to 'fisher_model_params.bin'.")




##############
# theta projected unto ker D (for a calib data only)
sample = calib_data #valid_data
num_atoms = sample['positions'].shape[1]
batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
batch_size = 1
dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
atomic_numbers = sample['atomic_numbers']

### Fisher projected unto kerD
def extr_desc(positions):
    return jnp.squeeze(message_passing_model.apply(
    params,
    atomic_numbers,
    positions, 
    dst_idx,
    src_idx,
    batch_segments,
    batch_size,
    method = MessagePassingModel.extract_descriptor

))
batch_extr_desc = jax.vmap(extr_desc)

D = batch_extr_desc(sample['positions'])
U, S, Vt = jnp.linalg.svd(D,full_matrices=False)

T = theta_basic-theta_star
X = Vt[0:10] # !!!!!!!!!!!!!!!!!!! 10 arbitrary, might make it a variable ? !!!!!!!!!!!!
A = X.T@X@T - T
theta_mixed = theta_basic + A

params_mixed = copy.deepcopy(params)
params_mixed["params"]["theta"]['kernel'] = theta_mixed.reshape(-1, 1)
# that's it
with open("mixed_model_params.bin", "wb") as f:
    f.write(flax.serialization.to_bytes(params_mixed))
print("Mixed parameters saved.")
results["theta_mixed"] = theta_mixed







## validation computation
sample = valid_data
positions = sample['positions']             # shape: (N_samples, N_atoms, 3)
atomic_numbers = sample['atomic_numbers']   # shape: (N_atoms,)
num_atoms = atomic_numbers.shape[0]
batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
batch_size = 1
dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

# Function to apply model for a single sample
def predict_single(params, pos):
    energy, forces = message_passing_model.apply(
        params,
        atomic_numbers,
        pos,
        dst_idx,
        src_idx,
        batch_segments,
        batch_size
    )
    return energy, forces

# Vectorize over batch dimension (i.e., over samples in `positions`)
batched_predict_initial = jax.vmap(lambda pos: predict_single(params, pos), in_axes=0)
batched_predict_fisher = jax.vmap(lambda pos: predict_single(params_fisher, pos), in_axes=0)
batched_predict_mixed = jax.vmap(lambda pos: predict_single(params_mixed, pos), in_axes=0)

# Run predictions
initial_energies, initial_forces = batched_predict_initial(positions)
fisher_energies, fisher_forces = batched_predict_fisher(positions)
mixed_energies, mixed_forces = batched_predict_mixed(positions)

# Store in a results dictionary (or use a dataclass or struct if preferred)
validation = {
    "initial": {"energies": initial_energies, "forces": initial_forces},
    "fisher":  {"energies": fisher_energies,  "forces": fisher_forces},
    "mixed":   {"energies": mixed_energies,   "forces": mixed_forces}
}

## to see if above works, here a bit of code printing it (not entirely to be manageable)
# from pprint import pprint
# # Print energies and forces for the first 3 samples
# for key in ["initial", "fisher", "mixed"]:
#     print(f"\n--- {key.upper()} ---")
#     print("Energies (first 3):")
#     pprint(validation[key]["energies"][:3])
    
#     print("\nForces (first 3):")
#     pprint(validation[key]["forces"][:3])
results["validation"] = validation


xml_res = XMLManager(results_path, mode='writing')
xml_res.generate_xml(results)


















## can be deleted i believe
'''
## validation of the obtained theta (energy and forces computation on validation data)
# sample = valid_data
# num_atoms = sample['positions'].shape[1]
# batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
# batch_size = 1
# dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
# atomic_numbers = sample['atomic_numbers']
# # --- Test Predictions (Validation sample) ---
# positions = sample['positions'][0]


# # Predict with uncalibrated model
# initial_energy, initial_forces = message_passing_model.apply(
#     params,
#     atomic_numbers,
#     positions,
#     dst_idx,
#     src_idx,
#     batch_segments,
#     batch_size
# )
# results["initial_energy"] = initial_energy
# results["initial_forces"] = initial_forces



# # 2. Predict energy and forces with Fisher calibrated model
# fisher_energy, fisher_forces = message_passing_model.apply(
#     params_fisher,
#     atomic_numbers,
#     positions,
#     dst_idx,
#     src_idx,
#     batch_segments,
#     batch_size
# )
# results["fisher_energy"] = fisher_energy
# results["fisher_forces"] = fisher_forces



# mixed_energy, mixed_forces = message_passing_model.apply(
#     params_mixed,
#     atomic_numbers,
#     positions,
#     dst_idx,
#     src_idx,
#     batch_segments,
#     batch_size
# )
# results["mixed_energy"] = mixed_energy
# results["mixed_forces"] = mixed_forces

### ///////////////////////////////




# 4. Select another sample (here sample[1])
sample = valid_data
num_atoms = sample['positions'].shape[1]
batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
batch_size = 1
dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
atomic_numbers = sample['atomic_numbers']
positions = sample['positions'][1]  # Second molecule

# Uncalibrated model prediction
initial_energy, initial_forces = message_passing_model.apply(
    params,
    atomic_numbers,
    positions,
    dst_idx,
    src_idx,
    batch_segments,
    batch_size
)

# Calibrated model prediction
calib_energy = calibrated_energy(
    params,
    theta_star,
    message_passing_model,
    atomic_numbers,
    positions,
    dst_idx,
    src_idx,
    batch_segments,
    batch_size
)
calib_forces = calibrated_forces(
    params,
    theta_star,
    message_passing_model,
    atomic_numbers,
    positions,
    dst_idx,
    src_idx,
    batch_segments,
    batch_size
)

# Fisher-calibrated model prediction
fisher_energy, fisher_forces = message_passing_model.apply(
    params_fisher,
    atomic_numbers,
    positions,
    dst_idx,
    src_idx,
    batch_segments,
    batch_size
)

# 5. Display results
print("\n=== Model Comparison on Sample 1 ===")
print(f"Uncalibrated Energy: {initial_energy}")
print(f"Calibrated Energy: {calib_energy}")
print(f"Fisher-Calibrated Energy: {fisher_energy}")

print(f"\nUncalibrated Forces norm: {jnp.linalg.norm(initial_forces)}")
print(f"Calibrated Forces norm: {jnp.linalg.norm(calib_forces)}")
print(f"Fisher-Calibrated Forces norm: {jnp.linalg.norm(fisher_forces)}")



# # Predict with calibrated model
# calib_energy = calibrated_energy(params, theta_star, message_passing_model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
# calib_forces = calibrated_forces(params, theta_star, message_passing_model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)

# Print results
# print("\n=== Uncalibrated Model Predictions ===")
# print(f"Energy: {initial_energy}")
# print(f"Forces shape: {initial_forces.shape}")

# print("\n=== Calibrated Model Predictions ===")
# print(f"Calibrated Energy: {calib_energy}")
# print(f"Calibrated Forces shape: {calib_forces.shape}")

# basic_energy = calibrated_energy(params, theta_basic, message_passing_model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
# basic_forces = calibrated_forces(params, theta_basic, message_passing_model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
# print("\n=== Basic Model Predictions ===")
# print("Calibrated Energy:", basic_energy)
# print("Calibrated Forces:", basic_forces)

# --- Apply calibrated (Fisher) parameters ---
# 3. Save Fisher-calibrated parameters


# print("\n=== Fisher-Calibrated Model Predictions ===")
# print(f"Energy: {fisher_energy}")
# print(f"Forces shape: {fisher_forces.shape}")

# --- Test: Compare Uncalibrated vs Calibrated vs Fisher-Calibrated Models ---


#### not useful ???
# # Save calibrated parameters
# calibrated_params = {"model_params": params, "theta": theta_star}
# with open("calibrated_model_params.bin", "wb") as f:
#     f.write(flax.serialization.to_bytes(calibrated_params))
# print("Calibrated model parameters saved to 'calibrated_model_params.bin'.")
'''
