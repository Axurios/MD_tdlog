import functools
import os
import urllib.request
import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Environment setup ---
new_paths = "/gpfslocalsup/pub/anaconda-py3/2023.09/condabin:/gpfslocalsys/cuda/12.2.0/samples:/gpfslocalsys/cuda/12.2.0/nvvm/bin:/gpfslocalsys/cuda/12.2.0/bin:/gpfslocalsup/spack_soft/environment-modules/4.3.1/gcc-4.8.5-ism7cdy4xverxywj27jvjstqwk5oxe2v/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/c3/bin:/usr/lpp/mmfs/bin:/sbin:/bin:/gpfslocalsys/slurm/current/bin:/gpfslocalsup/bin:/gpfslocalsys/bin"
current_path = os.environ.get('PATH', '')
os.environ['PATH'] = new_paths + ':' + current_path
print("PATH:", os.environ['PATH'])

# --- Download dataset ---
filename = "md17_ethanol.npz"
if not os.path.exists(filename):
    print(f"Downloading {filename} (this may take a while)...")
    urllib.request.urlretrieve(f"http://www.quantum-machine.org/gdml/data/npz/{filename}", filename)

# --- Hyperparameters ---
features = 32
max_degree = 2
num_iterations = 3
num_basis_functions = 32
cutoff = 3.0

num_train = 200
num_valid = 25
num_epochs = 20  # short for testing; increase as needed
learning_rate = 0.01
forces_weight = 1.0
batch_size = 20
kb_in_eV_pre_K = 8.617333262145e-5  # Boltzmann constant in eV/K
beta = 1.0/(300.0 * kb_in_eV_pre_K)  # inverse thermal energy for calibration in 1/eV
kcal_to_ev = 0.0433641153088  # conversion factor from kcal/mol to eV

# --- Data preparation function ---
def prepare_datasets(key, num_train, num_valid):
    kcal_to_ev = 0.0433641153088  # conversion factor from kcal/mol to eV
    dataset = np.load(filename)
    num_data = len(dataset['E'])
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f'dataset only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}')
    choice = np.asarray(jax.random.choice(key, num_data, shape=(num_draw,), replace=False))
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]
    mean_energy = np.mean(dataset['E'][train_choice])
    train_data = dict(
        energy=jnp.asarray(dataset['E'][train_choice, 0] - mean_energy)*kcal_to_ev,
        forces=jnp.asarray(dataset['F'][train_choice])*kcal_to_ev,
        atomic_numbers=jnp.asarray(dataset['z']),
        positions=jnp.asarray(dataset['R'][train_choice]),
    )
    valid_data = dict(
        energy=jnp.asarray(dataset['E'][valid_choice, 0] - mean_energy)*kcal_to_ev,
        forces=jnp.asarray(dataset['F'][valid_choice])*kcal_to_ev,
        atomic_numbers=jnp.asarray(dataset['z']),
        positions=jnp.asarray(dataset['R'][valid_choice]),
    )
    return train_data, valid_data, mean_energy

def prepare_calibration_dataset(filename, mean_energy, num_calib=200):
    kcal_to_ev = 0.0433641153088  # conversion factor from kcal/mol to eV
    dataset = np.load(filename)
    calib_data = dict(
        energy=jnp.asarray(dataset['E'][-num_calib:, 0] - mean_energy)*kcal_to_ev,
        forces=jnp.asarray(dataset['F'][-num_calib:])*kcal_to_ev,
        atomic_numbers=jnp.asarray(dataset['z']),
        positions=jnp.asarray(dataset['R'][-num_calib:])
    )
    return calib_data

# --- MessagePassingModel definition ---
class MessagePassingModel(nn.Module):
    features: int = features
    max_degree: int = max_degree
    num_iterations: int = num_iterations
    num_basis_functions: int = num_basis_functions
    cutoff: float = cutoff
    max_atomic_number: int = 118

    def energy(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
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
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(atomic_numbers)

        # 4. Message passing iterations.
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

        # 5. Predict atomic energies.
        element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))
        atomic_energies += element_bias[atomic_numbers]
        energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)
        return -jnp.sum(energy), energy

    # This method extracts the molecule-level descriptor.
    @nn.compact
    def extract_descriptor(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
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
        descriptor = jax.ops.segment_sum(x, segment_ids=batch_segments, num_segments=batch_size)
        return descriptor

    # New method: extract descriptor and its gradient with respect to positions.
    @nn.compact
    def extract_descriptor_and_gradient(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        # Compute the descriptor.
        descriptor = self.extract_descriptor(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
        # For calibration, assume batch_size == 1.
        pos_shape = positions.shape  # (num_atoms, 3)
        pos_flat = positions.reshape(-1)  # shape: (num_atoms*3,)
        variables = self.scope.variables()
        def desc_fn(pos_flat):
            pos_reshaped = pos_flat.reshape(pos_shape)
            mod = self  # capture instance
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
            return jnp.ravel(desc[0])  # return as 1D vector
        # Use jax.vmap to compute the gradient for each descriptor component.
        descriptor_dim = descriptor.shape[-1]
        G = jax.vmap(lambda i: jax.grad(lambda x: desc_fn(x)[i])(pos_flat))(jnp.arange(descriptor_dim))
        # G has shape (descriptor_dim, num_atoms*3)
        return descriptor, G

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
        (_, energy), forces = energy_and_forces(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
        return energy, forces

# --- Batch preparation, loss, and training functions ---
def prepare_batches(key, data, batch_size):
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

@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size'))
def eval_step(model_apply, batch, batch_size, forces_weight, params):
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
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data['atomic_numbers']))
    params = model.init(init_key,
                        atomic_numbers=train_data['atomic_numbers'],
                        positions=train_data['positions'][0],
                        dst_idx=dst_idx,
                        src_idx=src_idx)
    opt_state = optimizer.init(params)
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
        if epoch % 10 == 0:
            print(f"epoch: {epoch: 3d}    train loss: {train_loss:8.3f}   valid loss: {valid_loss:8.3f}")
            print(f"    energy mae: {train_energy_mae:8.3f}   valid energy mae: {valid_energy_mae:8.3f}")
            print(f"    forces mae: {train_forces_mae:8.3f}   valid forces mae: {valid_forces_mae:8.3f}")
    return params

# --- Calibration functions ---
def calibrate_model(params, model, dataset, beta=beta):
    """
    For each sample in the calibration dataset:
      1. Extract descriptor and its gradient using extract_descriptor_and_gradient.
      2. Compute T_sample = beta^2 * (G @ G.T) and c_sample = beta^2 * (G @ (-force)).
    Then average and solve T theta = c.
    """
    T_accum = 0.0
    c_accum = 0.0
    num_samples = dataset['positions'].shape[0]
    for i in range(num_samples):
        pos = dataset['positions'][i]           # shape: (num_atoms, 3)
        force = dataset['forces'][i]            # shape: (num_atoms, 3)
        atomic_numbers = dataset['atomic_numbers']
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
        # Flatten force vector.
        f = -force.reshape(-1)
        T_sample = beta**2 * (G @ G.T)
        c_sample = beta**2 * (G @ f)
        T_accum += T_sample
        c_accum += c_sample
    T_avg = T_accum / num_samples
    c_avg = c_accum / num_samples
    theta_star = jnp.linalg.solve(T_avg, c_avg)
    print("Calibration complete: theta_star =", theta_star)
    return theta_star

def calibrated_energy(params, theta, model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
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
    energy_calibrated = jnp.dot(descriptor, theta)
    return energy_calibrated

def calibrated_forces(params, theta, model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
    def energy_fn(pos):
        return calibrated_energy(params, theta, model, atomic_numbers, pos, dst_idx, src_idx, batch_segments, batch_size).sum()
    forces = -jax.grad(energy_fn)(positions)
    return forces

# --- Main execution ---
data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)
train_data, valid_data, mean_energy = prepare_datasets(data_key, num_train=num_train, num_valid=num_valid)

# Train the initial (uncalibrated) message-passing model.
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

# Save bare (uncalibrated) model parameters.
import flax
with open("bare_model_params.bin", "wb") as f:
    f.write(flax.serialization.to_bytes(params))
print("Bare model parameters saved.")

## Run calibration on the training data.
#theta_star = calibrate_model(params, message_passing_model, train_data, beta=beta)

# Prepare calibration dataset from the last 200 configurations of the full dataset.
# or we should put other MD dataset here ... 
calib_data = prepare_calibration_dataset(filename, mean_energy=mean_energy, num_calib=200)
# Calibrate using the calibration dataset.
theta_star = calibrate_model(params, message_passing_model, calib_data, beta=beta)



# Save calibrated model parameters as a dictionary.
calibrated_params = {"model_params": params, "theta": theta_star}
with open("calibrated_model_params.bin", "wb") as f:
    f.write(flax.serialization.to_bytes(calibrated_params))
print("Calibrated model parameters saved.")

# --- Compare Predictions from Uncalibrated and Calibrated Models ---
# Use the first sample from the validation set.
sample = valid_data
num_atoms = sample['positions'].shape[1]  # (num_samples, num_atoms, 3)
batch_segments = jnp.zeros(num_atoms, dtype=jnp.int32)
batch_size = 1
dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
atomic_numbers = sample['atomic_numbers']
positions = sample['positions'][0]  # first molecule

# Uncalibrated model predictions.
initial_energy, initial_forces = message_passing_model.apply(
    params,
    atomic_numbers,
    positions,
    dst_idx,
    src_idx,
    batch_segments,
    batch_size
)

# Calibrated model predictions.
calib_energy = calibrated_energy(params, theta_star, message_passing_model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
calib_forces = calibrated_forces(params, theta_star, message_passing_model, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)

print("\n=== Uncalibrated Model Predictions ===")
print("Energy:", initial_energy)
print("Forces:", initial_forces)

print("\n=== Calibrated Model Predictions ===")
print("Calibrated Energy:", calib_energy)
print("Calibrated Forces:", calib_forces)
