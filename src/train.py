import functools
import os
import urllib.request
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import e3x
import pickle
# Disable future warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Download the dataset.
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


class MessagePassingModel(nn.Module):
  features: int = 32
  max_degree: int = 2
  num_iterations: int = 3
  num_basis_functions: int = 8
  cutoff: float = 5.0
  max_atomic_number: int = 118  # This is overkill for most applications.


  def energy(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
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
    atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)  # (..., Natoms, 1, 1, 1)
    atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
    atomic_energies += element_bias[atomic_numbers]

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
  


def prepare_batches(key, data, batch_size):
  # Determine the number of training steps per epoch.
  data_size = len(data['energy'])
  steps_per_epoch = data_size//batch_size

  # Draw random permutations for fetching batches from the train data.
  perms = jax.random.permutation(key, data_size)
  perms = perms[:steps_per_epoch * batch_size]  # Skip the last batch (if incomplete).
  perms = perms.reshape((steps_per_epoch, batch_size))

  # Prepare entries that are identical for each batch.
  num_atoms = len(data['atomic_numbers'])
  batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
  atomic_numbers = jnp.tile(data['atomic_numbers'], batch_size)
  offsets = jnp.arange(batch_size) * num_atoms
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
  dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
  src_idx = (src_idx + offsets[:, None]).reshape(-1)

  # Assemble and return batches.
  return [
    dict(
        energy=data['energy'][perm],
        forces=data['forces'][perm].reshape(-1, 3),
        atomic_numbers=atomic_numbers,
        positions=data['positions'][perm].reshape(-1, 3),
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments = batch_segments,
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


def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, forces_weight, batch_size):
  # Initialize model parameters and optimizer state.
  key, init_key = jax.random.split(key)
  optimizer = optax.adam(learning_rate)
  dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(train_data['atomic_numbers']))
  params = model.init(init_key,
    atomic_numbers=train_data['atomic_numbers'],
    positions=train_data['positions'][0],
    dst_idx=dst_idx,
    src_idx=src_idx,
  )
  opt_state = optimizer.init(params)

  # Batches for the validation set need to be prepared only once.
  key, shuffle_key = jax.random.split(key)
  valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

  # Train for 'num_epochs' epochs.
  for epoch in range(1, num_epochs + 1):
    # Prepare batches.
    key, shuffle_key = jax.random.split(key)
    train_batches = prepare_batches(shuffle_key, train_data, batch_size)

    # Loop over train batches.
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
      train_loss += (loss - train_loss)/(i+1)
      train_energy_mae += (energy_mae - train_energy_mae)/(i+1)
      train_forces_mae += (forces_mae - train_forces_mae)/(i+1)

    # Evaluate on validation set.
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
      valid_loss += (loss - valid_loss)/(i+1)
      valid_energy_mae += (energy_mae - valid_energy_mae)/(i+1)
      valid_forces_mae += (forces_mae - valid_forces_mae)/(i+1)

    # Print progress.
    print(f"epoch: {epoch: 3d}                    train:   valid:")
    print(f"    loss [a.u.]             {train_loss : 8.3f} {valid_loss : 8.3f}")
    print(f"    energy mae [kcal/mol]   {train_energy_mae: 8.3f} {valid_energy_mae: 8.3f}")
    print(f"    forces mae [kcal/mol/Å] {train_forces_mae: 8.3f} {valid_forces_mae: 8.3f}")


  # Return final model parameters.
  return params



# Model hyperparameters.
features = 32
max_degree = 1
num_iterations = 3
num_basis_functions = 16
cutoff = 5.0

# Training hyperparameters.
num_train = 900
num_valid = 100
num_epochs = 100
learning_rate = 0.01
forces_weight = 1.0
batch_size = 10



# Create PRNGKeys.
data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

# Draw training and validation sets.
train_data, valid_data, _ = prepare_datasets(data_key, num_train=num_train, num_valid=num_valid)

# Create and train model.
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


# Create "Model" directory if it doesn't exist
model_dir = "Model"
os.makedirs(model_dir, exist_ok=True)

# Save the model parameters as a binary file
serialized_params = flax.serialization.to_bytes(params)
params_file_path = os.path.join(model_dir, "model_params.bin")
with open(params_file_path, "wb") as f:
    f.write(serialized_params)
print(f"Serialized model parameters saved to: {params_file_path}")

# Save the model as a .pkl file
model_checkpoint = {
    "model": message_passing_model,  # Save the model architecture
    "params": params,  # Save the trained model parameters
    "features": features,
    "max_degree": max_degree,
    "num_iterations": num_iterations,
    "num_basis_functions": num_basis_functions,
    "cutoff": cutoff
}
checkpoint_file_path = os.path.join(model_dir, "model_checkpoint.pkl")
with open(checkpoint_file_path, "wb") as f:
    pickle.dump(model_checkpoint, f)
print(f"Model checkpoint saved to: {checkpoint_file_path}")


















##################################################################
def fisher_theta(G_list, F_list, beta):
    """
    Compute Fisher-optimal parameters
    
    Args:
        G_list: List of gradient matrices
        F_list: List of force matrices
        beta: Temperature factor
    """
    GGT = [jnp.dot(G.T, G) for G in G_list]
    
    c_list = jnp.array( [(beta**2) * jnp.dot(G_list[i].T, F_list[i]) for i in range(len(G_list))])
    print("ok")
    # Average over samples
    
    stacked_c = jnp.mean(jnp.stack(c_list), axis=0)
    stacked_GGT = jnp.mean(jnp.stack(GGT), axis=0)
    
    # Compute temperature-scaled matrix
    T = beta**2 * stacked_GGT
    
    # Solve linear system
    theta_dot = jnp.linalg.solve(T + 1e-6 * jnp.eye(T.shape[0]), stacked_c)
    #print(theta_dot)
    return theta_dot




class fisher_model(nn.Module):  # Doesn't inherit from MessagePassingModel
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 118

    @nn.compact  # Important for parameter initialization
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
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
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)

            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

            # Residual connection.
            x = e3x.nn.add(x, y)

        # 5. Predict atomic energies with an ordinary dense layer.
        element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number + 1))
        atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)  # (..., Natoms, 1, 1, 1)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
        atomic_energies += element_bias[atomic_numbers]

        # 6. Sum atomic energies to obtain the total energy.
        energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)

        return -jnp.sum(energy), energy  # Return energy and forces (negative gradient)
    
    @nn.compact
    def energy_only(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        energy, _ = self(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)
        return energy
    
    @nn.compact
    def get_grad_desc(self, params, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).
        
        print(displacements.shape)
        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )
        
        print(basis.shape)
        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers)
        print(x.shape)
        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)

            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

            # Residual connection.
            x = e3x.nn.add(x, y)

        # 5. Get the activations before the last layer
        x = jax.lax.stop_gradient(x)  # Freeze all the layers before this one
        print(x.shape)
        # 6. Calculate the gradient of the output with respect to x
        @nn.compact
        def energy_no_grad(x_local):
            element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
            #atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x)  # (..., Natoms, 1, 1, 1)
            atomic_energies = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x_local)  # (..., Natoms, 1, 1, 1)
            atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # Squeeze last 3 dimensions.
            atomic_energies += element_bias[atomic_numbers]
            energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)
            return jnp.sum(energy)

        grad_desc = jax.grad(energy_no_grad)(x)
        return grad_desc


    @nn.compact
    def get_layer_gradients(self, params, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # # 2. Expand displacement vectors in basis functions.
        # basis = e3x.nn.basis(
        #     displacements,
        #     num=self.num_basis_functions,
        #     max_degree=self.max_degree,
        #     radial_fn=e3x.nn.reciprocal_bernstein,
        #     cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        # )

        @nn.compact
        # Final layer gradient computation
        def displacement_grad(local_displacements):
            # # Recreate the entire network computation with input displacements
            local_basis = e3x.nn.basis(
                local_displacements,
                num=self.num_basis_functions,
                max_degree=self.max_degree,
                radial_fn=e3x.nn.reciprocal_bernstein,
                cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
            )

            # Embed atomic numbers
            local_x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers)

            # Perform message passing iterations
            for i in range(self.num_iterations):
                if i == self.num_iterations - 1:
                    local_y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(local_x, local_basis, dst_idx=dst_idx, src_idx=src_idx)
                    local_x = e3x.nn.change_max_degree_or_type(local_x, max_degree=0, include_pseudotensors=False)
                else:
                    local_y = e3x.nn.MessagePass()(local_x, local_basis, dst_idx=dst_idx, src_idx=src_idx)
                local_y = e3x.nn.add(local_x, local_y)

                # Atom-wise refinement MLP
                local_y = e3x.nn.Dense(self.features)(local_y)
                local_y = e3x.nn.silu(local_y)
                local_y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(local_y)

                # Residual connection
                local_x = e3x.nn.add(local_x, local_y)
            
            return local_x
            # # Final layer energy computation
            # element_bias = self.param('element_bias', lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number+1))
            # dense_layer = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)
            # atomic_energies = dense_layer(local_x)

            #return atomic_energies
            # atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))
            # atomic_energies += element_bias[atomic_numbers]
            
            # energy = jax.ops.segment_sum(atomic_energies, segment_ids=batch_segments, num_segments=batch_size)
            # return jnp.sum(energy)

        # Compute gradients with respect to displacements
        layer_gradients = jax.jacobian(displacement_grad)(displacements)
        
        return layer_gradients

    
    @nn.compact
    def apply_fisher(self, params, batch, beta, batch_size):
        G = self.get_layer_gradients(params, batch['atomic_numbers'], batch["positions"], batch['dst_idx'], batch['src_idx'],
                               batch['batch_segments'], batch_size)
        
        G = jnp.mean(G, axis=4)
        G = jnp.squeeze(G, axis=(1, 2))
        G = jnp.transpose(G, (0, 2, 1))
        print(G.shape)
        G = G.astype(jnp.float16) 
        F = batch['forces']  # Target forces

        print("G shape", G.shape)
        print("F shape", F.shape)

        theta_fisher = fisher_theta(G, F, beta)  # Assuming you have this function defined
        return theta_fisher




from copy import deepcopy
new_params = deepcopy(params)
# Create fisher_model instance
fisher_model_instance = fisher_model(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
)

# *** Initialize the fisher_model_instance ***
key = jax.random.PRNGKey(0)
key, shuffle_key = jax.random.split(key)
valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)
train_batches = prepare_batches(shuffle_key, train_data, batch_size)

dummy_data = train_batches[0]  # Get a batch for initialization
num_atoms = len(valid_data['atomic_numbers'])
batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
new_params = fisher_model_instance.init(
    key,
    atomic_numbers=dummy_data['atomic_numbers'],
    positions=dummy_data['positions'],
    dst_idx=dummy_data['dst_idx'],
    src_idx=dummy_data['src_idx'],
    batch_segments=batch_segments, 
    batch_size=batch_size
)

# Convert new_params to a FrozenDict
new_params = flax.core.freeze(new_params)
new_params = new_params.unfreeze()
# print(new_params)
# Update parameters (Corrected)
for k, v in params.items():
    if 'Dense_5' not in k:  # Skip the last layer
        new_params['params'][k] = v  # Access the correct level 'param'

# Freeze new_params again
new_params = flax.core.freeze(new_params)
from scipy.constants import Boltzmann
temperature = 300
to_beta = 1 / (Boltzmann * temperature)

# Apply Fisher update and update the last layer's weights for ONE batch
batch = next(iter(train_batches)) # Get one batch
theta_fisher = fisher_model_instance.apply(
    params,
    method=fisher_model_instance.apply_fisher,
    params=new_params,
    batch=batch,
    beta=to_beta,
    batch_size=batch_size
)
#     theta_fisher = fisher_model_instance.apply_fisher(new_params, batch, to_beta, batch_size)
if theta_fisher is not None:
    print(theta_fisher)
    new_params = new_params.unfreeze()
    new_params['params']['Dense_5']['kernel'] = theta_fisher.T  # Update the last layer
    new_params = flax.core.freeze(new_params)

# # Now create a frozen model using the trained parameters.
# frozen_model = fisher_model(
#     features=32,
#     max_degree=1,
#     num_iterations=3,
#     num_basis_functions=16,
#     cutoff=5.0,
# )

# # Load the trained parameters into the frozen model
# frozen_model.params = message_passing_model.params

# def apply_fisher_update(model, batch, beta, batch_size):
#     """
#     Apply Fisher update to the model parameters
    
#     Args:
#         model: FrozenMessagePassingModel instance
#         batch: Dictionary containing batch data
#         beta: Temperature factor (1/kT)
#         batch_size: Size of the batch
#     """
#     # Get model predictions
#     energy, forces = model.apply(
#         model.params,
#         atomic_numbers=batch['atomic_numbers'],
#         positions=batch['positions'],
#         dst_idx=batch['dst_idx'],
#         src_idx=batch['src_idx'],
#         batch_segments=batch['batch_segments'],
#         batch_size=batch_size
#     )
    
#     # Initialize lists for gradients and forces
#     G_list = []
#     F_list = []
    
#     # Compute gradients with respect to positions
#     def compute_energy_grad(positions):
#         return model.apply(
#             model.params,
#             atomic_numbers=batch['atomic_numbers'],
#             positions=positions,
#             dst_idx=batch['dst_idx'],
#             src_idx=batch['src_idx'],
#             batch_segments=batch['batch_segments'],
#             batch_size=batch_size
#         )[0]  # Only take energy output
    
#     # Compute Jacobian
#     jacobian_fn = jax.jacfwd(compute_energy_grad)
#     G = jacobian_fn(batch['positions'])
    
#     # Reshape gradients and forces
#     G = G.reshape(-1, 3)  # Reshape to (num_atoms * batch_size, 3)
#     F = batch['forces']  # Target forces
    
#     G_list.append(G)
#     F_list.append(F)
    
#     # Compute Fisher-optimal parameters
#     try:
#         theta_fisher = fisher_theta(G_list, F_list, beta)
#         return theta_fisher
#     except Exception as e:
#         print(f"Error in Fisher update: {e}")
#         return None

# # Modified fisher_theta function
# def fisher_theta(G_list, F_list, beta):
#     """
#     Compute Fisher-optimal parameters
    
#     Args:
#         G_list: List of gradient matrices
#         F_list: List of force matrices
#         beta: Temperature factor
#     """
#     GGT = [jnp.dot(G.T, G) for G in G_list]
#     c_list = jnp.array([(beta**2) * jnp.dot(G_list[i].T, F_list[i]) 
#                         for i in range(len(G_list))])
    
#     # Average over samples
#     stacked_c = jnp.mean(jnp.stack(c_list), axis=0)
#     stacked_GGT = jnp.mean(jnp.stack(GGT), axis=0)
    
#     # Compute temperature-scaled matrix
#     T = beta**2 * stacked_GGT
    
#     # Solve linear system
#     theta_dot = jnp.linalg.solve(T + 1e-6 * jnp.eye(T.shape[0]), stacked_c)
#     return theta_dot

# # def fisher_theta(G_list, F_list, beta):
# #     GGT = [jnp.dot(G.T, G) for G in G_list]
# #     c_list = jnp.array([(beta**2) * jnp.dot(G_list[i].T, F_list[i]) for i in range(len(G_list))])
# #     stacked_c = jnp.mean(jnp.stack(c_list), axis=0)
# #     stacked_GGT = jnp.mean(jnp.stack(GGT), axis=0)
# #     T = beta**2 * stacked_GGT
# #     theta_dot = jnp.linalg.solve(T, stacked_c)
# #     return theta_dot

# # from jax import grad, jit
# # def apply_fisher_update(model, dataloader, beta):
# #     G_list, F_list = [], []
# #     for batch in dataloader:
# #         descriptors, forces = batch  # Extract descriptors and target forces
        
# #         def model_output(params, x):
# #             return model.apply(params, x)
        
# #         # Compute descriptor gradients (Jacobian for last layer)
# #         grads = []
# #         for i in range(model.output_dim):  # Loop over output dimensions
# #             grad_fn = jit(grad(lambda x: model_output(model.params, x)[..., i]))
# #             grad_val = grad_fn(descriptors)
# #             grads.append(np.array(grad_val))  # Convert to NumPy for processing
        
# #         G_list.append(np.concatenate(grads, axis=-1))  # Collect gradients
# #         F_list.append(np.array(forces))  # Collect forces
    
# #     # Compute Fisher-optimal parameters
# #     theta_fisher = fisher_theta(G_list, F_list, beta)
    
# #     # Update last layer parameters
# #     model.params['output_layer']['kernel'] = jnp.array(theta_fisher.T)

# #     return model


# from scipy.constants import Boltzmann
# temperature = 300
# to_beta = 1/(Boltzmann*temperature)

# # Create PRNGKeys.
# data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)
# # Draw training and validation sets.
# train_data, valid_data, _ = prepare_datasets(data_key, num_train=num_train, num_valid=num_valid)

# # Batches for the validation set need to be prepared only once.
# key, shuffle_key = jax.random.split(data_key)
# valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)
# train_batches = prepare_batches(shuffle_key, train_data, batch_size)

# fisher_theta = []
# for i, batch in enumerate(train_batches):
#     fisher_theta.append(apply_fisher_update(frozen_model, batch, to_beta))

# print(fisher_theta)