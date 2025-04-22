# schrodinger_fcnn_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
# Removed DataLoader imports
import numpy as np
import matplotlib.pyplot as plt
import timeit
import csv
import os
import math

class SchrodingerFCNN(nn.Module): # Renamed from MLP
    """
    PyTorch implementation of a 1-hidden-layer FCNN with manual weights/biases
    for predicting Schrodinger equation wavefunctions.
    Architecture: Manual Linear1 -> Softplus -> Manual Linear2
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SchrodingerFCNN, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Dynamically create layers based on hidden_sizes list
        for i in range(len(layer_sizes) - 1):
            # Note: Bias is True by default in nn.Linear
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Add activation except for the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.Softplus()) # Using Softplus as in the original code

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SchrodingerFCNNTrainer: # Renamed from MLPTrainer
    """
    Handles data loading, training (manual batching), evaluation, and plotting
    for the SchrodingerFCNN model.
    """
    def __init__(self,
                 # Data parameters
                 Rx=3, Nx=128,
                 all_data=8000,
                 data_path_prefix='.',
                 data_file_name_template='NN_valid_data_schrodinger_1D_sol-Rx={Rx}-Nx={Nx}-num_data={num_data}-size_data={size_data}.csv',
                 num_data_in_filename=25000,
                 # Model architecture parameters
                 hidden_sizes=[128],
                 # Training parameters
                 batch_size=200,
                 learning_rate=0.01,
                 epochnum=200,
                 weight_decay=1e-4,
                 shuffle_each_epoch=True, # Added shuffle option
                 early_stopping_patience=10): # Added patience

        self.Rx = Rx
        self.Nx = Nx
        self.data_size = Nx - 1         # Input size (199 for Nx=200)
        self.output_neuron = Nx - 1     # Output size (199 for Nx=200)
        self.hidden_sizes = hidden_sizes
        self.all_data_to_use = all_data # Use only this many samples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochnum = epochnum
        self.weight_decay = weight_decay
        self.data_path_prefix = data_path_prefix
        self.shuffle_each_epoch = shuffle_each_epoch # Store shuffle preference
        self.early_stopping_patience = early_stopping_patience

        # Construct filename
        # User code uses data_size+2 (e.g., 199+2=201 for Nx=200) in filename
        size_data_in_filename = self.Nx + 1
        self.data_file_name = data_file_name_template.format(
            Rx=self.Rx, Nx=self.Nx, num_data=num_data_in_filename, size_data=size_data_in_filename
        )

        self.dx = 2 * self.Rx / self.Nx
        # x coordinates correspond to the data size (Nx-1 points)
        self.x = np.linspace(-self.Rx + self.dx/2, self.Rx - self.dx/2, self.Nx - 1)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load data (without creating DataLoaders)
        self._load_data()

        # Initialize model (renamed)
        self.model = SchrodingerFCNN(
            input_size=self.data_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_neuron
        ).to(self.device)

        print("\nModel Architecture:")
        print(self.model)

        # Optimizer (Using SGD to match user's original implementation style)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Loss function (Base criterion is MSE)
        self.criterion = nn.MSELoss(reduction='mean')

    def _load_data(self):
        """Loads data, processes, splits into tensors (no DataLoaders)."""
        full_file_path = os.path.join(self.data_path_prefix, self.data_file_name)
        print(f"Loading data from: {full_file_path}")

        try:
            # Load data using numpy/csv as before
            with open(full_file_path, newline='') as csvfile:
                dataconfig = list(csv.reader(csvfile))
                dataconfig = np.array(dataconfig, dtype="float32")
        except FileNotFoundError:
            print(f"Error: Data file not found at {full_file_path}")
            print(f"Current working directory: {os.getcwd()}")
            raise
        except Exception as e:
            print(f"Error loading or parsing data: {e}")
            raise

        # --- Data Slicing (Matching User Script) ---
        # potential = dataconfig[:2000, 1:200] # For Nx=200 -> cols 1 to 199
        # fnum = dataconfig[:2000, 202:401]    # For Nx=200 -> cols 202 to 400
        potential_start_col = 1
        potential_end_col = self.Nx # Exclusive index -> 1 to Nx-1
        fnum_start_col = self.Nx + 2 # User code starts at 202 for Nx=200
        fnum_end_col = 2 * self.Nx + 1 # User code ends at 401 for Nx=200

        potential_raw = dataconfig[:self.all_data_to_use, potential_start_col:potential_end_col]
        fnum_raw = dataconfig[:self.all_data_to_use, fnum_start_col:fnum_end_col]

        # Verify shapes
        if potential_raw.shape[1] != self.data_size:
             raise ValueError(f"Potential data width ({potential_raw.shape[1]}) != expected ({self.data_size})")
        if fnum_raw.shape[1] != self.output_neuron:
             raise ValueError(f"Wavefunction data width ({fnum_raw.shape[1]}) != expected ({self.output_neuron})")
        if potential_raw.shape[0] != self.all_data_to_use:
            raise ValueError(f"Loaded fewer samples ({potential_raw.shape[0]}) than requested ({self.all_data_to_use})")

        print(f"Raw data shapes - Potential: {potential_raw.shape}, Wavefunction: {fnum_raw.shape}")

        # Convert to PyTorch tensors
        potential_all = torch.from_numpy(potential_raw).float()
        fnum_all = torch.from_numpy(fnum_raw).float()

        # Split data into training, validation, and testing sets
        total_samples = potential_all.shape[0]
        self.training_data_count = int(total_samples * 0.60) # 60% train
        self.validation_data_count = int(total_samples * 0.20) # 20% validation
        self.test_data_count = total_samples - self.training_data_count - self.validation_data_count # Remaining 20% test

        # Assign data splits based on calculated counts
        train_end_idx = self.training_data_count
        val_end_idx = self.training_data_count + self.validation_data_count

        # No shuffle here, data is split sequentially. Shuffling happens per epoch in train method.
        self.potential_train = potential_all[:train_end_idx]
        self.fnum_train = fnum_all[:train_end_idx]
        self.potential_validation = potential_all[train_end_idx:val_end_idx]
        self.fnum_validation = fnum_all[train_end_idx:val_end_idx]
        self.potential_test = potential_all[val_end_idx:]
        self.fnum_test = fnum_all[val_end_idx:]

        # Update counts just in case rounding changed things slightly
        self.training_data_count = self.potential_train.shape[0]
        self.validation_data_count = self.potential_validation.shape[0]
        self.test_data_count = self.potential_test.shape[0]

        print(f"Data loaded successfully:")
        print(f"  Training samples: {self.training_data_count}")
        print(f"  Validation samples: {self.validation_data_count}")
        print(f"  Testing samples:  {self.test_data_count}")


    def calculate_loss(self, f_pred, f_target):
        """
        Calculates the loss, scaling MSE to match the user's definition:
        Loss = dx * sum((fnum-fp)**2) / batch_size
        """
        mse_loss = self.criterion(f_pred, f_target)
        # Scale MSE: dx * mse_loss * output_neuron (number of elements per sample)
        scaled_loss = self.dx * mse_loss * self.output_neuron
        return scaled_loss

    def train_one_epoch(self):
        """Performs a single training epoch using manual batch slicing."""
        self.model.train() # Set model to training mode
        running_loss = 0.0
        samples_processed = 0
        # Calculate number of batches needed, rounding up
        num_batches = (self.training_data_count + self.batch_size - 1) // self.batch_size

        # --- Optional: Shuffle training data indices ---
        indices = torch.arange(self.training_data_count)
        if self.shuffle_each_epoch:
            indices = indices[torch.randperm(self.training_data_count)] # Get shuffled indices

        # Iterate through the training data using manual slicing
        for i in range(0, self.training_data_count, self.batch_size):
            # Determine indices for the current batch
            batch_indices = indices[i:min(i + self.batch_size, self.training_data_count)]
            if len(batch_indices) == 0: continue # Should not happen with correct range

            # Get batch data using the shuffled indices
            batch_potential = self.potential_train[batch_indices].to(self.device)
            batch_fnum = self.fnum_train[batch_indices].to(self.device)

            current_batch_size = len(batch_indices)
            samples_processed += current_batch_size

            # --- Forward Pass ---
            f_pred = self.model(batch_potential)

            # --- Calculate Loss ---
            loss = self.calculate_loss(f_pred, batch_fnum) # Use scaled loss

            # --- Backward Pass and Optimization ---
            self.optimizer.zero_grad() # Clear previous gradients
            loss.backward()           # Compute gradients
            self.optimizer.step()      # Update weights

            # Accumulate loss, weighted by batch size for correct epoch average
            running_loss += loss.item() * current_batch_size

            # Print loss per batch like user script
            current_batch_num = (i // self.batch_size) + 1
            print(f'  Batch {current_batch_num}/{num_batches} - Loss: {loss.item():.3f}')

        # Calculate average loss for the epoch
        avg_epoch_loss = running_loss / samples_processed if samples_processed > 0 else 0
        return avg_epoch_loss, samples_processed


    def evaluate(self):
        """Evaluates the model on the test set using manual batch slicing."""
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        num_samples = 0
        with torch.no_grad(): # Disable gradient calculations during evaluation
            # Iterate through test data using manual slicing
            for i in range(0, self.test_data_count, self.batch_size):
                end_idx = min(i + self.batch_size, self.test_data_count)
                if i >= end_idx: continue # Skip if start index is already past the end

                batch_potential = self.potential_test[i:end_idx].to(self.device)
                batch_fnum = self.fnum_test[i:end_idx].to(self.device)

                current_batch_size = len(batch_potential)
                if current_batch_size == 0: continue

                # Forward pass
                f_pred = self.model(batch_potential)
                # Calculate scaled loss
                loss = self.calculate_loss(f_pred, batch_fnum)

                # Accumulate total loss, weighted by batch size
                total_loss += loss.item() * current_batch_size
                num_samples += current_batch_size

        # Calculate average loss per individual sample across the entire test set
        avg_loss_per_sample = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss_per_sample


    def _evaluate_validation_set(self):
        """Evaluates the model on the validation set."""
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation
            if self.validation_data_count > 0:
                # Process validation data in one go (or batches if large)
                potential_val = self.potential_validation.to(self.device)
                fnum_val = self.fnum_validation.to(self.device)
                f_pred_val = self.model(potential_val)
                val_loss_tensor = self.calculate_loss(f_pred_val, fnum_val)
                total_val_loss = val_loss_tensor.item() * self.validation_data_count # Get total loss
            else:
                return float('inf') # Return infinity if no validation data

        avg_val_loss = total_val_loss / self.validation_data_count if self.validation_data_count > 0 else float('inf')
        self.model.train() # Set model back to training mode
        return avg_val_loss


    def train(self):
        """Main training loop over multiple epochs."""
        print("\n--- Starting Training ---")
        epoch_train_losses = [] # Store average training loss per epoch
        epoch_valid_losses = [] # Store average validation loss per epoch
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        early_stop_triggered = False
        start_time = timeit.default_timer()

        for epoch in range(self.epochnum):
            print(f"\n--- Epoch {epoch + 1}/{self.epochnum} ---")
            # Train for one epoch
            avg_train_loss, samples_this_epoch = self.train_one_epoch()
            epoch_train_losses.append(avg_train_loss)

            # --- Validation Step ---
            avg_valid_loss = self._evaluate_validation_set()
            epoch_valid_losses.append(avg_valid_loss)
            print(f"Epoch {epoch+1} complete. Avg Training Loss: {avg_train_loss:.3f} | Avg Validation Loss: {avg_valid_loss:.3f}")

            # Evaluate on test set periodically
            if (epoch + 1) % 5 == 0 or epoch == self.epochnum - 1:
                 test_loss = self.evaluate() # Call evaluation method
                 print(f'Test Loss after Epoch {epoch+1}: {test_loss:.3f}')

            # --- Early Stopping Check --- 
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_no_improve = 0
                # Save the best model state dictionary
                best_model_state = self.model.state_dict()
                print(f"  (New best validation loss: {best_valid_loss:.3f}. Saving model state)")
            else:
                epochs_no_improve += 1
                print(f"  (Validation loss did not improve for {epochs_no_improve} epoch(s))", end='')
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f"\n\nEarly stopping triggered after {epoch+1} epochs.")
                    early_stop_triggered = True
                    break # Exit the training loop
                else:
                    print() # Add newline if not stopping

            # Shuffle indices for the next epoch if enabled
            if self.shuffle_each_epoch:
                indices = torch.randperm(self.training_data_count)

        stop_time = timeit.default_timer()
        total_time = stop_time - start_time

        # --- Load best model state if early stopping occurred --- 
        if early_stop_triggered and best_model_state is not None:
             print("\nLoading best model state from early stopping.")
             self.model.load_state_dict(best_model_state)
        # ---

        print(f"\n--- Training Finished ---")
        print(f"Total Training time: {total_time:.2f} seconds")

        # Plot training and validation loss
        self._plot_loss(epoch_train_losses, epoch_valid_losses)

        # Evaluate final performance on test set (using best model if early stopped)
        final_test_loss = self.evaluate()
        print(f"\nFinal Test Loss (using {'best' if early_stop_triggered else 'final'} model): {final_test_loss:.3f}")

        # Plot some predictions from the test set
        self._plot_predictions(num_plots=5) # Plot 5 examples

        # Save the trained model state (final or best)
        save_suffix = '_best' if early_stop_triggered else '_final'
        self.save_model(suffix=save_suffix)


    def _plot_loss(self, epoch_train_losses, epoch_valid_losses):
        """Plots the average training and validation loss per epoch."""
        print("\n--- Plotting Training and Validation Loss ---")
        plt.figure(figsize=(10, 6))
        # Use number of recorded epochs in case of early stopping
        num_epochs_recorded = len(epoch_train_losses)
        epochs_range = range(1, num_epochs_recorded + 1)
        plt.plot(epochs_range, epoch_train_losses, marker='o', linestyle='-', label='Training Loss')
        plt.plot(epochs_range, epoch_valid_losses, marker='x', linestyle='--', label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = 'fcnn_train_validation_loss.png' # Updated filename
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
        plt.close()


    def _plot_predictions(self, num_plots=5):
        """Plots model predictions against target values for a few test samples."""
        print(f"\n--- Plotting {num_plots} Test Predictions ---")
        self.model.eval() # Ensure model is in evaluation mode

        # Select evenly spaced indices from the test set
        if self.test_data_count < num_plots:
            num_plots = self.test_data_count
        if num_plots <= 0:
             print("No test samples to plot.")
             return

        plot_indices = np.linspace(0, self.test_data_count - 1, num_plots, dtype=int)

        with torch.no_grad():
            for i, sample_idx in enumerate(plot_indices):
                # Get potential and target wavefunction for the sample
                # Need unsqueeze(0) to add batch dimension for the model
                potential_sample_tensor = self.potential_test[sample_idx].unsqueeze(0).to(self.device)
                fnum_target_numpy = self.fnum_test[sample_idx].cpu().numpy()
                potential_input_numpy = self.potential_test[sample_idx].cpu().numpy()

                # Get model prediction
                f_pred_tensor = self.model(potential_sample_tensor)
                # Remove batch dim and move to CPU for plotting
                f_pred_numpy = f_pred_tensor.squeeze(0).cpu().numpy()

                # Calculate loss for this specific sample for context
                loss_sample = self.calculate_loss(f_pred_tensor,
                                                  self.fnum_test[sample_idx].unsqueeze(0).to(self.device)).item()

                plt.figure(dpi=150)
                plt.plot(self.x, f_pred_numpy, color='blue', linewidth=2, label=r'Predicted $\psi(x)$')
                plt.plot(self.x, fnum_target_numpy, color='red', linestyle='--', label=r'Target $\psi(x)$')
                # --- Plot unscaled potential --- 
                # Create a secondary y-axis for the potential for better visibility
                ax1 = plt.gca() # Get current axes
                ax2 = ax1.twinx() # Create a twin axis sharing the x-axis
                ax2.plot(self.x, potential_input_numpy, color='green', linestyle=':', alpha=0.7, label='Potential')
                ax2.set_ylabel('Potential V(x)', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                # --- 

                # Adjust labels and legends for two axes
                plt.xlabel(r'$x$')
                ax1.set_ylabel(r'$\psi(x)$', color='black') # Label for primary axis
                ax1.tick_params(axis='y', labelcolor='black')
                # Use the index relative to the start of the full dataset
                full_data_index = self.training_data_count + self.validation_data_count + sample_idx
                plt.title(f'FCNN Test Sample (Index: {full_data_index}) | Loss: {loss_sample:.3f}') # Updated title
                # Match y-lim from user script
                # ax1.set_ylim(0, 1.2) 
                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                # ax1.legend()
                plt.grid(True, linestyle='--', alpha=0.6)

                plot_filename = f'fcnn_prediction_sample_{full_data_index}.png' # Updated filename
                plt.savefig(plot_filename) # Save the plot
                print(f"Saved prediction plot to {plot_filename}")
                plt.show()


    def save_model(self, base_filename="schrodinger_fcnn_model", suffix="_final"):
        """Saves the model's state dictionary with a suffix."""
        try:
            filename = f"{base_filename}{suffix}.pth"
            torch.save(self.model.state_dict(), filename)
            print(f"Model state dictionary saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving model state: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- User Configuration ---
    # 1. Set path to data directory
    #    Example: If data is in the same directory: '.'
    #    Example: If data is in Google Drive: '/content/drive/My Drive/1D_GPE_NN_data/'
    DATA_DIRECTORY = '.'

    # 2. Parameters from user script (MUST match intended data file)
    PARAMS = {
        "Rx": 3,
        "Nx": 128,
        "all_data": 3000, # Number of samples to load/use from CSV
        "num_data_in_filename": 25000, # The 'num_data' value IN THE FILENAME ITSELF
        "hidden_sizes": [128, 128], # Changed to two hidden layers
        "batch_size": 200,
        "learning_rate": 0.01, # Lowered learning rate
        "epochnum": 200,
        "weight_decay": 1e-4,
        "shuffle_each_epoch": True, # Set to False to mimic user script exactly (no shuffling)
        "early_stopping_patience": 20 # Added patience

    }
    # --- End User Configuration ---

    # Instantiate the renamed trainer
    trainer = SchrodingerFCNNTrainer(
        data_path_prefix=DATA_DIRECTORY,
        **PARAMS
    )

    # Train the model
    trainer.train()
