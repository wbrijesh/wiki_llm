import os
import time
import math
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

class ModelConfig:
    """Configuration for transformer language model"""
    def __init__(self):
        # Model architecture
        self.d_model = 512        # Embedding dimension
        self.num_heads = 8        # Number of attention heads
        self.num_layers = 8       # Number of transformer layers
        self.d_ff = 2048         # Feed-forward dimension
        self.max_seq_length = 128 # Sequence length
        self.dropout = 0.1       # Dropout rate
        self.vocab_size = 10000  # To be updated from tokenizer

        # Training parameters
        self.batch_size = 64     # Optimized for M2 Pro
        self.learning_rate = 3e-4
        self.max_epochs = 10
        self.warmup_steps = 4000
        self.gradient_clip = 1.0

        # Paths
        self.train_path = "dataset/train_seqlen128.npy"
        self.val_path = "dataset/test_seqlen128.npy"
        self.vocab_path = "wiki_vocab.txt"
        self.checkpoint_dir = "checkpoints"

        # M2 Pro specific settings
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.mixed_precision = False  # Set to False for now as MPS doesn't fully support it
        self.num_workers = 4  # Reduced for stability
        self.pin_memory = True

def create_dataloaders(config):
    """Create training and validation dataloaders"""
    # Load datasets
    train_data = torch.from_numpy(np.load(config.train_path))
    val_data = torch.from_numpy(np.load(config.val_path))

    # Create input/target pairs
    def create_lm_dataset(data):
        x = data[:, :-1]  # Input sequence
        y = data[:, 1:]   # Target sequence
        return TensorDataset(x, y)

    # Create datasets
    train_dataset = create_lm_dataset(train_data)
    val_dataset = create_lm_dataset(val_data)

    # Create dataloaders optimized for M2 Pro
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, val_loader

def get_lr(step, d_model, warmup_steps):
    """Learning rate schedule from the transformer paper"""
    return d_model ** (-0.5) * min(
        step ** (-0.5),
        step * warmup_steps ** (-1.5)
    )

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Setup directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Load tokenizer
        from simple_tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load_vocab(config.vocab_path)
        config.vocab_size = len(self.tokenizer.vocab)

        # Initialize model
        from transformer_lm import TransformerLM
        self.model = TransformerLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout
        ).to(config.device)

        # Initialize optimizer with M2 Pro optimized settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(config)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        print(f"Model initialized with {self.model.get_num_params():,} parameters")
        print(f"Training on {config.device}")

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'step': self.step,
            'config': {k: str(v) if isinstance(v, torch.device) else v
                      for k, v in self.config.__dict__.items()}
        }

        path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, path)

        # Save latest checkpoint
        latest_path = os.path.join(
            self.config.checkpoint_dir,
            'checkpoint_latest.pt'
        )
        torch.save(checkpoint, latest_path)

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.step = checkpoint['step']
        return epoch, loss

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        start_time = time.time()

        # Progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}"
        )

        for batch_idx, (data, targets) in pbar:
            # Move to device - keep as Long integers
            data = data.to(self.config.device)
            targets = targets.to(self.config.device)

            # Forward pass
            logits = self.model(data)
            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            # Update weights
            self.optimizer.step()

            # Update learning rate
            self.step += 1
            lr = get_lr(
                self.step,
                self.config.d_model,
                self.config.warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Update statistics
            total_loss += loss.item()
            current_loss = total_loss / (batch_idx + 1)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.2f}',
                'ppl': f'{math.exp(current_loss):.2f}',
                'lr': f'{lr:.2e}'
            })

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0

        for data, targets in tqdm(self.val_loader, desc="Validating"):
            # Move to device - keep as Long integers
            data = data.to(self.config.device)
            targets = targets.to(self.config.device)

            # Forward pass
            logits = self.model(data)
            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def generate_sample(self, prompt="The", max_tokens=50):
        """Generate sample text"""
        self.model.eval()

        # Encode prompt
        tokens = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long,
            device=self.config.device
        ).unsqueeze(0)

        # Generate text
        generated = self.model.generate(tokens, max_tokens, temperature=0.7)

        # Decode and print
        text = self.tokenizer.decode(generated.tolist())
        print('\nGenerated Text:')
        print('-' * 80)
        print(text)
        print('-' * 80)

    def train(self):
        """Main training loop"""
        print("Starting training...")
        best_val_loss = float('inf')
        start_time = time.time()

        try:
            for epoch in range(self.config.max_epochs):
                epoch_start = time.time()

                # Training phase
                train_loss = self.train_epoch(epoch)

                # Validation phase
                val_loss = self.evaluate()

                # Save checkpoint if best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)

                # Log progress
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch} Summary:")
                print(f"Train loss: {train_loss:.2f} | Train PPL: {math.exp(train_loss):.2f}")
                print(f"Valid loss: {val_loss:.2f} | Valid PPL: {math.exp(val_loss):.2f}")
                print(f"Epoch time: {epoch_time:.2f}s")

                # Generate sample
                self.generate_sample()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")

def main():
    # Initialize config
    config = ModelConfig()

    # Save config
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config_path = os.path.join(
        config.checkpoint_dir,
        f'config_{datetime.now():%Y%m%d_%H%M%S}.json'
    )
    with open(config_path, 'w') as f:
        # Handle non-serializable types
        config_dict = {k: str(v) if isinstance(v, torch.device) else v
                      for k, v in config.__dict__.items()}
        json.dump(config_dict, f, indent=4)

    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
