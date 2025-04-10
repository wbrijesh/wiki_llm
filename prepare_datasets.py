import os
import random
import numpy as np
from tqdm import tqdm
from simple_tokenizer import SimpleTokenizer

# Constants
SEQUENCE_LENGTH = 128  # Length of sequences for training/testing
TRAIN_SPLIT = 0.9      # 90% training, 10% testing

def read_articles(file_path, max_articles=None):
    """Read articles from the processed wiki file."""
    articles = []
    current_article = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for end of article marker
            if line.strip() == "<|endoftext|>":
                if current_article:
                    articles.append("".join(current_article))
                    current_article = []

                    # Check if we've reached max articles
                    if max_articles and len(articles) >= max_articles:
                        break
            else:
                current_article.append(line)

    print(f"Read {len(articles)} articles")
    return articles

def create_sequences(tokenized_articles, sequence_length):
    """Create fixed-length sequences for training/testing."""
    all_sequences = []

    for tokens in tqdm(tokenized_articles, desc="Creating sequences"):
        # Skip very short articles
        if len(tokens) < sequence_length // 2:
            continue

        # Create sequences with overlap
        for i in range(0, len(tokens) - sequence_length + 1, sequence_length // 2):
            # Extract sequence
            sequence = tokens[i:i + sequence_length]

            # Skip short sequences at the end
            if len(sequence) < sequence_length:
                continue

            all_sequences.append(sequence)

    return all_sequences

def main():
    # Parameters
    wiki_file = "processed_data/wiki_articles.txt"
    output_dir = "dataset"
    vocab_file = "wiki_vocab.txt"
    vocab_size = 10000
    max_articles = None  # Set to a number to limit articles, or None for all

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load or create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    if os.path.exists(vocab_file):
        print(f"Loading vocabulary from {vocab_file}")
        tokenizer.load_vocab(vocab_file)
    else:
        # Read articles
        print("Reading articles for vocabulary building...")
        articles = read_articles(wiki_file, max_articles)

        # Build vocabulary
        print("Building vocabulary...")
        tokenizer.build_vocab(articles)

        # Save vocabulary
        tokenizer.save_vocab(vocab_file)
        print(f"Saved vocabulary to {vocab_file}")

    # Read articles (again if needed)
    print("Reading articles for dataset creation...")
    articles = read_articles(wiki_file, max_articles)

    # Tokenize articles
    print("Tokenizing articles...")
    tokenized_articles = []
    for article in tqdm(articles, desc="Tokenizing"):
        tokens = tokenizer.encode(article)
        tokenized_articles.append(tokens)

    # Create sequences
    print("Creating sequences...")
    sequences = create_sequences(tokenized_articles, SEQUENCE_LENGTH)
    print(f"Created {len(sequences)} sequences of length {SEQUENCE_LENGTH}")

    # Shuffle sequences
    random.shuffle(sequences)

    # Split into train/test
    split_idx = int(len(sequences) * TRAIN_SPLIT)
    train_sequences = sequences[:split_idx]
    test_sequences = sequences[split_idx:]

    print(f"Training sequences: {len(train_sequences)}")
    print(f"Testing sequences: {len(test_sequences)}")

    # Save datasets
    train_path = os.path.join(output_dir, f"train_seqlen{SEQUENCE_LENGTH}.npy")
    test_path = os.path.join(output_dir, f"test_seqlen{SEQUENCE_LENGTH}.npy")

    np.save(train_path, np.array(train_sequences, dtype=np.int32))
    np.save(test_path, np.array(test_sequences, dtype=np.int32))

    print(f"Saved training dataset to {train_path}")
    print(f"Saved testing dataset to {test_path}")

    # Save a sample for inspection
    sample_path = os.path.join(output_dir, "sample_sequences.txt")
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("=== SAMPLE SEQUENCES ===\n\n")
        for i, seq in enumerate(random.sample(sequences, 5)):
            f.write(f"Sequence {i+1}:\n")
            f.write(f"Token IDs: {seq[:10]}...\n")  # Just first 10 tokens
            decoded = tokenizer.decode(seq)
            f.write(f"Decoded: {decoded[:100]}...\n\n")  # First 100 chars

    print(f"Saved sample sequences to {sample_path}")

if __name__ == "__main__":
    main()
