import re
import collections
from typing import List

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 10000):
        """Initialize a simple word-level tokenizer.

        Args:
            vocab_size: Maximum size of vocabulary
        """
        self.vocab_size = vocab_size
        self.word_counts = collections.Counter()
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token

        # Special tokens
        self.pad_token = "[PAD]"  # Used for padding sequences to same length
        self.unk_token = "[UNK]"  # Used for unknown tokens
        self.bos_token = "[BOS]"  # Beginning of sequence
        self.eos_token = "[EOS]"  # End of sequence

        # Add special tokens to vocabulary
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        # Simple regex to capture words and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of documents/texts to process
        """
        # Count word frequencies
        total_tokens = 0
        unique_tokens = set()

        for text in texts:
            tokens = self.tokenize(text)
            total_tokens += len(tokens)
            unique_tokens.update(tokens)
            self.word_counts.update(tokens)

        # Print token statistics before building vocab
        print(f"Total tokens in corpus: {total_tokens}")
        print(f"Unique tokens in corpus: {len(unique_tokens)}")

        # Select top words by frequency
        num_special = len(self.special_tokens)
        num_words = min(self.vocab_size - num_special, len(self.word_counts))

        # Add most common words to vocabulary
        for word, count in self.word_counts.most_common(num_words):
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word

        print(f"Vocabulary built with {len(self.vocab)} tokens")

        # Calculate coverage
        total_occurrences = sum(self.word_counts.values())
        vocab_occurrences = sum(self.word_counts[word] for word in self.vocab if word not in self.special_tokens)
        coverage = vocab_occurrences / total_occurrences * 100 if total_occurrences > 0 else 0
        print(f"Vocabulary covers {coverage:.2f}% of all token occurrences")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert text to a sequence of token IDs."""
        tokens = self.tokenize(text)
        ids = []

        # Add BOS token if requested
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])

        # Convert tokens to IDs
        for token in tokens:
            # Use UNK for tokens not in vocabulary
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.unk_token])

        # Add EOS token if requested
        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        tokens = []

        for idx in ids:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]

                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue

                tokens.append(token)
            else:
                tokens.append(self.unk_token)

        # Simple space-joining (not perfect for punctuation)
        text = ' '.join(tokens)

        # Fix spacing around punctuation (basic)
        text = re.sub(r'\s([,.!?;:])', r'\1', text)

        return text

    def save_vocab(self, path: str) -> None:
        """Save vocabulary to a file."""
        with open(path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")

    def load_vocab(self, path: str) -> None:
        """Load vocabulary from a file."""
        self.vocab = {}
        self.inverse_vocab = {}

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                idx = int(idx)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token

        print(f"Loaded vocabulary with {len(self.vocab)} tokens")

    def print_vocab_stats(self):
        """Print detailed vocabulary statistics."""
        print("\n=== Vocabulary Statistics ===")
        print(f"Total vocabulary size: {len(self.vocab)}")
        print(f"Number of special tokens: {len(self.special_tokens)}")
        print(f"Number of regular tokens: {len(self.vocab) - len(self.special_tokens)}")

        # Token length distribution
        token_lengths = [len(token) for token in self.vocab.keys() if token not in self.special_tokens]
        avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        print(f"Average token length: {avg_length:.2f} characters")

        # Most common tokens
        print("\n= Most common tokens =")
        for token, count in self.word_counts.most_common(20):
            if token in self.vocab:
                print(f"'{token}': {count} occurrences (ID: {self.vocab[token]})")

        # Least common tokens in vocabulary
        print("\n= Least common tokens in vocabulary =")
        least_common = sorted([(token, self.word_counts[token])
                              for token in self.vocab
                              if token not in self.special_tokens],
                             key=lambda x: x[1])[:10]

        for token, count in least_common:
            print(f"'{token}': {count} occurrences (ID: {self.vocab[token]})")
