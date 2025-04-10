from simple_tokenizer import SimpleTokenizer
import time

def main():
    start_time = time.time()

    # Initialize tokenizer
    vocab_size = 10000  # Feel free to adjust this
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    # Read a sample of articles
    print("Reading articles...")
    articles = []
    total_chars = 0

    with open("processed_data/wiki_articles.txt", 'r', encoding='utf-8') as f:
        current_article = []
        for line in f:
            # Check for end of article marker
            if line.strip() == "<|endoftext|>":
                if current_article:
                    article_text = "".join(current_article)
                    articles.append(article_text)
                    total_chars += len(article_text)
                    current_article = []
            else:
                current_article.append(line)

            # For testing, limit to first 500 articles
            if len(articles) >= 500:
                break

    print(f"Loaded {len(articles)} articles ({total_chars:,} characters)")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    tokenizer.build_vocab(articles)

    # Print detailed stats
    tokenizer.print_vocab_stats()

    # Test encoding and decoding
    print("\n=== Encoding/Decoding Test ===")
    test_texts = [
        "The capital of France is Paris, located on the Seine river.",
        "In August 1991, several countries regained independence from the Soviet Union.",
        "Natural language processing uses machine learning to analyze text."
    ]

    for test_text in test_texts:
        print(f"\nOriginal: {test_text}")

        # Encode
        token_ids = tokenizer.encode(test_text)
        tokens = [tokenizer.inverse_vocab[idx] for idx in token_ids]

        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")

        # Count unknown tokens
        unknown_count = token_ids.count(tokenizer.vocab[tokenizer.unk_token])
        if unknown_count > 0:
            print(f"Contains {unknown_count} unknown tokens")

        # Decode
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded_text}")

    # Save vocab
    tokenizer.save_vocab("wiki_vocab.txt")
    print(f"\nSaved vocabulary to wiki_vocab.txt")

    # Print runtime
    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
