"""
BPE (Byte Pair Encoding) Tokenizer - From Scratch Implementation

This module implements the BPE algorithm using pure Python and NumPy for pedagogical purposes.
The goal is to demonstrate the mathematical operations and logic behind BPE tokenization
without relying on optimized libraries.

BPE Algorithm Overview:
1. Start with a vocabulary of individual characters
2. Count the frequency of all adjacent character pairs
3. Merge the most frequent pair into a new token
4. Repeat steps 2-3 for a specified number of iterations

Mathematical Formulation:
- Pair frequency: f(a,b) = Σ 1[seq_i = (a,b)] for all sequences
- Merge criterion: (a*, b*) = argmax f(a,b)

Requirements Addressed:
- 1.1: Build vocabulary by iteratively merging most frequent character pairs
- 1.2: Tokenize text into subword units using learned BPE vocabulary
- 1.4: Maintain bidirectional mapping between tokens and integer IDs
"""

from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import re


class SimpleBPETokenizer:
    """
    A from-scratch implementation of Byte Pair Encoding tokenizer.
    
    This implementation uses pure Python data structures to demonstrate
    the BPE algorithm step-by-step for educational purposes.
    
    Attributes:
        num_merges (int): Number of merge operations to perform
        vocab (Set[str]): Set of all tokens in the vocabulary
        merges (List[Tuple[str, str]]): Ordered list of merge operations
        token_to_id (Dict[str, int]): Mapping from token string to integer ID
        id_to_token (Dict[int, str]): Mapping from integer ID to token string
    """
    
    def __init__(self, num_merges: int = 1000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            num_merges: Number of merge operations to perform during training
        """
        self.num_merges = num_merges
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
    
    def train(self, corpus: str) -> None:
        """
        Train the BPE tokenizer on a corpus by learning merge operations.
        
        This method implements the core BPE algorithm:
        1. Initialize vocabulary with individual characters
        2. For each merge iteration:
           a. Count frequency of all adjacent token pairs
           b. Find the most frequent pair
           c. Merge this pair into a new token
           d. Update the vocabulary and merge list
        
        Args:
            corpus: Training text corpus as a single string
        """
        # Step 1: Preprocess corpus - split into words
        # We use a simple regex to split on whitespace while preserving word boundaries
        words = re.findall(r'\S+', corpus)
        
        # Step 2: Initialize vocabulary with individual characters
        # Each word is represented as a list of characters with a special end-of-word marker
        # Example: "hello" -> ['h', 'e', 'l', 'l', 'o', '</w>']
        word_freqs = Counter(words)
        
        # Convert each word to a list of characters (with end-of-word marker)
        split_words = {}
        for word, freq in word_freqs.items():
            # Split word into characters and add end-of-word marker
            chars = list(word) + ['</w>']
            split_words[' '.join(chars)] = freq
        
        # Build initial vocabulary from all unique characters
        for word in split_words.keys():
            for char in word.split():
                self.vocab.add(char)
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        print(f"Initial vocabulary (first 20): {sorted(list(self.vocab))[:20]}")
        
        # Step 3: Perform iterative merging
        for merge_idx in range(self.num_merges):
            # Step 3a: Count frequency of all adjacent pairs
            pair_freqs = self._count_pair_frequencies(split_words)
            
            # Check if we have any pairs left to merge
            if not pair_freqs:
                print(f"No more pairs to merge. Stopping at iteration {merge_idx}")
                break
            
            # Step 3b: Find the most frequent pair
            # This implements: (a*, b*) = argmax f(a,b)
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Step 3c: Merge the best pair
            # Record this merge operation for later use during encoding
            self.merges.append(best_pair)
            
            # Create the new merged token
            new_token = ''.join(best_pair)
            self.vocab.add(new_token)
            
            # Step 3d: Update all words by applying this merge
            split_words = self._merge_pair_in_words(best_pair, split_words)
            
            # Print progress every 100 merges
            if (merge_idx + 1) % 100 == 0:
                print(f"Merge {merge_idx + 1}/{self.num_merges}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' "
                      f"(freq: {pair_freqs[best_pair]})")
        
        # Step 4: Build token-to-ID and ID-to-token mappings
        # This ensures bidirectional mapping as required by Requirement 1.4
        sorted_vocab = sorted(list(self.vocab))
        for idx, token in enumerate(sorted_vocab):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        print(f"\nTraining complete!")
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges performed: {len(self.merges)}")
    
    def _count_pair_frequencies(self, split_words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Count the frequency of all adjacent token pairs in the current vocabulary.
        
        This implements the mathematical formula:
        f(a,b) = Σ 1[seq_i = (a,b)] for all sequences
        
        Args:
            split_words: Dictionary mapping space-separated token sequences to their frequencies
        
        Returns:
            Dictionary mapping token pairs to their total frequency across all words
        """
        pair_freqs = defaultdict(int)
        
        for word, freq in split_words.items():
            tokens = word.split()
            
            # Count all adjacent pairs in this word
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        return dict(pair_freqs)
    
    def _merge_pair_in_words(self, pair: Tuple[str, str], 
                            split_words: Dict[str, int]) -> Dict[str, int]:
        """
        Apply a merge operation to all words in the vocabulary.
        
        This function replaces all occurrences of the specified pair with
        the merged token in all words.
        
        Args:
            pair: The pair of tokens to merge (a, b)
            split_words: Dictionary of space-separated token sequences
        
        Returns:
            Updated dictionary with the pair merged in all words
        """
        new_split_words = {}
        merged_token = ''.join(pair)
        
        for word, freq in split_words.items():
            tokens = word.split()
            
            # Apply the merge operation
            i = 0
            new_tokens = []
            while i < len(tokens):
                # Check if current position matches the pair to merge
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    # Merge the pair
                    new_tokens.append(merged_token)
                    i += 2  # Skip both tokens in the pair
                else:
                    # Keep the token as is
                    new_tokens.append(tokens[i])
                    i += 1
            
            # Store the updated word
            new_word = ' '.join(new_tokens)
            new_split_words[new_word] = freq
        
        return new_split_words
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs using the learned BPE vocabulary.
        
        This method applies the learned merge operations in order to tokenize
        the input text into subword units, then converts them to integer IDs.
        
        Process:
        1. Split text into words
        2. For each word, start with character-level tokens
        3. Apply all learned merges in order
        4. Convert final tokens to IDs
        
        Args:
            text: Input text to encode
        
        Returns:
            List of integer token IDs
        
        Requirements: 1.2 - Tokenize text into subword units
        """
        if not self.merges:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
        
        # Split text into words
        words = re.findall(r'\S+', text)
        
        all_token_ids = []
        
        for word in words:
            # Start with character-level representation
            tokens = list(word) + ['</w>']
            
            # Apply all learned merges in order
            for merge_pair in self.merges:
                tokens = self._apply_merge_to_tokens(merge_pair, tokens)
            
            # Convert tokens to IDs
            for token in tokens:
                if token in self.token_to_id:
                    all_token_ids.append(self.token_to_id[token])
                else:
                    # Handle unknown tokens by encoding character-by-character
                    # This provides graceful fallback as mentioned in Requirement 1.5
                    for char in token:
                        if char in self.token_to_id:
                            all_token_ids.append(self.token_to_id[char])
        
        return all_token_ids
    
    def _apply_merge_to_tokens(self, pair: Tuple[str, str], tokens: List[str]) -> List[str]:
        """
        Apply a single merge operation to a list of tokens.
        
        Args:
            pair: The pair of tokens to merge
            tokens: List of current tokens
        
        Returns:
            Updated list of tokens with the merge applied
        """
        if len(tokens) < 2:
            return tokens
        
        merged_token = ''.join(pair)
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            # Check if we can merge at this position
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.
        
        This method converts integer IDs back to their corresponding tokens,
        then reconstructs the original text by removing end-of-word markers
        and joining tokens appropriately.
        
        Args:
            token_ids: List of integer token IDs
        
        Returns:
            Decoded text string
        
        Requirements: 1.2 - Support round-trip encoding/decoding
        """
        if not self.id_to_token:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                # Unknown ID - skip it
                continue
        
        # Reconstruct text by joining tokens and handling end-of-word markers
        text = ''.join(tokens)
        
        # Replace end-of-word markers with spaces
        text = text.replace('</w>', ' ')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of tokens in the vocabulary
        """
        return len(self.vocab)
    
    def get_token_from_id(self, token_id: int) -> str:
        """
        Get the token string corresponding to a token ID.
        
        Args:
            token_id: Integer token ID
        
        Returns:
            Token string
        
        Requirements: 1.4 - Bidirectional token-ID mapping
        """
        return self.id_to_token.get(token_id, '<UNK>')
    
    def get_id_from_token(self, token: str) -> int:
        """
        Get the token ID corresponding to a token string.
        
        Args:
            token: Token string
        
        Returns:
            Integer token ID, or -1 if token not in vocabulary
        
        Requirements: 1.4 - Bidirectional token-ID mapping
        """
        return self.token_to_id.get(token, -1)
    
    def display_merge_examples(self, num_examples: int = 10) -> None:
        """
        Display examples of learned merge operations for pedagogical purposes.
        
        Args:
            num_examples: Number of merge examples to display
        """
        print(f"\nFirst {num_examples} merge operations:")
        print("-" * 60)
        for i, (token1, token2) in enumerate(self.merges[:num_examples]):
            merged = ''.join([token1, token2])
            print(f"{i+1}. '{token1}' + '{token2}' -> '{merged}'")
        print("-" * 60)


# Example usage for pedagogical demonstration
if __name__ == "__main__":
    # Example 1: Simple French corpus
    print("=" * 70)
    print("Example 1: Training on simple French words")
    print("=" * 70)
    
    simple_corpus = "bas basse basses bassement"
    tokenizer = SimpleBPETokenizer(num_merges=5)
    tokenizer.train(simple_corpus)
    tokenizer.display_merge_examples(5)
    
    # Test encoding and decoding
    test_text = "basse"
    print(f"\nTest encoding: '{test_text}'")
    encoded = tokenizer.encode(test_text)
    print(f"Encoded IDs: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: '{decoded}'")
    
    # Example 2: Slightly larger corpus
    print("\n" + "=" * 70)
    print("Example 2: Training on a larger French corpus")
    print("=" * 70)
    
    french_corpus = """
    le chat est sur le tapis
    le chien est dans le jardin
    le chat mange la souris
    le chien court dans le parc
    """
    
    tokenizer2 = SimpleBPETokenizer(num_merges=20)
    tokenizer2.train(french_corpus)
    tokenizer2.display_merge_examples(10)
    
    # Test encoding and decoding
    test_text2 = "le chat mange"
    print(f"\nTest encoding: '{test_text2}'")
    encoded2 = tokenizer2.encode(test_text2)
    print(f"Encoded IDs: {encoded2}")
    decoded2 = tokenizer2.decode(encoded2)
    print(f"Decoded text: '{decoded2}'")
    
    print("\n" + "=" * 70)
    print("BPE From-Scratch Implementation Complete!")
    print("=" * 70)
