"""
BPE (Byte Pair Encoding) Tokenizer - Professional Implementation

This module implements an optimized, production-ready BPE tokenizer with:
- Efficient encoding/decoding with error handling
- Vocabulary persistence (save/load functionality)
- Robust unknown character handling with fallback strategies
- Input validation and comprehensive error messages

This implementation builds upon the from-scratch version but adds:
- Better performance through optimized data structures
- Comprehensive error handling for edge cases
- Serialization support for model persistence
- Special tokens support (PAD, UNK, BOS, EOS)
- Detailed logging and debugging capabilities

Requirements Addressed:
- 1.1: Build vocabulary by iteratively merging most frequent character pairs
- 1.2: Tokenize text into subword units using learned BPE vocabulary
- 1.4: Maintain bidirectional mapping between tokens and integer IDs
- 1.5: Handle unknown characters gracefully with fallback to character-level encoding
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import Counter, defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BPETokenizer:
    """
    Professional BPE tokenizer with optimized implementation and error handling.
    
    This tokenizer provides:
    - Efficient training on large corpora
    - Robust encoding/decoding with error handling
    - Vocabulary persistence (save/load)
    - Unknown character handling with multiple fallback strategies
    - Special tokens support (PAD, UNK, BOS, EOS)
    
    Attributes:
        num_merges (int): Number of merge operations to perform
        vocab (Set[str]): Set of all tokens in the vocabulary
        merges (List[Tuple[str, str]]): Ordered list of merge operations
        token_to_id (Dict[str, int]): Mapping from token string to integer ID
        id_to_token (Dict[int, str]): Mapping from integer ID to token string
        special_tokens (Dict[str, str]): Special tokens (PAD, UNK, BOS, EOS)
        unk_token (str): Token used for unknown characters
        pad_token (str): Token used for padding
        bos_token (str): Beginning of sequence token
        eos_token (str): End of sequence token
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    EOW_TOKEN = '</w>'  # End of word marker
    
    def __init__(
        self,
        num_merges: int = 1000,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        use_special_tokens: bool = True
    ):
        """
        Initialize the professional BPE tokenizer.
        
        Args:
            num_merges: Number of merge operations to perform during training
            unk_token: Token for unknown characters (default: <UNK>)
            pad_token: Token for padding (default: <PAD>)
            bos_token: Beginning of sequence token (default: <BOS>)
            eos_token: End of sequence token (default: <EOS>)
            use_special_tokens: Whether to include special tokens in vocabulary
        """
        self.num_merges = num_merges
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Special tokens
        self.unk_token = unk_token or self.UNK_TOKEN
        self.pad_token = pad_token or self.PAD_TOKEN
        self.bos_token = bos_token or self.BOS_TOKEN
        self.eos_token = eos_token or self.EOS_TOKEN
        self.use_special_tokens = use_special_tokens
        
        # Training state
        self._is_trained = False
        
        logger.info(f"Initialized BPETokenizer with {num_merges} merges")
    
    def train(self, corpus: Union[str, List[str]], verbose: bool = True) -> None:
        """
        Train the BPE tokenizer on a corpus.
        
        This method implements the BPE algorithm with optimizations:
        1. Initialize vocabulary with individual characters
        2. Iteratively merge the most frequent adjacent token pairs
        3. Build token-to-ID and ID-to-token mappings
        4. Add special tokens to vocabulary
        
        Args:
            corpus: Training text corpus (string or list of strings)
            verbose: Whether to print training progress
            
        Raises:
            ValueError: If corpus is empty or invalid
            TypeError: If corpus is not a string or list of strings
        """
        # Input validation
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        
        if isinstance(corpus, list):
            if not all(isinstance(text, str) for text in corpus):
                raise TypeError("All corpus elements must be strings")
            corpus = ' '.join(corpus)
        elif not isinstance(corpus, str):
            raise TypeError("Corpus must be a string or list of strings")
        
        if verbose:
            logger.info("Starting BPE training...")
            logger.info(f"Corpus length: {len(corpus)} characters")
        
        # Step 1: Preprocess corpus - split into words
        words = re.findall(r'\S+', corpus)
        
        if not words:
            raise ValueError("Corpus contains no valid words")
        
        word_freqs = Counter(words)
        
        if verbose:
            logger.info(f"Found {len(words)} total words, {len(word_freqs)} unique words")
        
        # Step 2: Initialize vocabulary with character-level tokens
        split_words = {}
        for word, freq in word_freqs.items():
            # Split word into characters and add end-of-word marker
            chars = list(word) + [self.EOW_TOKEN]
            split_words[' '.join(chars)] = freq
        
        # Build initial vocabulary from all unique characters
        for word in split_words.keys():
            for char in word.split():
                self.vocab.add(char)
        
        if verbose:
            logger.info(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Step 3: Perform iterative merging
        for merge_idx in range(self.num_merges):
            # Count frequency of all adjacent pairs
            pair_freqs = self._count_pair_frequencies(split_words)
            
            # Check if we have any pairs left to merge
            if not pair_freqs:
                if verbose:
                    logger.info(f"No more pairs to merge. Stopping at iteration {merge_idx}")
                break
            
            # Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Record this merge operation
            self.merges.append(best_pair)
            
            # Create the new merged token
            new_token = ''.join(best_pair)
            self.vocab.add(new_token)
            
            # Update all words by applying this merge
            split_words = self._merge_pair_in_words(best_pair, split_words)
            
            # Print progress
            if verbose and (merge_idx + 1) % 100 == 0:
                logger.info(
                    f"Merge {merge_idx + 1}/{self.num_merges}: "
                    f"'{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' "
                    f"(freq: {pair_freqs[best_pair]})"
                )
        
        # Step 4: Add special tokens to vocabulary
        if self.use_special_tokens:
            self.vocab.add(self.pad_token)
            self.vocab.add(self.unk_token)
            self.vocab.add(self.bos_token)
            self.vocab.add(self.eos_token)
        
        # Step 5: Build token-to-ID and ID-to-token mappings
        # Special tokens get the first IDs for consistency
        sorted_vocab = []
        if self.use_special_tokens:
            sorted_vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Add remaining tokens in sorted order
        remaining_tokens = sorted(list(self.vocab - set(sorted_vocab)))
        sorted_vocab.extend(remaining_tokens)
        
        for idx, token in enumerate(sorted_vocab):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self._is_trained = True
        
        if verbose:
            logger.info("Training complete!")
            logger.info(f"Final vocabulary size: {len(self.vocab)}")
            logger.info(f"Number of merges performed: {len(self.merges)}")
    
    def _count_pair_frequencies(self, split_words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Count the frequency of all adjacent token pairs.
        
        Args:
            split_words: Dictionary mapping space-separated token sequences to frequencies
        
        Returns:
            Dictionary mapping token pairs to their total frequency
        """
        pair_freqs = defaultdict(int)
        
        for word, freq in split_words.items():
            tokens = word.split()
            
            # Count all adjacent pairs in this word
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        return dict(pair_freqs)
    
    def _merge_pair_in_words(
        self,
        pair: Tuple[str, str],
        split_words: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Apply a merge operation to all words in the vocabulary.
        
        Args:
            pair: The pair of tokens to merge
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
                if (i < len(tokens) - 1 and 
                    tokens[i] == pair[0] and 
                    tokens[i + 1] == pair[1]):
                    # Merge the pair
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    # Keep the token as is
                    new_tokens.append(tokens[i])
                    i += 1
            
            # Store the updated word
            new_word = ' '.join(new_tokens)
            new_split_words[new_word] = freq
        
        return new_split_words
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        handle_unknown: str = 'char_fallback'
    ) -> List[int]:
        """
        Encode text into a list of token IDs with error handling.
        
        This method applies the learned merge operations to tokenize the input
        text, then converts tokens to integer IDs. Unknown characters are handled
        gracefully using the specified strategy.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            handle_unknown: Strategy for unknown characters:
                - 'char_fallback': Fall back to character-level encoding (default)
                - 'unk_token': Replace with UNK token
                - 'skip': Skip unknown characters
        
        Returns:
            List of integer token IDs
            
        Raises:
            ValueError: If tokenizer is not trained or text is invalid
            KeyError: If handle_unknown strategy is invalid
        
        Requirements:
            - 1.2: Tokenize text into subword units
            - 1.5: Handle unknown characters gracefully
        """
        # Validation
        if not self._is_trained:
            raise ValueError(
                "Tokenizer has not been trained. Call train() first."
            )
        
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text)}")
        
        if not text:
            return []
        
        if handle_unknown not in ['char_fallback', 'unk_token', 'skip']:
            raise KeyError(
                f"Invalid handle_unknown strategy: {handle_unknown}. "
                f"Must be one of: 'char_fallback', 'unk_token', 'skip'"
            )
        
        # Split text into words
        words = re.findall(r'\S+', text)
        
        all_token_ids = []
        
        # Add BOS token if requested
        if add_special_tokens and self.use_special_tokens:
            all_token_ids.append(self.token_to_id[self.bos_token])
        
        # Encode each word
        for word in words:
            # Start with character-level representation
            tokens = list(word) + [self.EOW_TOKEN]
            
            # Apply all learned merges in order
            for merge_pair in self.merges:
                tokens = self._apply_merge_to_tokens(merge_pair, tokens)
            
            # Convert tokens to IDs with unknown handling
            for token in tokens:
                if token in self.token_to_id:
                    all_token_ids.append(self.token_to_id[token])
                else:
                    # Handle unknown token based on strategy
                    all_token_ids.extend(
                        self._handle_unknown_token(token, handle_unknown)
                    )
        
        # Add EOS token if requested
        if add_special_tokens and self.use_special_tokens:
            all_token_ids.append(self.token_to_id[self.eos_token])
        
        return all_token_ids
    
    def _handle_unknown_token(
        self,
        token: str,
        strategy: str
    ) -> List[int]:
        """
        Handle unknown tokens using the specified strategy.
        
        Args:
            token: Unknown token string
            strategy: Handling strategy ('char_fallback', 'unk_token', 'skip')
        
        Returns:
            List of token IDs (may be empty if strategy is 'skip')
        
        Requirements: 1.5 - Handle unknown characters gracefully
        """
        if strategy == 'char_fallback':
            # Fall back to character-level encoding
            char_ids = []
            for char in token:
                if char in self.token_to_id:
                    char_ids.append(self.token_to_id[char])
                elif self.use_special_tokens:
                    # If even the character is unknown, use UNK token
                    char_ids.append(self.token_to_id[self.unk_token])
            return char_ids
        
        elif strategy == 'unk_token':
            # Replace with UNK token
            if self.use_special_tokens:
                return [self.token_to_id[self.unk_token]]
            else:
                return []
        
        elif strategy == 'skip':
            # Skip unknown tokens
            return []
        
        return []
    
    def _apply_merge_to_tokens(
        self,
        pair: Tuple[str, str],
        tokens: List[str]
    ) -> List[str]:
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
            if (i < len(tokens) - 1 and 
                tokens[i] == pair[0] and 
                tokens[i + 1] == pair[1]):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode a list of token IDs back into text with error handling.
        
        This method converts integer IDs back to their corresponding tokens,
        then reconstructs the original text.
        
        Args:
            token_ids: List of integer token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
            
        Raises:
            ValueError: If tokenizer is not trained
            TypeError: If token_ids is not a list
        
        Requirements: 1.2 - Support round-trip encoding/decoding
        """
        # Validation
        if not self._is_trained:
            raise ValueError(
                "Tokenizer has not been trained. Call train() first."
            )
        
        if not isinstance(token_ids, list):
            raise TypeError(f"token_ids must be a list, got {type(token_ids)}")
        
        if not token_ids:
            return ""
        
        # Convert IDs to tokens
        tokens = []
        special_tokens_set = {
            self.pad_token, self.unk_token, 
            self.bos_token, self.eos_token
        } if self.use_special_tokens else set()
        
        for token_id in token_ids:
            if not isinstance(token_id, int):
                logger.warning(f"Skipping non-integer token ID: {token_id}")
                continue
            
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in special_tokens_set:
                    continue
                
                tokens.append(token)
            else:
                # Unknown ID - log warning and skip
                logger.warning(f"Unknown token ID: {token_id}")
                continue
        
        # Reconstruct text by joining tokens and handling end-of-word markers
        text = ''.join(tokens)
        
        # Replace end-of-word markers with spaces
        text = text.replace(self.EOW_TOKEN, ' ')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def save_vocabulary(self, filepath: Union[str, Path]) -> None:
        """
        Save the tokenizer vocabulary and merge rules to a file.
        
        This method serializes the tokenizer state to enable model persistence.
        The saved file can be loaded later to restore the tokenizer.
        
        Args:
            filepath: Path to save the vocabulary file
            
        Raises:
            ValueError: If tokenizer is not trained
            IOError: If file cannot be written
        
        Requirements: Vocabulary save functionality
        """
        if not self._is_trained:
            raise ValueError(
                "Tokenizer has not been trained. Call train() first."
            )
        
        filepath = Path(filepath)
        
        # Prepare data to save
        vocab_data = {
            'num_merges': self.num_merges,
            'vocab': list(self.vocab),
            'merges': self.merges,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'special_tokens': {
                'unk_token': self.unk_token,
                'pad_token': self.pad_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
            },
            'use_special_tokens': self.use_special_tokens,
        }
        
        try:
            # Save as JSON for human readability
            if filepath.suffix == '.json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Vocabulary saved to {filepath} (JSON format)")
            
            # Save as pickle for efficiency
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(vocab_data, f)
                logger.info(f"Vocabulary saved to {filepath} (pickle format)")
        
        except IOError as e:
            raise IOError(f"Failed to save vocabulary to {filepath}: {e}")
    
    def load_vocabulary(self, filepath: Union[str, Path]) -> None:
        """
        Load the tokenizer vocabulary and merge rules from a file.
        
        This method deserializes a previously saved tokenizer state.
        
        Args:
            filepath: Path to the vocabulary file
            
        Raises:
            FileNotFoundError: If file does not exist
            IOError: If file cannot be read
            ValueError: If file format is invalid
        
        Requirements: Vocabulary load functionality
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        try:
            # Load from JSON
            if filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                logger.info(f"Vocabulary loaded from {filepath} (JSON format)")
            
            # Load from pickle
            else:
                with open(filepath, 'rb') as f:
                    vocab_data = pickle.load(f)
                logger.info(f"Vocabulary loaded from {filepath} (pickle format)")
            
            # Restore tokenizer state
            self.num_merges = vocab_data['num_merges']
            self.vocab = set(vocab_data['vocab'])
            self.merges = [tuple(pair) for pair in vocab_data['merges']]
            self.token_to_id = vocab_data['token_to_id']
            self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
            
            # Restore special tokens
            special_tokens = vocab_data.get('special_tokens', {})
            self.unk_token = special_tokens.get('unk_token', self.UNK_TOKEN)
            self.pad_token = special_tokens.get('pad_token', self.PAD_TOKEN)
            self.bos_token = special_tokens.get('bos_token', self.BOS_TOKEN)
            self.eos_token = special_tokens.get('eos_token', self.EOS_TOKEN)
            self.use_special_tokens = vocab_data.get('use_special_tokens', True)
            
            self._is_trained = True
            
            logger.info(f"Loaded vocabulary with {len(self.vocab)} tokens")
        
        except (IOError, json.JSONDecodeError, pickle.UnpicklingError) as e:
            raise IOError(f"Failed to load vocabulary from {filepath}: {e}")
        except KeyError as e:
            raise ValueError(f"Invalid vocabulary file format: missing key {e}")
    
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
            Token string, or UNK token if ID not found
        
        Requirements: 1.4 - Bidirectional token-ID mapping
        """
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained")
        
        return self.id_to_token.get(
            token_id,
            self.unk_token if self.use_special_tokens else '<UNK>'
        )
    
    def get_id_from_token(self, token: str) -> int:
        """
        Get the token ID corresponding to a token string.
        
        Args:
            token: Token string
        
        Returns:
            Integer token ID, or UNK token ID if token not in vocabulary
        
        Requirements: 1.4 - Bidirectional token-ID mapping
        """
        if not self._is_trained:
            raise ValueError("Tokenizer has not been trained")
        
        return self.token_to_id.get(
            token,
            self.token_to_id.get(self.unk_token, -1) if self.use_special_tokens else -1
        )
    
    def is_trained(self) -> bool:
        """
        Check if the tokenizer has been trained.
        
        Returns:
            True if trained, False otherwise
        """
        return self._is_trained
    
    def get_special_tokens(self) -> Dict[str, int]:
        """
        Get the special tokens and their IDs.
        
        Returns:
            Dictionary mapping special token names to their IDs
        """
        if not self.use_special_tokens:
            return {}
        
        return {
            'pad': self.token_to_id.get(self.pad_token, -1),
            'unk': self.token_to_id.get(self.unk_token, -1),
            'bos': self.token_to_id.get(self.bos_token, -1),
            'eos': self.token_to_id.get(self.eos_token, -1),
        }
    
    def display_merge_examples(self, num_examples: int = 10) -> None:
        """
        Display examples of learned merge operations for pedagogical purposes.
        
        Args:
            num_examples: Number of merge examples to display
        """
        if not self._is_trained:
            logger.warning("Tokenizer has not been trained")
            return
        
        print(f"\nFirst {num_examples} merge operations:")
        print("-" * 60)
        for i, (token1, token2) in enumerate(self.merges[:num_examples]):
            merged = ''.join([token1, token2])
            print(f"{i+1}. '{token1}' + '{token2}' -> '{merged}'")
        print("-" * 60)
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        status = "trained" if self._is_trained else "untrained"
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, "
            f"num_merges={len(self.merges)}, status={status})"
        )


# Example usage demonstrating professional features
if __name__ == "__main__":
    print("=" * 70)
    print("Professional BPE Tokenizer - Example Usage")
    print("=" * 70)
    
    # Example 1: Basic training and encoding
    print("\n1. Basic Training and Encoding")
    print("-" * 70)
    
    french_corpus = """
    le chat est sur le tapis
    le chien est dans le jardin
    le chat mange la souris
    le chien court dans le parc
    la souris est petite
    le tapis est grand
    """
    
    tokenizer = BPETokenizer(num_merges=30)
    tokenizer.train(french_corpus, verbose=True)
    tokenizer.display_merge_examples(10)
    
    # Test encoding
    test_text = "le chat mange la souris"
    print(f"\nEncoding: '{test_text}'")
    encoded = tokenizer.encode(test_text)
    print(f"Token IDs: {encoded}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    print(f"Round-trip successful: {test_text == decoded}")
    
    # Example 2: Unknown character handling
    print("\n\n2. Unknown Character Handling")
    print("-" * 70)
    
    unknown_text = "le chat xyz123"
    print(f"Text with unknown characters: '{unknown_text}'")
    
    # Strategy 1: Character fallback (default)
    encoded_fallback = tokenizer.encode(unknown_text, handle_unknown='char_fallback')
    print(f"Char fallback: {encoded_fallback}")
    print(f"Decoded: '{tokenizer.decode(encoded_fallback)}'")
    
    # Strategy 2: UNK token
    encoded_unk = tokenizer.encode(unknown_text, handle_unknown='unk_token')
    print(f"UNK token: {encoded_unk}")
    print(f"Decoded: '{tokenizer.decode(encoded_unk)}'")
    
    # Example 3: Save and load vocabulary
    print("\n\n3. Vocabulary Persistence")
    print("-" * 70)
    
    # Save vocabulary
    save_path = Path("vocab_test.json")
    tokenizer.save_vocabulary(save_path)
    print(f"Vocabulary saved to {save_path}")
    
    # Load vocabulary into a new tokenizer
    new_tokenizer = BPETokenizer()
    new_tokenizer.load_vocabulary(save_path)
    print(f"Vocabulary loaded: {new_tokenizer}")
    
    # Verify it works
    test_encoded = new_tokenizer.encode("le chat")
    print(f"Encoding with loaded tokenizer: {test_encoded}")
    print(f"Decoded: '{new_tokenizer.decode(test_encoded)}'")
    
    # Clean up
    save_path.unlink()
    print(f"Cleaned up test file")
    
    # Example 4: Special tokens
    print("\n\n4. Special Tokens")
    print("-" * 70)
    
    special_tokens = tokenizer.get_special_tokens()
    print("Special token IDs:")
    for name, token_id in special_tokens.items():
        print(f"  {name.upper()}: {token_id}")
    
    # Encoding with special tokens
    encoded_special = tokenizer.encode("le chat", add_special_tokens=True)
    print(f"\nWith BOS/EOS: {encoded_special}")
    print(f"Decoded (skip special): '{tokenizer.decode(encoded_special, skip_special_tokens=True)}'")
    print(f"Decoded (keep special): '{tokenizer.decode(encoded_special, skip_special_tokens=False)}'")
    
    print("\n" + "=" * 70)
    print("Professional BPE Tokenizer - Examples Complete!")
    print("=" * 70)
