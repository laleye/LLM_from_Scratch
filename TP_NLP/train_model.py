import pandas as pd
import sentencepiece as spm

# Load corpus
df = pd.read_csv('corpus/corpus_multilingue.csv')
sw_text = df[df['language'] == 'sw']['text'].tolist()

# Write to a temporary file
with open('swahili_corpus.txt', 'w', encoding='utf-8') as f:
    for line in sw_text:
        f.write(line + '\n')

# Train sentencepiece model
spm.SentencePieceTrainer.train(
    input='swahili_corpus.txt',
    model_prefix='models/bpe_swahili_adapted',
    vocab_size=200,
    model_type='bpe',
    character_coverage=1.0
)
print("Model trained successfully!")
