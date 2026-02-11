# Assignment-1: Introduction to NLP, IIIT, Hyderabad
This project implements the three tokenizers(viz. WhiteSpace, Regex and BPE) and langauge models(viz. Witten-Bell and Kneser-Ney).


## 📂 File Structure

The project is organized as follows:

```text
Assignment-1/
├── dataset/
│   ├── corpora/                # Raw corpus files
│   │   ├── cc100_en.jsonl
│   │   └── cc100_mn.jsonl
│   └── partitions/             # Data splits
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── environment.yaml            # Conda environment configuration
├── language_models.py          # Language model architecture classes
├── tokenizers.py               # Tokenizer logic (BPE) and training script
├── README.md                   # Project documentation
└── .gitignore                  # Files to ignore (e.g., __pycache__, large data)
```



## To clean and tokenize
```bash
python tokenizers.py --mode {corpus_clean,tokenization} --tokenizer {WhitespaceTokenizer, RegexTokenizer, BPETokenizer} --input dataset/corpora/cc100_{en, mn}.jsonl --output dataset/corpora/partitions/mongolean --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

## To run langauge models
```bash
python language_models.py --train dataset/corpora/partitions/train.txt --test dataset/corpora/partitions/test.txt --tokenizer {whitespace, regex, bpe} --smoothing {none, witten-bell, kneser-ney} 
```