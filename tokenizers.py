#@Author: Akash Manna, 2024801008
import os
import re
import unicodedata
import random
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def remove_text_tag(line):
    match = re.search(r'\{"text":\s"(.*?)"\}', line)
    return match.group(1) if match else ''

# Function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        r'['
        r'\U0001F600-\U0001F64F'  
        r'\U0001F300-\U0001F5FF'  
        r'\U0001F680-\U0001F6FF'  
        r'\U0001F1E0-\U0001F1FF'  
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251'
        r']+', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Function to remove forward slashes
def remove_forward_slashes(text):
    return text.replace('/', '').replace('\\', '')

def clean_text(text):
    text = remove_text_tag(text)
    text = remove_emojis(text)
    text = remove_forward_slashes(text)
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFKC', text) #Converting canonically similar word --> uniform format
    text = ''.join(c for c in text if c.isprintable())
    return text.strip()

def partition_corpus(input_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    random.seed(seed)
    lines = []
    count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned = clean_text(line)
            if cleaned:
                lines.append(cleaned)
            count += 1
            if count % 1000 == 0:
                logging.info(f"Processed {count} lines so far...")
                #NOTE: For dev
                # if count>10000:
                #     break
    logging.info(f"Total lines processed: {count}")
    total = len(lines)
    indices = list(range(total))
    random.shuffle(indices)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    os.makedirs(output_dir, exist_ok=True)
    def write_partition(name, idx):
        out_path = os.path.join(output_dir, f'{name}.txt')
        with open(out_path, 'w', encoding='utf-8') as out:
            for i in idx:
                out.write(lines[i] + '\n')
    write_partition('train', train_idx)
    write_partition('validation', val_idx)
    write_partition('test', test_idx)
    print(f"Corpus partitioned: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print(f"Ratios: train={train_ratio}, validation={val_ratio}, test={test_ratio}")
    
class WhitespaceTokenizer:
    def tokenize(self, text):
        # Assumption: Special characters are treated as tokens
        tokens = []
        for token in text.split():
            subtokens = re.findall(r'\w+|[^\w\s]', token)
            tokens.extend(subtokens)
        return tokens

class RegexTokenizer:
    def __init__(self, pattern=None):
        # Assumption: User can define pattern; default splits on word boundaries and punctuation
        self.pattern = pattern or r'\w+|[^\w\s]'
    def tokenize(self, text):
        return re.findall(self.pattern, text)

class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = {}
        self.bpe_codes = {}
        self.is_trained = False

    def build_vocab(self, corpus):
        vocab = {}
        for line in corpus:
            for word in line.strip().split():
                chars = ' '.join(list(word)) + ' </w>'
                vocab[chars] = vocab.get(chars, 0) + 1
        return vocab

    def merge_word(self, word, pair):
        symbols = word.split()
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                new_symbols.append(symbols[i] + symbols[i+1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols

    def learn_bpe(self, vocab):
        for i in range(self.num_merges):
            pairs = {}
            for word, freq in vocab.items():
                symbols = word.split()
                for j in range(len(symbols)-1):
                    pair = (symbols[j], symbols[j+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_codes[best] = i
            new_vocab = {}
            for word in vocab:
                new_word = ' '.join(self.merge_word(word, best))
                new_vocab[new_word] = vocab[word]
            vocab = new_vocab
        self.vocab = vocab
        self.is_trained = True

    def fit(self, corpus):
        vocab = self.build_vocab(corpus)
        self.learn_bpe(vocab)

    def tokenize(self, text):
        if not self.is_trained:
            raise RuntimeError("BPETokenizer must be trained with .fit(corpus) before tokenizing.")
        tokens = []
        for word in text.strip().split():
            word = list(word)
            word.append('</w>')
            while True:
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                candidates = [pair for pair in pairs if pair in self.bpe_codes]
                if not candidates:
                    break
                best = min(candidates, key=lambda pair: self.bpe_codes[pair])
                i = 0
                while i < len(word)-1:
                    if (word[i], word[i+1]) == best:
                        word[i:i+2] = [word[i]+word[i+1]]
                        break
                    i += 1
            tokens.extend(word)
        return tokens

# Driver code
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type= str, default= "clean_corpus", choices= ["clean_corpus", "tokenization"])
    parser.add_argument('--input', type=str, default="dataset/corpora/cc100_en.jsonl", required=True, help= "input file path")
    parser.add_argument('--output', type=str, default= "dataset/corpora/partitions", required=True, help= "ouput directory path to store partions")
    parser.add_argument('--train_ratio', type=float, default=0.8, required=True)
    parser.add_argument('--val_ratio', type=float, default= 0.1, required=True)
    parser.add_argument('--test_ratio', type=float, default= 0.1, required= True)
    parser.add_argument("--tokenizer", type= str, default= "WhitespaceTokenizer", required= False)

    args= parser.parse_args()
    if args.mode== "clean_corpus":
        partition_corpus(
            args.input,
            args.output,
            train_ratio=args.train_ratio, val_ratio= args.val_ratio, test_ratio=args.test_ratio
        )

    elif args.mode == "tokenization":
        if args.tokenizer == "WhitespaceTokenizer":
            tokenizer = WhitespaceTokenizer()
        elif args.tokenizer == "RegexTokenizer":
            tokenizer = RegexTokenizer(r"\w+(?:'\w+)?|[^\w\s]") #don't stays don't
        elif args.tokenizer == "BPETokenizer":
            tokenizer = BPETokenizer()
            with open(args.output + "/train.txt", 'r', encoding='utf-8') as f_train:
                train_corpus = [line.strip() for line in f_train if line.strip()]
            tokenizer.fit(train_corpus)
        else:
            raise ValueError("Unknown tokenizer type")

        with open(args.output+ "/test.txt", 'r', encoding='utf-8') as f:
            for line in f:
                tokens = tokenizer.tokenize(line.strip())
                print(tokens)


           
