#@Author: Akash Manna, 2024801008
import os
import re
import unicodedata
import random
import logging


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
	return text.replace('/', '')

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
				if count>10000:
					break
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
	


# 1. Whitespace-based tokenizer
class WhitespaceTokenizer:
	def tokenize(self, text):
		# Assumption: Special characters are treated as tokens
		tokens = []
		for token in text.split():
			# Split off special characters
			subtokens = re.findall(r'\w+|[^\w\s]', token)
			tokens.extend(subtokens)
		return tokens

# 2. Regex-based tokenizer
class RegexTokenizer:
	def __init__(self, pattern=None):
		# Assumption: User can define pattern; default splits on word boundaries and punctuation
		self.pattern = pattern or r'\w+|[^\w\s]'
	def tokenize(self, text):
		return re.findall(self.pattern, text)

# 3. Byte Pair Encoding (BPE) tokenizer
class BPETokenizer:
	def __init__(self, num_merges=1000):
		self.num_merges = num_merges
		self.vocab = {}
		self.bpe_codes = {}

	def get_vocab(self, corpus):
		vocab = {}
		for line in corpus:
			for word in line.strip().split():
				word = ' '.join(list(word)) + ' </w>'
				vocab[word] = vocab.get(word, 0) + 1
		return vocab

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

	def merge_word(self, word, pair):
		symbols = word.split()
		i = 0
		new_symbols = []
		while i < len(symbols):
			if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair:
				new_symbols.append(symbols[i]+symbols[i+1])
				i += 2
			else:
				new_symbols.append(symbols[i])
				i += 1
		return new_symbols

	def tokenize(self, text):
		# Assumption: BPE codes must be learned from corpus before tokenizing
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

# --- Simplifying Assumptions ---
# - WhitespaceTokenizer: Treats special characters as tokens, splits on whitespace.
# - RegexTokenizer: Default pattern splits on words and punctuation; can be customized.
# - BPETokenizer: Learns merges from corpus, requires corpus to be preprocessed and cleaned.


# Example usage:
if __name__ =="__main__":
    partition_corpus(
        '/Users/akashmanna/ws/course_works/iNLP/Assignment_1/dataset/corpora/cc100_en.jsonl',
        '/Users/akashmanna/ws/course_works/iNLP/Assignment_1/dataset/corpora/partitions',
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
