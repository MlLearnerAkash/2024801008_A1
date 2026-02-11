# --- NGram Language Model Implementation ---
import collections
import argparse
from tokenizers import WhitespaceTokenizer, RegexTokenizer, BPETokenizer

import numpy as np

class NGramLanguageModel:
    def __init__(self, n, smoothing=None):
        self.n = n
        self.smoothing = smoothing  # None, 'witten-bell', 'kneser-ney'
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.vocab = set()
        self.UNK = '<UNK>'

    def train(self, tokenized_sentences):
        for tokens in tokenized_sentences:
            tokens = ['<SOS>'] * (self.n - 1) + tokens + ['<EOS>']
            for i in range(self.n - 1, len(tokens)):
                ngram = tuple(tokens[i - self.n + 1:i + 1])
                context = tuple(tokens[i - self.n + 1:i])
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.vocab.update(ngram)

    def next_token_prob(self, context, candidate):
        # context: tuple of n-1 tokens
        ngram = context + (candidate,)
        V = len(self.vocab)
        if self.smoothing == 'witten-bell':
            # Witten-Bell smoothing
            context_count = self.context_counts.get(context, 0)
            ngram_count = self.ngram_counts.get(ngram, 0)
            T = len([w for w in self.vocab if self.ngram_counts.get(context + (w,), 0) > 0])
            Z = V - T
            if context_count + T == 0:
                return 1.0 / V
            if ngram_count > 0:
                return ngram_count / (context_count + T)
            else:
                return T / (Z * (context_count + T))
        elif self.smoothing == 'kneser-ney':
            # Kneser-Ney smoothing (simplified, not full recursive)
            D = 0.75
            ngram_count = self.ngram_counts.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)
            cont_count = len([1 for ng in self.ngram_counts if ng[1:] == ngram[1:]])
            if context_count == 0:
                return 1.0 / V
            return max(ngram_count - D, 0) / context_count + D * cont_count / context_count / V
        else:
            # No smoothing (MLE)
            ngram_count = self.ngram_counts.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)
            if context_count == 0:
                return 1.0 / V
            return ngram_count / context_count

    def predict_next(self, context):
        # context: list of n-1 tokens
        context = tuple(context[-(self.n - 1):])
        candidates = list(self.vocab)
        probs = [(w, self.next_token_prob(context, w)) for w in candidates]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[0][0] if probs else self.UNK

    def complete_sentence(self, prompt_tokens, max_len=30):
        tokens = prompt_tokens[:]
        for _ in range(max_len):
            next_tok = self.predict_next(tokens)
            if next_tok == '<EOS>':
                break
            tokens.append(next_tok)
        return tokens

    def perplexity(self, tokenized_sentences):
        log_prob_sum = 0
        word_count = 0
        for tokens in tokenized_sentences:
            tokens = ['<SOS>'] * (self.n - 1) + tokens + ['<EOS>']
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                target = tokens[i]
                prob = self.next_token_prob(context, target)
                log_prob_sum += np.log(prob + 1e-12)
                word_count += 1
        return np.exp(-log_prob_sum / word_count) if word_count > 0 else float('inf')

def load_sentences(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_tokenizer(name, train_path):
    if name == "whitespace":
        return WhitespaceTokenizer()
    elif name == "regex":
        return RegexTokenizer(r"\w+(?:'\w+)?|[^\w\s]")
    elif name == "bpe":
        tokenizer = BPETokenizer()
        with open(train_path, 'r', encoding='utf-8') as f_train:
                train_corpus = [line.strip() for line in f_train if line.strip()]
        tokenizer.fit(train_corpus)
        return tokenizer
    else:
        raise ValueError("Unknown tokenizer")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, choices=['whitespace', 'regex', 'bpe'], required=True)
    parser.add_argument('--smoothing', type=str, choices=['none', 'witten-bell', 'kneser-ney'], required=True)
    parser.add_argument('--prompt', type=str, default="The quick brown")
    args = parser.parse_args()

    # Load and tokenize data
    train_sents = load_sentences(args.train)
    test_sents = load_sentences(args.test)
    tokenizer = get_tokenizer(args.tokenizer, args.train)
    train_tokens = [tokenizer.tokenize(sent) for sent in train_sents]
    test_tokens = [tokenizer.tokenize(sent) for sent in test_sents]

    # Train LM
    lm = NGramLanguageModel(n=4, smoothing=None if args.smoothing == 'none' else args.smoothing)
    lm.train(train_tokens)

    # Perplexity
    ppl = lm.perplexity(test_tokens)
    print(f"Perplexity ({args.tokenizer}, {args.smoothing}): {ppl:.3f}")

    # # Sentence completion
    # prompt_tokens = tokenizer.tokenize(args.prompt)
    # completed = lm.complete_sentence(prompt_tokens)
    # print("Prompt:", args.prompt)
    # print("Completed:", " ".join(completed))

if __name__ == '__main__':
    main()