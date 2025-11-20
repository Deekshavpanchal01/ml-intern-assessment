from collections import Counter
import math
import random

class TrigramModel:
    """
    A simple trigram language model with add-one smoothing.
    """

    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.total_tokens = 0

    def _tokenize(self, text):
        if not text:
            return []
        return text.lower().split()

    def fit(self, text):
        tokens = self._tokenize(text)

        # Unigrams
        for t in tokens:
            self.unigrams[t] += 1
            self.vocab.add(t)
            self.total_tokens += 1

        # Bigrams
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            self.bigrams[pair] += 1

        # Trigrams
        for i in range(len(tokens) - 2):
            triple = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.trigrams[triple] += 1

    def vocab_size(self):
        return len(self.vocab)

    def prob(self, w1, w2, w3):
        V = self.vocab_size() or 1
        trigram_count = self.trigrams[(w1, w2, w3)]
        bigram_count = self.bigrams[(w1, w2)]

        # Trigram known
        if bigram_count > 0:
            return (trigram_count + 1) / (bigram_count + V)

        # Backoff bigram
        unigram_count = self.unigrams[w2]
        bigram_count2 = self.bigrams[(w2, w3)]
        if unigram_count > 0:
            return (bigram_count2 + 1) / (unigram_count + V)

        # Unigram backoff
        unigram_count3 = self.unigrams[w3]
        return (unigram_count3 + 1) / (self.total_tokens + V)

    def log_prob(self, w1, w2, w3):
        return math.log(self.prob(w1, w2, w3))

    def generate(self, max_length=50):
        """
        Generate a sentence using trigram probabilities.
        """
        if not self.vocab:
            return ""

        vocab_list = sorted(self.vocab)

        # If too small, return the vocab joined
        if len(vocab_list) < 2:
            return " ".join(vocab_list)

        w1, w2 = random.sample(vocab_list, 2)
        sentence = [w1, w2]

        for _ in range(max_length - 2):
            candidates = []
            for w3 in vocab_list:
                candidates.append((self.prob(w1, w2, w3), w3))

            candidates.sort(reverse=True)
            next_word = candidates[0][1]

            sentence.append(next_word)
            w1, w2 = w2, next_word

        return " ".join(sentence)
