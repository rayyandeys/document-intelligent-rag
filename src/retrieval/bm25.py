"""Transparent BM25 baseline; regex tokens, no stemming or stopword removal."""
import math
import re
from collections import Counter, defaultdict


class BM25:
    def __init__(self, documents, k1=1.2, b=.75):
        if not documents:
            raise ValueError('Empty corpus')
        self.ids = sorted(documents)
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.lengths = []
        for i, doc in enumerate(self.ids):
            counts = Counter(self.tokenize(documents[doc]))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i,tf))
        self.avg = sum(self.lengths)/len(self.ids) or 1

    @staticmethod
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    def retrieve(self, query, k=10):
        scores = [0.0]*len(self.ids)
        for term in set(self.tokenize(query)):
            postings = self.postings.get(term, [])
            idf = math.log(1+(len(self.ids)-len(postings)+.5)/(len(postings)+.5))
            for i, tf in postings:
                scores[i] += idf*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths[i]/self.avg))
        order = sorted(range(len(scores)), key=lambda i:(-scores[i], self.ids[i]))[:k]
        return [(self.ids[i], scores[i]) for i in order]
