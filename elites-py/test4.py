import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import math
from tqdm import tqdm
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
import re
import random

@dataclass
class Document:
    url: str
    title: str
    text: str
    categories: List[str]
    inbound_links: int  # For PageRank
    outbound_links: List[str]  # For PageRank
    term_frequencies: Counter  # For BM25

class BM25MapElites:
    def __init__(self, documents: List[Document], k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        print("Precomputing scores...")
        self.precompute_scores()
        print("Computing PageRank...")
        self.page_ranks = self.compute_pagerank()
        
        # Normalize PageRank scores to be more meaningful
        self.page_ranks = (self.page_ranks - self.page_ranks.min()) / (self.page_ranks.max() - self.page_ranks.min())
    
    def precompute_scores(self):
        # Calculate document lengths
        self.doc_lengths = np.array([len(doc.term_frequencies) for doc in self.documents])
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # Calculate IDF for all terms
        self.N = len(self.documents)
        self.idf = {}
        all_terms = set().union(*[doc.term_frequencies.keys() for doc in self.documents])
        
        for term in all_terms:
            df = sum(1 for doc in self.documents if term in doc.term_frequencies)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
    
    def compute_pagerank(self, damping=0.85, iterations=50):
        N = len(self.documents)
        ranks = np.ones(N) / N
        
        # Create adjacency matrix
        adjacency = np.zeros((N, N))
        for i, doc in enumerate(self.documents):
            for link in doc.outbound_links:
                for j, target in enumerate(self.documents):
                    if target.url == link:
                        adjacency[j, i] = 1
        
        # Normalize adjacency matrix
        out_degrees = np.sum(adjacency, axis=0)
        adjacency = adjacency / np.maximum(out_degrees, 1)
        
        # Power iteration
        for _ in range(iterations):
            new_ranks = (1 - damping) / N + damping * adjacency @ ranks
            if np.allclose(ranks, new_ranks):
                break
            ranks = new_ranks
        
        return ranks
    
    def get_behavior_characteristics(self, result_set: List[int]) -> Tuple[float, float]:
        """
        BCs:
        1. Authority spread (variance in PageRank scores)
        2. Topic diversity (unique categories)
        """
        authority_scores = self.page_ranks[result_set]
        authority_spread = np.std(authority_scores)
        
        unique_categories = set()
        for idx in result_set:
            unique_categories.update(self.documents[idx].categories)
        topic_diversity = len(unique_categories) / 20  # Normalize
        
        return authority_spread, topic_diversity
    
    def evaluate_solution(self, query_terms: List[str], result_indices: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        # Check for duplicates
        if len(set(result_indices)) != len(result_indices):
            return 0.0, (0.0, 0.0)
        
        scores = []
        for idx in result_indices:
            doc = self.documents[idx]
            
            # BM25 scoring with better term matching
            bm25_score = 0
            for term in query_terms:
                if term in doc.term_frequencies:
                    tf = doc.term_frequencies[term]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[idx] / self.avg_doc_length)
                    bm25_score += self.idf.get(term, 0) * (numerator / denominator)
                
                # Bonus for title matches
                if term in doc.title.lower():
                    bm25_score *= 1.5
            
            # Combine with PageRank more effectively
            final_score = bm25_score * (1 + np.log1p(self.page_ranks[idx]))
            scores.append(final_score)
        
        # Penalize low scores in top positions
        sorted_scores = sorted(scores, reverse=True)
        if sorted_scores[0] < 0.1:  # Minimum threshold for best result
            return 0.0, (0.0, 0.0)
        
        # Weight earlier results more heavily
        position_weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        weighted_score = np.average(scores, weights=position_weights)
        
        # Get behavior characteristics
        bcs = self.get_behavior_characteristics(result_indices)
        
        return weighted_score, bcs

def preprocess_text(text: str) -> List[str]:
    # Handle NaN or None values
    if pd.isna(text) or text is None:
        return []
    
    # Convert to string if not already
    text = str(text)
    
    # Basic preprocessing
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

class MapElites:
    def __init__(self, dimensions, bins):
        self.dimensions = dimensions
        self.bins = bins
        self.solutions = {}
        self.bins_per_dimension = bins
        
    def random_solution(self, n_pages):
        # Generate random indices for top 10 results
        return np.random.randint(0, n_pages, size=10)
        
    def mutate(self, solution, n_pages):
        # Mutate by replacing 1-3 results randomly
        new_solution = solution.copy()
        n_mutations = np.random.randint(1, 4)
        indices = np.random.choice(len(solution), n_mutations, replace=False)
        new_solution[indices] = np.random.randint(0, n_pages, size=n_mutations)
        return new_solution

    def features_to_bins(self, features):
        return tuple(min(int(f * self.bins_per_dimension), self.bins_per_dimension - 1) 
                    for f in features)

    def run(self, evaluate_solution, n_pages, iterations=1000):
        # Initialize with random solutions
        for _ in range(100):
            solution = self.random_solution(n_pages)
            fitness, features = evaluate_solution(solution)
            bins = self.features_to_bins(features)
            
            if bins not in self.solutions or fitness > self.solutions[bins][0]:
                self.solutions[bins] = (fitness, solution, features)

        # Main loop
        for _ in range(iterations):
            if not self.solutions:
                continue
                
            # Select random solution
            _, parent, _ = random.choice(list(self.solutions.values()))
            
            # Create and evaluate offspring
            offspring = self.mutate(parent, n_pages)
            fitness, features = evaluate_solution(offspring)
            bins = self.features_to_bins(features)
            
            if bins not in self.solutions or fitness > self.solutions[bins][0]:
                self.solutions[bins] = (fitness, offspring, features)

def main():
    # Load Wikipedia data and clean it
    df = pd.read_csv('wikipedia_articles.csv')
    df = df.fillna('')  # Replace NaN with empty string
    
    # Create documents with better link structure
    documents = []
    url_to_idx = {row['url']: i for i, row in df.iterrows()}
    
    for _, row in tqdm(df.iterrows(), desc="Processing documents"):
        # Better term frequency computation with error handling
        title = str(row['title']) if not pd.isna(row['title']) else ''
        text = str(row['text']) if not pd.isna(row['text']) else ''
        categories = str(row['categories']) if not pd.isna(row['categories']) else ''
        
        title_terms = preprocess_text(title)
        text_terms = preprocess_text(text)
        
        # Give more weight to title terms
        term_freq = Counter(title_terms * 3 + text_terms)
        
        # Create more realistic link structure
        n_links = np.random.pareto(2)  # Power law distribution
        potential_links = df['url'].values
        outbound = np.random.choice(
            potential_links,
            size=min(int(n_links), len(potential_links)),
            replace=False
        )
        
        doc = Document(
            url=row['url'],
            title=row['title'],
            text=row['text'],
            categories=row['categories'].split('|'),
            inbound_links=0,  # Will be computed from outbound links
            outbound_links=outbound.tolist(),
            term_frequencies=term_freq
        )
        documents.append(doc)
    
    # Compute inbound links
    for doc in documents:
        for target_url in doc.outbound_links:
            if target_url in url_to_idx:
                documents[url_to_idx[target_url]].inbound_links += 1
    
    # Initialize MAP-Elites with BM25
    search = BM25MapElites(documents)
    
    # Run search
    query = input("Enter search query: ")
    query_terms = preprocess_text(query)
    
    # MAP-Elites parameters
    n_iterations = 10000
    solution_length = 10  # Top 10 results
    
    # Initialize and run MAP-Elites
    map_elites = MapElites(dimensions=2, bins=10)
    
    def evaluate_wrapper(solution):
        return search.evaluate_solution(query_terms, solution)
    
    print("\nOptimizing search results...")
    map_elites.run(evaluate_wrapper, len(documents), iterations=n_iterations)
    
    # Get best solutions
    best_solutions = sorted(
        map_elites.solutions.values(),
        key=lambda x: x[0],
        reverse=True
    )[:5]
    
    # Print results
    print("\nTop Results:")
    print("=" * 50)
    
    for i, (fitness, solution, bcs) in enumerate(best_solutions, 1):
        print(f"\nSolution Set {i} (fitness: {fitness:.3f}):")
        print(f"Authority Spread: {bcs[0]:.2f}")
        print(f"Topic Diversity: {bcs[1]:.2f}")
        
        for j, idx in enumerate(solution[:5], 1):
            doc = documents[idx]
            print(f"\n{j}. {doc.title}")
            print(f"   URL: {doc.url}")
            print(f"   PageRank: {search.page_ranks[idx]:.3f}")
            print(f"   Categories: {', '.join(doc.categories)}")
            print(f"   Preview: {doc.text[:100]}...")

if __name__ == "__main__":
    main() 