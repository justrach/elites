import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time
import random
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import os
import json
import hashlib
from pathlib import Path
from sentence_transformers import util
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet', quiet=True)
import logging
from datetime import datetime

@dataclass
class WebPage:
    url: str
    title: str
    description: str
    clicks: int
    categories: List[str]
    embedding: np.ndarray

def load_wikipedia_data(n_pages: int = 1000) -> List[WebPage]:
    # Load Wikipedia articles dataset
    df = pd.read_csv('wikipedia_articles.csv')
    df = df.dropna()  # Remove rows with NaN values
    df = df.head(n_pages)
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    pages = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading pages"):
        # Get embeddings for title + description
        text = f"{row['title']} {str(row['text'])[:200]}"
        embedding = model.encode(text, convert_to_tensor=False)
        
        # Extract categories safely
        try:
            categories = [cat.strip() for cat in str(row['categories']).split('|') 
                         if cat.strip()]
        except:
            categories = ['uncategorized']
        
        # Generate realistic random click data
        # Use Pareto distribution for power-law behavior (few very popular pages)
        clicks = int(np.random.pareto(2) * 1000)  # alpha=2 gives reasonable distribution
        
        pages.append(WebPage(
            url=str(row['url']),
            title=str(row['title']),
            description=str(row['text'])[:200],
            clicks=clicks,
            categories=categories[:5],
            embedding=embedding
        ))
    
    return pages

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

class SearchCache:
    def __init__(self, cache_dir="search_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, query: str) -> str:
        # Create a deterministic but readable cache key
        query_clean = "".join(c.lower() for c in query if c.isalnum())
        return f"{query_clean[:30]}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
    
    def save_grid(self, query: str, solutions: dict, metadata: dict):
        key = self.get_cache_key(query)
        cache_file = self.cache_dir / f"{key}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_solutions = {}
        for bin_key, (fitness, solution, features) in solutions.items():
            serializable_solutions[str(bin_key)] = {
                'fitness': float(fitness),
                'solution': solution.tolist(),
                'features': [float(f) for f in features]
            }
        
        data = {
            'query': query,
            'solutions': serializable_solutions,
            'metadata': metadata
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def load_grid(self, query: str) -> tuple[dict, dict]:
        key = self.get_cache_key(query)
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None, None
        
        with open(cache_file) as f:
            data = json.load(f)
            
        # Convert back to proper format
        solutions = {}
        for bin_key_str, sol_data in data['solutions'].items():
            bin_key = tuple(map(int, bin_key_str.strip('()').split(',')))
            solutions[bin_key] = (
                sol_data['fitness'],
                np.array(sol_data['solution']),
                np.array(sol_data['features'])
            )
            
        return solutions, data['metadata']

class SearchLogger:
    def __init__(self, query: str):
        # Set up logging
        self.logger = logging.getLogger('search_optimizer')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path('search_logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp and query
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        query_clean = "".join(c for c in query if c.isalnum() or c.isspace())[:30]
        log_file = log_dir / f"search_{query_clean}_{timestamp}.log"
        
        # Add file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log(self, msg: str, level: str = 'info'):
        getattr(self.logger, level)(msg)

class SearchOptimizer:
    def __init__(self, pages: List[WebPage], query: str, model: SentenceTransformer):
        self.logger = SearchLogger(query)
        self.pages = pages
        self.original_query = query
        self.model = model
        
        self.logger.log(f"Initializing search for query: {query}")
        
        # Expand query
        self.expanded_query = self._expand_query(query)
        self.logger.log(f"Expanded query: {self.expanded_query}")
        
        # Pre-compute embeddings
        self.logger.log("Computing embeddings...")
        self.query_embedding = self.model.encode(self.expanded_query)
        self.title_embeddings = self.model.encode([p.title.lower() for p in pages])
        self.content_embeddings = self.model.encode([p.title + " " + p.description for p in pages])
        
        self.logger.log("Computing similarity matrices...")
        self.content_similarity = util.cos_sim(self.content_embeddings, self.query_embedding).numpy().flatten()
        self.title_similarity = util.cos_sim(self.title_embeddings, self.query_embedding).numpy().flatten()
        
        # Log top direct matches
        top_matches = sorted(enumerate(self.content_similarity), key=lambda x: x[1], reverse=True)[:5]
        self.logger.log("\nTop direct matches:")
        for idx, score in top_matches:
            self.logger.log(f"  {pages[idx].title} (score: {score:.3f})")
        
        self.logger.log("Computing authority scores...")
        self.authority_scores = self._compute_authority_scores()
        
        # Log pages with highest authority
        top_authority = sorted(enumerate(self.authority_scores), key=lambda x: x[1], reverse=True)[:5]
        self.logger.log("\nTop authority pages:")
        for idx, score in top_authority:
            self.logger.log(f"  {pages[idx].title} (score: {score:.3f})")
        
        self.query_terms = set(self.original_query.lower().split())
        
        self.best_fitness_seen = -float('inf')
        self.generations_without_improvement = 0
        
        # Define behavior characteristics
        self.behavior_ranges = {
            'content_type': ['technical', 'overview', 'application', 'news', 'academic'],
            'specificity': ['broad', 'medium', 'specific'],
            'temporal': ['recent', 'established', 'historical']
        }

    def _expand_query(self, query: str) -> str:
        expanded_terms = set([query])
        
        # Add synonyms from WordNet
        for word in query.split():
            for syn in wordnet.synsets(word):
                expanded_terms.update(syn.lemma_names())
        
        # Add common AI/tech terms if relevant
        tech_terms = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
            'ml': ['machine learning', 'algorithm', 'model'],
            'neural': ['neural network', 'deep learning', 'ai'],
            # Add more mappings
        }
        
        for term, expansions in tech_terms.items():
            if term in query.lower():
                expanded_terms.update(expansions)
        
        return " ".join(expanded_terms)
    
    def _compute_authority_scores(self) -> np.ndarray:
        # PageRank-like scoring based on clicks and category relationships
        scores = np.zeros(len(self.pages))
        
        for i, page in enumerate(self.pages):
            # Base score from clicks (log-scaled)
            scores[i] = np.log1p(page.clicks)
            
            # Boost pages with more diverse categories
            scores[i] *= (1 + len(page.categories) / 10)
            
            # Boost pages with technical/academic categories
            academic_terms = {'research', 'science', 'technology', 'theory', 'algorithm'}
            if any(term in cat.lower() for cat in page.categories for term in academic_terms):
                scores[i] *= 1.2
        
        return scores / scores.max()  # Normalize

    def get_behavior_characteristics(self, solution: np.ndarray) -> Tuple[float, float]:
        pages = [self.pages[int(i)] for i in solution]
        
        # BC1: Technical Depth (0-1)
        technical_terms = {
            'algorithm', 'implementation', 'theory', 'framework', 'architecture',
            'system', 'method', 'technique', 'analysis', 'research'
        }
        technical_scores = []
        for page in pages:
            text = f"{page.title} {page.description}".lower()
            score = sum(term in text for term in technical_terms) / len(technical_terms)
            technical_scores.append(score)
        technical_depth = np.mean(technical_scores)
        
        # BC2: Topic Diversity (0-1)
        all_categories = set().union(*[set(p.categories) for p in pages])
        category_overlap = len(all_categories) / (sum(len(p.categories) for p in pages))
        
        return (technical_depth, category_overlap)

    def evaluate_solution(self, solution: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        relevance_scores = self.content_similarity[solution]
        title_scores = self.title_similarity[solution]
        
        # Basic quality threshold
        min_relevance = min(relevance_scores[:3])  # First 3 results
        if min_relevance < 0.2:  # Quality threshold
            return 0.0, (0.0, 0.0)  # Reject poor solutions
            
        # Get behavior characteristics
        bcs = self.get_behavior_characteristics(solution)
        
        # Simple fitness - just needs to be "good enough"
        fitness = float(min_relevance > 0.2)
        
        self.logger.log(f"Solution: fitness={fitness:.3f}, bcs={bcs}")
        return fitness, bcs

def main():
    pages = load_wikipedia_data()
    cache = SearchCache()
    
    query = input("Enter your search query: ")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize search optimizer
    optimizer = SearchOptimizer(pages, query, model)
    
    # Try to load from cache first
    cached_solutions, metadata = cache.load_grid(query)
    if cached_solutions:
        print(f"\nFound cached results from previous search")
        if metadata.get('expanded_query'):
            print(f"Expanded query: {metadata['expanded_query']}")
        map_elites = MapElites(dimensions=2, bins=20)
        map_elites.solutions = cached_solutions
    else:
        map_elites = MapElites(dimensions=2, bins=30)  # Finer-grained behavior space
        print("\nExploring search result behaviors...")
        
        with tqdm(total=20000, desc="Exploring") as pbar:
            def progress_wrapper(solution):
                result = optimizer.evaluate_solution(solution)
                pbar.update(1)
                if pbar.n % 1000 == 0:
                    coverage = len(map_elites.solutions) / (map_elites.bins ** map_elites.dimensions)
                    optimizer.logger.log(f"Coverage: {coverage:.2%}")
                return result
            
            map_elites.run(progress_wrapper, len(pages), iterations=20000)
    
    # Print diverse results
    print("\nDiverse Result Sets:")
    print("=" * 50)
    
    # Get solutions from different regions of behavior space
    solutions = []
    bins = list(map_elites.solutions.keys())
    if bins:
        # Get solutions from different corners of behavior space
        corners = [
            min(bins, key=lambda x: sum(x)),  # Low-low
            max(bins, key=lambda x: x[0]),    # High-low
            max(bins, key=lambda x: x[1]),    # Low-high
            max(bins, key=lambda x: sum(x))   # High-high
        ]
        
        for bin_key in corners:
            if map_elites.solutions[bin_key][0] > 0:  # Only show valid solutions
                solutions.append(map_elites.solutions[bin_key])
    
    # Print results with behavior characteristics
    for i, (fitness, solution, bcs) in enumerate(solutions, 1):
        print(f"\nResult Set {i}:")
        print(f"Technical Depth: {bcs[0]:.2f}")
        print(f"Topic Diversity: {bcs[1]:.2f}")
        print("-" * 30)
        
        results = [pages[int(i)] for i in solution]
        for j, page in enumerate(results[:5], 1):  # Show top 5 from each set
            print(f"{j}. {page.title}")
            print(f"   Categories: {', '.join(page.categories)}")

if __name__ == "__main__":
    main() 