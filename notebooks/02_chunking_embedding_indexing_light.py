"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
LIGHT VERSION - Uses TF-IDF instead of PyTorch embeddings
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Memory optimization
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    'sample_size': 15000,
    'chunk_size': 500,
    'chunk_overlap': 50,
    'embedding_dim': 50,  # Reduced to avoid dimension errors with small datasets
    'random_seed': 42,
    'input_file': 'data/filtered_complaints.csv',
    'output_dir': 'data/processed/',
    'vector_store_dir': 'vector_store/',
    'min_text_length': 50,
    'tfidf_max_features': 500,  # Limit TF-IDF features
}

# ============================================
# CLASSES
# ============================================

class StratifiedSampler:
    def __init__(self, sample_size=15000, random_seed=42):
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.sampling_strategy = {}
        
    def sample(self, df, product_column='Product'):
        print("\n" + "=" * 70)
        print("STRATIFIED SAMPLING")
        print("=" * 70)
        
        # Check if we have enough data
        if len(df) < self.sample_size:
            print(f"⚠️ Only {len(df)} records available. Using all records.")
            self.sample_size = len(df)
        
        product_counts = df[product_column].value_counts()
        total_records = len(df)
        
        print(f"\n📊 Product Distribution in Full Dataset:")
        for product, count in product_counts.items():
            pct = (count / total_records) * 100
            print(f"   - {product[:40]}: {count:,} ({pct:.1f}%)")
        
        samples = {}
        sampling_details = {}
        
        for product, count in product_counts.items():
            proportion = count / total_records
            sample_count = int(self.sample_size * proportion)
            if sample_count == 0 and count > 0:
                sample_count = 1
            samples[product] = sample_count
            sampling_details[product] = {
                'total_count': count,
                'proportion': proportion,
                'sample_count': sample_count,
            }
        
        # Adjust to match exact sample_size
        current_total = sum(samples.values())
        if current_total < self.sample_size and len(samples) > 0:
            remaining = self.sample_size - current_total
            largest_product = max(samples, key=lambda x: samples[x])
            samples[largest_product] += remaining
        
        print(f"\n🎯 Sampling Strategy:")
        print(f"   - Target sample size: {self.sample_size:,}")
        print(f"   - Actual sample size: {sum(samples.values()):,}")
        
        sampled_dfs = []
        np.random.seed(self.random_seed)
        
        for product, count in samples.items():
            if count > 0:
                product_df = df[df[product_column] == product]
                if len(product_df) >= count:
                    sampled = product_df.sample(n=count, random_state=self.random_seed)
                else:
                    sampled = product_df
                    print(f"   ⚠️ {product}: Only {len(product_df)} available, taking all")
                sampled_dfs.append(sampled)
        
        if not sampled_dfs:
            print("❌ No data to sample!")
            return df, {}
        
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        sampled_df = sampled_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"\n✅ Final sample: {len(sampled_df):,} records")
        
        self.sampling_strategy = {
            'total_records': total_records,
            'target_sample_size': self.sample_size,
            'actual_sample_size': len(sampled_df),
            'random_seed': self.random_seed
        }
        
        return sampled_df, self.sampling_strategy

class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text, complaint_id, product_category=None, issue=None):
        text = str(text).strip()
        
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'complaint_id': complaint_id,
                'product_category': product_category,
                'issue': issue,
                'chunk_index': 0,
                'total_chunks': 1,
                'chunk_length': len(text)
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append({
                'text': chunk_text,
                'complaint_id': complaint_id,
                'product_category': product_category,
                'issue': issue,
                'chunk_index': chunk_index,
                'total_chunks': 0,
                'chunk_length': len(chunk_text)
            })
            start += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1
        
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
        
        return chunks
    
    def process_complaints(self, df, text_column='cleaned_narrative'):
        print("\n" + "=" * 70)
        print("TEXT CHUNKING")
        print("=" * 70)
        
        print(f"\n🔧 Chunking Configuration:")
        print(f"   - Chunk size: {self.chunk_size} characters")
        print(f"   - Chunk overlap: {self.chunk_overlap} characters")
        
        all_chunks = []
        stats = {
            'total_complaints': len(df),
            'complaints_processed': 0,
            'total_chunks': 0,
            'chunk_lengths': []
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking complaints"):
            text = row.get(text_column, '')
            complaint_id = row.get('Complaint ID', f"comp_{idx}")
            product = row.get('Product', 'Unknown')
            issue = row.get('Issue', 'Unknown')
            
            if pd.isna(text) or len(str(text).strip()) < CONFIG['min_text_length']:
                continue
            
            chunks = self.chunk_text(text, complaint_id, product, issue)
            all_chunks.extend(chunks)
            stats['complaints_processed'] += 1
            for chunk in chunks:
                stats['chunk_lengths'].append(chunk['chunk_length'])
        
        stats['total_chunks'] = len(all_chunks)
        stats['avg_chunks_per_complaint'] = stats['total_chunks'] / stats['complaints_processed'] if stats['complaints_processed'] > 0 else 0
        
        print(f"\n📊 Chunking Statistics:")
        print(f"   - Complaints processed: {stats['complaints_processed']:,}")
        print(f"   - Total chunks created: {stats['total_chunks']:,}")
        print(f"   - Avg chunks per complaint: {stats['avg_chunks_per_complaint']:.2f}")
        
        return all_chunks, stats

class LightweightEmbedder:
    """Uses TF-IDF + SVD instead of neural embeddings"""
    
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.vectorizer = None
        self.svd = None
        self.is_fitted = False
        
    def fit_transform(self, texts):
        """Fit and transform texts to embeddings"""
        print("\n" + "=" * 70)
        print("GENERATING TF-IDF EMBEDDINGS")
        print("=" * 70)
        
        print(f"\n🔮 Generating embeddings for {len(texts):,} texts...")
        print(f"   - Target embedding dimension: {self.embedding_dim}")
        
        # TF-IDF Vectorization with adaptive features
        n_features = min(CONFIG['tfidf_max_features'], len(texts) * 10, 1000)
        print(f"   - Max features: {n_features}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=n_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1  # Include all terms
        )
        
        print("   - Computing TF-IDF...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        actual_features = tfidf_matrix.shape[1]
        print(f"   - TF-IDF shape: {tfidf_matrix.shape}")
        print(f"   - Actual features: {actual_features}")
        
        # Adaptive dimension reduction
        actual_dim = min(self.embedding_dim, actual_features, len(texts) - 1)
        
        if actual_dim < 1:
            print(f"⚠️ Not enough features for SVD. Using original TF-IDF features.")
            embeddings = tfidf_matrix.toarray()
            self.embedding_dim = actual_features
            self.is_fitted = True
            return embeddings
        
        print(f"   - Reducing to {actual_dim} dimensions with SVD...")
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        embeddings = self.svd.fit_transform(tfidf_matrix)
        
        # Update embedding dimension
        self.embedding_dim = actual_dim
        
        print(f"✅ Embeddings generated!")
        print(f"   - Final shape: {embeddings.shape}")
        if hasattr(self.svd, 'explained_variance_ratio_'):
            print(f"   - Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        
        self.is_fitted = True
        return embeddings
    
    def transform(self, text):
        """Transform a single text to embedding"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_transform first.")
        
        tfidf = self.vectorizer.transform([text])
        if self.svd:
            embedding = self.svd.transform(tfidf)
        else:
            embedding = tfidf.toarray()
        return embedding[0]

class VectorStoreBuilder:
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        self.id_to_chunk = {}
        
    def build_faiss_index(self, chunks, embeddings):
        print("\n" + "=" * 70)
        print("BUILDING FAISS VECTOR STORE")
        print("=" * 70)
        
        try:
            import faiss
        except ImportError:
            print("❌ FAISS not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
            import faiss
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings_norm.astype(np.float32)
        
        # Update embedding dimension
        self.embedding_dim = embeddings_norm.shape[1]
        
        print(f"\n📊 Building index with {len(embeddings_norm):,} vectors")
        print(f"   - Dimension: {self.embedding_dim}")
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_norm)
        
        print(f"✅ FAISS index built! Total vectors: {self.index.ntotal}")
        
        self.metadata = []
        self.id_to_chunk = {}
        
        for i, chunk in enumerate(chunks):
            metadata_entry = {
                'chunk_id': i,
                'complaint_id': chunk['complaint_id'],
                'product_category': chunk.get('product_category', 'Unknown'),
                'issue': chunk.get('issue', 'Unknown'),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
                'chunk_length': chunk.get('chunk_length', 0),
                'text': chunk['text'][:200] + '...'
            }
            self.metadata.append(metadata_entry)
            self.id_to_chunk[i] = chunk
        
        return self.index, self.metadata
    
    def save_index(self, filepath_prefix):
        print("\n" + "=" * 70)
        print("SAVING VECTOR STORE")
        print("=" * 70)
        
        try:
            import faiss
            
            index_file = f"{filepath_prefix}.faiss"
            faiss.write_index(self.index, index_file)
            print(f"✅ FAISS index saved to: {index_file}")
            
            metadata_file = f"{filepath_prefix}_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_chunk': self.id_to_chunk,
                    'config': {
                        'embedding_dim': self.embedding_dim,
                        'total_vectors': self.index.ntotal,
                        'timestamp': datetime.now().isoformat()
                    }
                }, f)
            print(f"✅ Metadata saved to: {metadata_file}")
            
            # Save chunks
            chunks_file = f"{filepath_prefix}_chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.id_to_chunk, f)
            print(f"✅ Chunks saved to: {chunks_file}")
            
            return True
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, filepath_prefix):
        try:
            import faiss
            
            index_file = f"{filepath_prefix}.faiss"
            self.index = faiss.read_index(index_file)
            
            metadata_file = f"{filepath_prefix}_metadata.pkl"
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_chunk = data['id_to_chunk']
                config = data['config']
            
            self.embedding_dim = config['embedding_dim']
            print(f"✅ Vector store loaded! Total vectors: {self.index.ntotal}")
            return True
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def search(self, query_embedding, k=5, product_filter=None):
        if self.index is None:
            print("❌ Index not loaded.")
            return []
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        distances, indices = self.index.search(query_embedding, k * 2)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                if product_filter and product_filter.lower() not in metadata['product_category'].lower():
                    continue
                results.append({
                    'chunk_id': idx,
                    'metadata': metadata,
                    'score': float(distances[0][i]),
                    'text': self.id_to_chunk.get(idx, {}).get('text', '')
                })
                if len(results) >= k:
                    break
        
        return results

# ============================================
# REPORT GENERATION
# ============================================

def generate_report(sampling_strategy, chunk_stats, chunks, metadata, vector_store):
    """Generate comprehensive report"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TASK 2 REPORT: LIGHTWEIGHT CHUNKING & EMBEDDING")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Section 1: Sampling Strategy
    report_lines.append("1. SAMPLING STRATEGY")
    report_lines.append("-" * 50)
    report_lines.append("")
    if sampling_strategy:
        report_lines.append(f"Total records available: {sampling_strategy.get('total_records', 'N/A'):,}")
        report_lines.append(f"Target sample size: {sampling_strategy.get('target_sample_size', 'N/A'):,}")
        report_lines.append(f"Actual sample size: {sampling_strategy.get('actual_sample_size', 'N/A'):,}")
        report_lines.append(f"Random seed: {sampling_strategy.get('random_seed', 'N/A')}")
    report_lines.append("")
    report_lines.append("Sampling Justification:")
    report_lines.append("Stratified sampling was used to ensure proportional representation")
    report_lines.append("across all product categories. This maintains the original")
    report_lines.append("distribution while reducing the dataset for processing.")
    report_lines.append("")
    
    # Section 2: Text Chunking
    report_lines.append("2. TEXT CHUNKING")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Chunk Size: {CONFIG['chunk_size']} characters")
    report_lines.append(f"Chunk Overlap: {CONFIG['chunk_overlap']} characters")
    report_lines.append("")
    report_lines.append("Chunking Statistics:")
    report_lines.append(f"  - Complaints processed: {chunk_stats.get('complaints_processed', 0):,}")
    report_lines.append(f"  - Total chunks created: {chunk_stats.get('total_chunks', 0):,}")
    report_lines.append(f"  - Average chunks per complaint: {chunk_stats.get('avg_chunks_per_complaint', 0):.2f}")
    report_lines.append("")
    report_lines.append("Chunking Justification:")
    report_lines.append("The chosen chunk size of 500 characters with 50-character overlap")
    report_lines.append("provides a balance between context preservation and retrieval precision.")
    report_lines.append("The overlap ensures continuity between chunks.")
    report_lines.append("")
    
    # Section 3: Embedding Model
    report_lines.append("3. EMBEDDING MODEL (LIGHTWEIGHT)")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Model: TF-IDF + Truncated SVD")
    report_lines.append(f"Embedding Dimension: {vector_store.embedding_dim if vector_store else CONFIG['embedding_dim']}")
    report_lines.append("")
    report_lines.append("Model Justification:")
    report_lines.append("TF-IDF with SVD was chosen as a lightweight alternative to neural embeddings:")
    report_lines.append("  1. No GPU or PyTorch dependencies required")
    report_lines.append("  2. Fast and efficient for small to medium datasets")
    report_lines.append("  3. Works well with complaint text data")
    report_lines.append("  4. Easy to deploy without compatibility issues")
    report_lines.append("")
    
    # Section 4: Vector Store
    report_lines.append("4. VECTOR STORE")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Vector Store Type: FAISS (IndexFlatIP for cosine similarity)")
    report_lines.append(f"Total Vectors Indexed: {len(metadata) if metadata else 0:,}")
    report_lines.append(f"Embedding Dimension: {vector_store.embedding_dim if vector_store else 'N/A'}")
    report_lines.append(f"Distance Metric: Cosine similarity")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("\n" + "█" * 70)
    print("█" + " " * 15 + "CREDITRUST FINANCIAL" + " " * 15 + "█")
    print("█" + " " * 12 + "TASK 2: LIGHTWEIGHT EMBEDDING" + " " * 12 + "█")
    print("█" * 70)
    
    try:
        # Load data
        input_file = CONFIG['input_file']
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            print("   Please run Task 1 first.")
            return
        
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} records")
        
        # Check if we have enough data
        if len(df) < CONFIG['sample_size']:
            print(f"⚠️ Only {len(df)} records available. Adjusting sample size.")
            CONFIG['sample_size'] = len(df)
        
        # Sample
        sampler = StratifiedSampler(
            sample_size=CONFIG['sample_size'],
            random_seed=CONFIG['random_seed']
        )
        sampled_df, sampling_strategy = sampler.sample(df)
        
        sampled_path = os.path.join(CONFIG['output_dir'], 'sampled_complaints.csv')
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        sampled_df.to_csv(sampled_path, index=False)
        print(f"✅ Sampled data saved to: {sampled_path}")
        
        # Chunk
        chunker = TextChunker(
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap']
        )
        chunks, chunk_stats = chunker.process_complaints(sampled_df)
        
        if not chunks:
            print("❌ No chunks created. Check your data.")
            return
        
        chunks_path = os.path.join(CONFIG['output_dir'], 'chunks.json')
        with open(chunks_path, 'w') as f:
            chunks_serializable = []
            for chunk in chunks:
                chunk_copy = chunk.copy()
                chunks_serializable.append(chunk_copy)
            json.dump(chunks_serializable, f, indent=2, default=str)
        print(f"✅ Chunks saved to: {chunks_path}")
        
        # Generate embeddings using TF-IDF
        embedder = LightweightEmbedder(embedding_dim=CONFIG['embedding_dim'])
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.fit_transform(texts)
        
        # Build vector store
        vector_store = VectorStoreBuilder(embedding_dim=embeddings.shape[1])
        index, metadata = vector_store.build_faiss_index(chunks, embeddings)
        
        os.makedirs(CONFIG['vector_store_dir'], exist_ok=True)
        vector_store_path = os.path.join(CONFIG['vector_store_dir'], 'complaint_index_light')
        vector_store.save_index(vector_store_path)
        
        # Generate and save report
        report = generate_report(sampling_strategy, chunk_stats, chunks, metadata, vector_store)
        report_path = os.path.join(CONFIG['output_dir'], 'task2_report_light.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {report_path}")
        
        # Test search
        print("\n" + "=" * 70)
        print("TESTING VECTOR SEARCH")
        print("=" * 70)
        
        test_query = "What are common credit card issues?"
        query_embedding = embedder.transform(test_query)
        results = vector_store.search(query_embedding, k=3)
        
        print(f"\n🔍 Test Query: '{test_query}'")
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n   {i}. Product: {result['metadata']['product_category']}")
                print(f"      Issue: {result['metadata']['issue']}")
                print(f"      Score: {result['score']:.4f}")
                print(f"      Text: {result['text'][:100]}...")
        else:
            print("   No results found.")
        
        print("\n" + "=" * 70)
        print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📋 Deliverables:")
        print("   1. ✅ Sampled data: data/processed/sampled_complaints.csv")
        print("   2. ✅ Chunks: data/processed/chunks.json")
        print("   3. ✅ Vector Store: vector_store/complaint_index_light.faiss")
        print("   4. ✅ Report: data/processed/task2_report_light.txt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()