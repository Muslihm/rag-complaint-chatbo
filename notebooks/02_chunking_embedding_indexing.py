"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
CrediTrust Financial - Complaint Intelligence System

This script handles:
1. Stratified sampling of complaints
2. Text chunking with configurable parameters
3. Embedding generation using sentence-transformers
4. Vector store creation using FAISS
5. Metadata storage for traceability
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import hashlib
from datetime import datetime
from collections import Counter
from tqdm import tqdm
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
    'sample_size': 15000,  # Total sample size
    'chunk_size': 500,     # Characters per chunk
    'chunk_overlap': 50,   # Overlap between chunks
    'embedding_model': 'all-MiniLM-L6-v2',
    'embedding_dim': 384,
    'random_seed': 42,
    'input_file': 'data/filtered_complaints.csv',
    'output_dir': 'data/processed/',
    'vector_store_dir': 'vector_store/',
    'min_text_length': 50,  # Minimum characters for a complaint
}

# ============================================
# PART 1: STRATIFIED SAMPLING
# ============================================

class StratifiedSampler:
    """
    Handles stratified sampling of complaints across product categories
    """
    
    def __init__(self, sample_size=15000, random_seed=42):
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.sampling_strategy = {}
        
    def sample(self, df, product_column='Product'):
        """
        Perform stratified sampling across product categories
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        product_column : str
            Column name for product categories
            
        Returns:
        --------
        pd.DataFrame
            Sampled dataset
        dict
            Sampling strategy documentation
        """
        print("\n" + "=" * 70)
        print("STRATIFIED SAMPLING")
        print("=" * 70)
        
        # Get product distribution
        product_counts = df[product_column].value_counts()
        total_records = len(df)
        
        print(f"\n📊 Product Distribution in Full Dataset:")
        for product, count in product_counts.items():
            pct = (count / total_records) * 100
            print(f"   - {product[:40]}: {count:,} ({pct:.1f}%)")
        
        # Determine sample size per product (proportional)
        samples = {}
        sampling_details = {}
        
        # First, calculate proportional samples
        for product, count in product_counts.items():
            proportion = count / total_records
            sample_count = int(self.sample_size * proportion)
            
            # Ensure at least 1 sample per product if possible
            if sample_count == 0 and count > 0:
                sample_count = 1
            
            samples[product] = sample_count
            sampling_details[product] = {
                'total_count': count,
                'proportion': proportion,
                'sample_count': sample_count,
                'sample_pct': (sample_count / self.sample_size) * 100
            }
        
        # Adjust to match exact sample_size
        current_total = sum(samples.values())
        if current_total < self.sample_size:
            # Add remaining to largest category
            remaining = self.sample_size - current_total
            largest_product = max(samples, key=lambda x: samples[x])
            samples[largest_product] += remaining
            sampling_details[largest_product]['sample_count'] = samples[largest_product]
            sampling_details[largest_product]['sample_pct'] = (samples[largest_product] / self.sample_size) * 100
        
        print(f"\n🎯 Sampling Strategy:")
        print(f"   - Target sample size: {self.sample_size:,}")
        print(f"   - Actual sample size: {sum(samples.values()):,}")
        print(f"\n📊 Samples per product:")
        for product, count in samples.items():
            total = sampling_details[product]['total_count']
            print(f"   - {product[:40]}: {count:>6,} of {total:>8,} ({count/total*100:.1f}%)")
        
        # Perform sampling
        sampled_dfs = []
        np.random.seed(self.random_seed)
        
        for product, count in samples.items():
            if count > 0:
                product_df = df[df[product_column] == product]
                if len(product_df) >= count:
                    sampled = product_df.sample(n=count, random_state=self.random_seed)
                else:
                    # If not enough samples, take all available
                    sampled = product_df
                    print(f"   ⚠️ {product}: Only {len(product_df)} available, taking all")
                sampled_dfs.append(sampled)
        
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the final sample
        sampled_df = sampled_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"\n✅ Final sample: {len(sampled_df):,} records")
        
        # Store sampling strategy for report
        self.sampling_strategy = {
            'total_records': total_records,
            'target_sample_size': self.sample_size,
            'actual_sample_size': len(sampled_df),
            'product_distribution': sampling_details,
            'random_seed': self.random_seed
        }
        
        return sampled_df, self.sampling_strategy

# ============================================
# PART 2: TEXT CHUNKING
# ============================================

class TextChunker:
    """
    Handles text chunking with configurable parameters
    Supports both custom implementation and LangChain compatibility
    """
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text, complaint_id, product_category=None, issue=None):
        """
        Split text into overlapping chunks
        
        Parameters:
        -----------
        text : str
            Text to chunk
        complaint_id : str or int
            Original complaint identifier
        product_category : str, optional
            Product category for metadata
        issue : str, optional
            Issue type for metadata
            
        Returns:
        --------
        list
            List of chunk dictionaries with metadata
        """
        text = str(text).strip()
        
        if len(text) <= self.chunk_size:
            # Single chunk
            return [{
                'text': text,
                'complaint_id': complaint_id,
                'product_category': product_category,
                'issue': issue,
                'chunk_index': 0,
                'total_chunks': 1,
                'chunk_length': len(text)
            }]
        
        # Multi-chunk with overlap
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
                'total_chunks': 0,  # Will update later
                'chunk_length': len(chunk_text)
            })
            
            start += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1
        
        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
            
        return chunks
    
    def process_complaints(self, df, text_column='cleaned_narrative'):
        """
        Process all complaints in a dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        text_column : str
            Column containing text to chunk
            
        Returns:
        --------
        list
            List of all chunks with metadata
        dict
            Chunking statistics
        """
        print("\n" + "=" * 70)
        print("TEXT CHUNKING")
        print("=" * 70)
        
        print(f"\n🔧 Chunking Configuration:")
        print(f"   - Chunk size: {self.chunk_size} characters")
        print(f"   - Chunk overlap: {self.chunk_overlap} characters")
        print(f"   - Text column: {text_column}")
        
        all_chunks = []
        stats = {
            'total_complaints': len(df),
            'complaints_processed': 0,
            'total_chunks': 0,
            'avg_chunks_per_complaint': 0,
            'chunk_lengths': []
        }
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking complaints"):
            text = row.get(text_column, '')
            complaint_id = row.get('Complaint ID', f"comp_{idx}")
            product = row.get('Product', 'Unknown')
            issue = row.get('Issue', 'Unknown')
            
            # Skip empty or very short text
            if pd.isna(text) or len(str(text).strip()) < CONFIG['min_text_length']:
                continue
            
            chunks = self.chunk_text(text, complaint_id, product, issue)
            all_chunks.extend(chunks)
            stats['complaints_processed'] += 1
            
            # Collect length statistics
            for chunk in chunks:
                stats['chunk_lengths'].append(chunk['chunk_length'])
        
        stats['total_chunks'] = len(all_chunks)
        stats['avg_chunks_per_complaint'] = stats['total_chunks'] / stats['complaints_processed'] if stats['complaints_processed'] > 0 else 0
        
        print(f"\n📊 Chunking Statistics:")
        print(f"   - Complaints processed: {stats['complaints_processed']:,}")
        print(f"   - Total chunks created: {stats['total_chunks']:,}")
        print(f"   - Average chunks per complaint: {stats['avg_chunks_per_complaint']:.2f}")
        print(f"   - Min chunk length: {min(stats['chunk_lengths'])}")
        print(f"   - Max chunk length: {max(stats['chunk_lengths'])}")
        print(f"   - Avg chunk length: {np.mean(stats['chunk_lengths']):.0f}")
        
        return all_chunks, stats

# ============================================
# PART 3: EMBEDDING GENERATION
# ============================================

class ComplaintEmbedder:
    """
    Handles embedding generation using sentence-transformers
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.model = None
        
    def load_model(self):
        """
        Load the embedding model
        """
        print("\n" + "=" * 70)
        print("LOADING EMBEDDING MODEL")
        print("=" * 70)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"\n🔧 Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully!")
            print(f"   - Model dimensions: {self.model.get_sentence_embedding_dimension()}")
            print(f"   - Max sequence length: {self.model.max_seq_length}")
            
            return self.model
            
        except ImportError:
            print("❌ sentence-transformers not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully!")
            return self.model
    
    def generate_embeddings(self, chunks, batch_size=32):
        """
        Generate embeddings for text chunks
        
        Parameters:
        -----------
        chunks : list
            List of chunk dictionaries
        batch_size : int
            Batch size for embedding generation
            
        Returns:
        --------
        list
            List of chunk dictionaries with embeddings added
        """
        print("\n" + "=" * 70)
        print("GENERATING EMBEDDINGS")
        print("=" * 70)
        
        if self.model is None:
            self.load_model()
        
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"\n🔮 Generating embeddings for {len(texts):,} chunks...")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Model: {self.model_name}")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize during indexing
        )
        
        print(f"\n✅ Embeddings generated!")
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Mean: {embeddings.mean():.4f}")
        print(f"   - Std: {embeddings.std():.4f}")
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks, embeddings

# ============================================
# PART 4: VECTOR STORE CREATION (FAISS)
# ============================================

class VectorStoreBuilder:
    """
    Builds and manages vector stores using FAISS
    """
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        self.id_to_chunk = {}
        
    def build_faiss_index(self, chunks, embeddings):
        """
        Build FAISS index from embeddings
        
        Parameters:
        -----------
        chunks : list
            List of chunk dictionaries with embeddings
        embeddings : np.ndarray
            Embedding matrix
        """
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
        
        print(f"\n📊 Building index with {len(embeddings_norm):,} vectors")
        print(f"   - Dimension: {self.embedding_dim}")
        print(f"   - Metric: Cosine similarity (inner product)")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        self.index.add(embeddings_norm)
        
        print(f"✅ FAISS index built!")
        print(f"   - Total vectors: {self.index.ntotal}")
        print(f"   - Index type: {type(self.index)}")
        
        # Store metadata
        self.metadata = []
        self.id_to_chunk = {}
        
        for i, chunk in enumerate(chunks):
            # Clean up metadata (remove embedding from chunk data)
            metadata_chunk = chunk.copy()
            metadata_chunk.pop('embedding', None)
            
            metadata_entry = {
                'chunk_id': i,
                'complaint_id': chunk['complaint_id'],
                'product_category': chunk.get('product_category', 'Unknown'),
                'issue': chunk.get('issue', 'Unknown'),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
                'chunk_length': chunk.get('chunk_length', 0),
                'text': chunk['text'][:200] + '...'  # Store snippet, not full text
            }
            self.metadata.append(metadata_entry)
            self.id_to_chunk[i] = chunk
        
        return self.index, self.metadata
    
    def save_index(self, filepath_prefix):
        """
        Save FAISS index and metadata to disk
        """
        print("\n" + "=" * 70)
        print("SAVING VECTOR STORE")
        print("=" * 70)
        
        try:
            import faiss
            
            # Save FAISS index
            index_file = f"{filepath_prefix}.faiss"
            faiss.write_index(self.index, index_file)
            print(f"✅ FAISS index saved to: {index_file}")
            
            # Save metadata
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
            
            # Save chunk data separately (for easy access)
            chunks_file = f"{filepath_prefix}_chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.id_to_chunk, f)
            print(f"✅ Chunk data saved to: {chunks_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, filepath_prefix):
        """
        Load FAISS index and metadata from disk
        """
        try:
            import faiss
            
            # Load FAISS index
            index_file = f"{filepath_prefix}.faiss"
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            metadata_file = f"{filepath_prefix}_metadata.pkl"
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_chunk = data['id_to_chunk']
                config = data['config']
            
            self.embedding_dim = config['embedding_dim']
            
            print(f"✅ Vector store loaded!")
            print(f"   - Total vectors: {self.index.ntotal}")
            print(f"   - Total chunks: {len(self.metadata)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def search(self, query_embedding, k=5, product_filter=None):
        """
        Search for similar chunks
        
        Parameters:
        -----------
        query_embedding : np.ndarray
            Query embedding
        k : int
            Number of results to return
        product_filter : str, optional
            Filter by product category
            
        Returns:
        --------
        list
            List of search results with metadata and scores
        """
        if self.index is None:
            print("❌ Index not loaded. Please load or build index first.")
            return []
        
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k * 2)  # Get extra for filtering
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # Apply product filter if specified
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
# PART 5: MAIN EXECUTION
# ============================================

def main():
    """
    Main execution function for Task 2
    """
    
    print("\n" + "█" * 70)
    print("█" + " " * 20 + "CREDITRUST FINANCIAL" + " " * 20 + "█")
    print("█" + " " * 16 + "TASK 2: CHUNKING & EMBEDDING" + " " * 18 + "█")
    print("█" * 70)
    
    start_time = datetime.now()
    
    try:
        # ============================================
        # STEP 1: Load Cleaned Data
        # ============================================
        print("\n" + "=" * 70)
        print("STEP 1: LOADING CLEANED DATA")
        print("=" * 70)
        
        input_file = CONFIG['input_file']
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            print("   Please run Task 1 first to generate filtered_complaints.csv")
            return None, None
        
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} records from {input_file}")
        print(f"   - Columns: {df.columns.tolist()}")
        
        # ============================================
        # STEP 2: Stratified Sampling
        # ============================================
        sampler = StratifiedSampler(
            sample_size=CONFIG['sample_size'],
            random_seed=CONFIG['random_seed']
        )
        
        sampled_df, sampling_strategy = sampler.sample(df)
        
        # Save sampled data
        sampled_path = os.path.join(CONFIG['output_dir'], 'sampled_complaints.csv')
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        sampled_df.to_csv(sampled_path, index=False)
        print(f"✅ Sampled data saved to: {sampled_path}")
        
        # ============================================
        # STEP 3: Text Chunking
        # ============================================
        chunker = TextChunker(
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap']
        )
        
        chunks, chunk_stats = chunker.process_complaints(sampled_df)
        
        # Save chunks
        chunks_path = os.path.join(CONFIG['output_dir'], 'chunks.json')
        with open(chunks_path, 'w') as f:
            # Remove embeddings for JSON serialization
            chunks_serializable = []
            for chunk in chunks:
                chunk_copy = chunk.copy()
                chunk_copy.pop('embedding', None)
                chunks_serializable.append(chunk_copy)
            json.dump(chunks_serializable, f, indent=2, default=str)
        print(f"✅ Chunks saved to: {chunks_path}")
        
        # ============================================
        # STEP 4: Generate Embeddings
        # ============================================
        embedder = ComplaintEmbedder(model_name=CONFIG['embedding_model'])
        embedder.load_model()
        
        chunks_with_embeddings, embeddings = embedder.generate_embeddings(chunks)
        
        # ============================================
        # STEP 5: Build Vector Store
        # ============================================
        vector_store = VectorStoreBuilder(embedding_dim=CONFIG['embedding_dim'])
        index, metadata = vector_store.build_faiss_index(chunks_with_embeddings, embeddings)
        
        # Save vector store
        os.makedirs(CONFIG['vector_store_dir'], exist_ok=True)
        vector_store_path = os.path.join(CONFIG['vector_store_dir'], 'complaint_index')
        vector_store.save_index(vector_store_path)
        
        # ============================================
        # STEP 6: Generate Report
        # ============================================
        report = generate_report(
            sampling_strategy,
            chunk_stats,
            chunks_with_embeddings,
            metadata,
            vector_store
        )
        
        # Save report
        report_path = os.path.join(CONFIG['output_dir'], 'task2_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {report_path}")
        
        # ============================================
        # STEP 7: Test Search
        # ============================================
        print("\n" + "=" * 70)
        print("TESTING VECTOR SEARCH")
        print("=" * 70)
        
        # Test query
        test_query = "Why are customers unhappy with credit card fees?"
        query_embedding = embedder.model.encode([test_query])[0]
        
        results = vector_store.search(query_embedding, k=3)
        
        print(f"\n🔍 Test Query: '{test_query}'")
        print(f"\n📊 Top Results:")
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Product: {result['metadata']['product_category']}")
            print(f"      Issue: {result['metadata']['issue']}")
            print(f"      Score: {result['score']:.4f}")
            print(f"      Text: {result['text'][:150]}...")
        
        # ============================================
        # Summary
        # ============================================
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print("\n" + "=" * 70)
        print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n⏱️  Execution time: {elapsed.total_seconds():.2f} seconds")
        
        print("\n📋 Deliverables Completed:")
        print("   1. ✅ Stratified Sampling: data/processed/sampled_complaints.csv")
        print("   2. ✅ Text Chunking: data/processed/chunks.json")
        print("   3. ✅ Embeddings: Generated and stored in vector store")
        print("   4. ✅ Vector Store: vector_store/complaint_index.faiss")
        print("   5. ✅ Report: data/processed/task2_report.txt")
        
        return vector_store, chunks_with_embeddings
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_report(sampling_strategy, chunk_stats, chunks, metadata, vector_store):
    """
    Generate comprehensive report for Task 2
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TASK 2 REPORT: TEXT CHUNKING, EMBEDDING, AND VECTOR STORE INDEXING")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Section 1: Sampling Strategy
    report_lines.append("1. SAMPLING STRATEGY")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Total records available: {sampling_strategy['total_records']:,}")
    report_lines.append(f"Target sample size: {sampling_strategy['target_sample_size']:,}")
    report_lines.append(f"Actual sample size: {sampling_strategy['actual_sample_size']:,}")
    report_lines.append(f"Random seed: {sampling_strategy['random_seed']}")
    report_lines.append("")
    report_lines.append("Product Distribution:")
    
    for product, details in sampling_strategy['product_distribution'].items():
        sample_pct = details['sample_count'] / details['total_count'] * 100
        report_lines.append(f"  - {product[:40]}: {details['sample_count']:,} samples (from {details['total_count']:,} total, {sample_pct:.1f}%)")
    
    report_lines.append("")
    report_lines.append("Sampling Justification:")
    report_lines.append("Stratified sampling was used to ensure proportional representation across all product categories.")
    report_lines.append("This approach maintains the original distribution of complaints while reducing the dataset")
    report_lines.append("to a manageable size for embedding and indexing.")
    report_lines.append("")
    
    # Section 2: Text Chunking
    report_lines.append("2. TEXT CHUNKING")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Chunk Size: {CONFIG['chunk_size']} characters")
    report_lines.append(f"Chunk Overlap: {CONFIG['chunk_overlap']} characters")
    report_lines.append(f"Minimum Text Length: {CONFIG['min_text_length']} characters")
    report_lines.append("")
    report_lines.append("Chunking Statistics:")
    report_lines.append(f"  - Total complaints processed: {chunk_stats['complaints_processed']:,}")
    report_lines.append(f"  - Total chunks created: {chunk_stats['total_chunks']:,}")
    report_lines.append(f"  - Average chunks per complaint: {chunk_stats['avg_chunks_per_complaint']:.2f}")
    report_lines.append(f"  - Min chunk length: {min(chunk_stats['chunk_lengths'])}")
    report_lines.append(f"  - Max chunk length: {max(chunk_stats['chunk_lengths'])}")
    report_lines.append(f"  - Avg chunk length: {np.mean(chunk_stats['chunk_lengths']):.0f}")
    report_lines.append("")
    report_lines.append("Chunking Justification:")
    report_lines.append("The chosen chunk size of 500 characters with 50-character overlap provides:")
    report_lines.append("  - Balance between context preservation and retrieval precision")
    report_lines.append("  - Good performance for semantic search (not too long, not too short)")
    report_lines.append("  - Overlap ensures continuity between chunks for better context")
    report_lines.append("  - Compatible with the embedding model's maximum sequence length")
    report_lines.append("")
    
    # Section 3: Embedding Model
    report_lines.append("3. EMBEDDING MODEL")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Model: {CONFIG['embedding_model']}")
    report_lines.append(f"Embedding Dimension: {CONFIG['embedding_dim']}")
    report_lines.append("")
    report_lines.append("Model Justification:")
    report_lines.append("all-MiniLM-L6-v2 was chosen for the following reasons:")
    report_lines.append("  1. Good performance on semantic similarity tasks")
    report_lines.append("  2. Compact size (~80MB) - efficient for deployment")
    report_lines.append("  3. Fast inference speed - suitable for production use")
    report_lines.append("  4. 384-dimensional embeddings - good balance of quality and storage")
    report_lines.append("  5. Well-suited for complaint text similarity")
    report_lines.append("  6. Open source with permissive license")
    report_lines.append("  7. Well-maintained and widely used in production systems")
    report_lines.append("")
    
    # Section 4: Vector Store
    report_lines.append("4. VECTOR STORE")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append(f"Vector Store Type: FAISS (IndexFlatIP for cosine similarity)")
    report_lines.append(f"Total Vectors Indexed: {len(metadata):,}")
    report_lines.append(f"Embedding Dimension: {CONFIG['embedding_dim']}")
    report_lines.append(f"Distance Metric: Cosine similarity (inner product on normalized vectors)")
    report_lines.append("")
    report_lines.append("Metadata Stored per Chunk:")
    report_lines.append("  - complaint_id: Original complaint identifier")
    report_lines.append("  - product_category: Product category for filtering")
    report_lines.append("  - issue: Issue type for analysis")
    report_lines.append("  - chunk_index: Position in the original complaint")
    report_lines.append("  - total_chunks: Total chunks per complaint")
    report_lines.append("  - chunk_length: Length of the chunk in characters")
    report_lines.append("  - text: Snippet of the chunk text (first 200 chars)")
    report_lines.append("")
    
    # Section 5: Performance Metrics
    report_lines.append("5. PERFORMANCE METRICS")
    report_lines.append("-" * 50)
    report_lines.append("")
    
    # Calculate some metrics
    if len(metadata) > 0:
        total_vectors = len(metadata)
    else:
        total_vectors = 0
    
    report_lines.append(f"Total Vectors: {total_vectors:,}")
    report_lines.append(f"Memory Usage: {total_vectors * 384 * 4 / 1024**2:.2f} MB (approximate)")
    report_lines.append(f"Index Type: Flat (exact search)")
    report_lines.append("")
    
    # Section 6: Usage Example
    report_lines.append("6. USAGE EXAMPLE")
    report_lines.append("-" * 50)
    report_lines.append("")
    report_lines.append("The vector store can be queried as follows:")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("# Load vector store")
    report_lines.append("from notebooks.02_chunking_embedding_indexing import VectorStoreBuilder")
    report_lines.append("vs = VectorStoreBuilder()")
    report_lines.append("vs.load_index('vector_store/complaint_index')")
    report_lines.append("")
    report_lines.append("# Query")
    report_lines.append("from sentence_transformers import SentenceTransformer")
    report_lines.append("model = SentenceTransformer('all-MiniLM-L6-v2')")
    report_lines.append("query = 'What are common credit card issues?'")
    report_lines.append("query_embedding = model.encode([query])[0]")
    report_lines.append("results = vs.search(query_embedding, k=5)")
    report_lines.append("")
    report_lines.append("# Filter by product")
    report_lines.append("results = vs.search(query_embedding, k=5, product_filter='Credit Card')")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

# ============================================
# SCRIPT ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
