import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from vnpt_api_client import VNPTAPIClient


class EmbeddingBuilder:
    def __init__(self, wiki_path: str = "vietnamese_wiki.jsonl", output_dir: str = "./data/vector_db"):
        self.client = VNPTAPIClient()
        self.wiki_path = wiki_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_wiki_data(self, max_records: int = None):
        print(f"Loading wiki data from {self.wiki_path}...")
        records = []
        with open(self.wiki_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except:
                    continue
        print(f"Loaded {len(records)} records")
        return records

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def embed_batch(self, texts: list, batch_size: int = 10):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                batch_emb = self.client.embed(batch)
                if isinstance(batch_emb[0], list):
                    embeddings.extend(batch_emb)
                else:
                    embeddings.append(batch_emb)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")
                for text in batch:
                    try:
                        emb = self.client.embed(text)
                        embeddings.append(emb)
                        time.sleep(0.2)
                    except:
                        embeddings.append([0] * 1024)
        return embeddings

    def build(self, max_records: int = 1000, chunk_size: int = 500):
        records = self.load_wiki_data(max_records)
        
        print(f"Chunking {len(records)} records...")
        chunks = []
        metadata = []
        
        for record in tqdm(records, desc="Chunking"):
            title = record.get("title", "")
            text = record.get("clean_text", "") or record.get("text", "")
            
            if not text:
                continue
            
            text_chunks = self.chunk_text(text, chunk_size=chunk_size)
            for chunk in text_chunks:
                chunks.append(chunk)
                metadata.append({
                    "title": title,
                    "text": chunk[:500]
                })
        
        print(f"Total chunks: {len(chunks)}")
        
        print("Embedding chunks...")
        embeddings = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding")):
            try:
                emb = self.client.embed(chunk)
                embeddings.append(emb)
                
                if (i + 1) % 50 == 0:
                    print(f"  Quota: {self.client.get_quota_status()}")
                    
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str:
                    print(f"\nRate limit at {i}, waiting until next hour...")
                    self._wait_until_next_hour()
                    try:
                        emb = self.client.embed(chunk)
                        embeddings.append(emb)
                    except:
                        embeddings.append([0] * 1024)
                else:
                    print(f"Error: {e}")
                    embeddings.append([0] * 1024)
            
            time.sleep(0.1)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        print(f"Saving to {self.output_dir}...")
        np.save(self.output_dir / "embeddings.npy", embeddings_array)
        
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
        
        with open(self.output_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False)
        
        print(f"Done! Saved {len(embeddings)} embeddings")
        print(f"Quota used: {self.client.get_quota_status()}")
        
        return embeddings_array, metadata

    def _wait_until_next_hour(self):
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0)
        if now.minute > 0 or now.second > 0:
            next_hour = next_hour.replace(hour=now.hour + 1)
        wait_seconds = (next_hour - now).total_seconds() + 10
        print(f"Waiting until {next_hour.strftime('%H:%M')} ({wait_seconds:.0f}s)...")
        time.sleep(wait_seconds)


class SimpleVectorSearch:
    def __init__(self, db_path: str = "./data/vector_db"):
        self.db_path = Path(db_path)
        self.embeddings = None
        self.metadata = None
        self.chunks = None
        self.client = VNPTAPIClient()
        self.load()
    
    def load(self):
        emb_path = self.db_path / "embeddings.npy"
        meta_path = self.db_path / "metadata.json"
        chunks_path = self.db_path / "chunks.json"
        
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            print(f"Loaded {len(self.embeddings)} embeddings")
        
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)

    def search(self, query: str, k: int = 5):
        if self.embeddings is None:
            return []
        
        query_emb = np.array(self.client.embed(query), dtype=np.float32)
        
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        top_k = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k:
            results.append({
                "title": self.metadata[idx]["title"] if self.metadata else "",
                "text": self.chunks[idx] if self.chunks else "",
                "score": float(similarities[idx])
            })
        
        return results


def test_embedding():
    print("=" * 60)
    print("TEST EMBEDDING API")
    print("=" * 60)
    
    client = VNPTAPIClient()
    
    texts = [
        "Thủ đô Việt Nam là Hà Nội",
        "Hồ Chí Minh là thành phố lớn nhất Việt Nam",
        "Python là ngôn ngữ lập trình phổ biến"
    ]
    
    for text in texts:
        try:
            emb = client.embed(text)
            print(f"Text: {text[:50]}...")
            print(f"Embedding dim: {len(emb)}")
            print(f"First 5 values: {emb[:5]}")
            print()
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"Quota: {client.get_quota_status()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "build", "search"], default="test")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()
    
    if args.mode == "test":
        test_embedding()
    
    elif args.mode == "build":
        builder = EmbeddingBuilder()
        builder.build(max_records=args.max_records)
    
    elif args.mode == "search":
        searcher = SimpleVectorSearch()
        if args.query:
            results = searcher.search(args.query, k=3)
            print(f"Query: {args.query}")
            print("-" * 40)
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['title']}] (score: {r['score']:.3f})")
                print(f"   {r['text']}...")
                print()
