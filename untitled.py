from typing import List, Dict
import hashlib
import datetime

# --- Input data (pretend these came from PDFs / APIs) ---

DOCUMENTS = [
    {
        "document_id": "doc-001",
        "source": "sharepoint",
        "version": 3,
        "text": "Azure OpenAI enables secure enterprise-grade AI solutions. "
                "Customers can deploy models with compliance guarantees. "
                "Rohirrim builds GenAI for GovTech."
    },
    {
        "document_id": "doc-002",
        "source": "s3",
        "version": 1,
        "text": "Vector databases store embeddings for semantic search. "
                "Chunking strategy strongly affects retrieval quality."
    }
]

# --- Expected output ---
# A list of chunk records, each ready for embedding:
#
# {
#   "chunk_id": str,
#   "document_id": str,
#   "chunk_index": int,
#   "text": str,
#   "metadata": {...},
#   "created_at": datetime
# }



def build_embedding_chunks(
    documents: List[Dict],
    max_chars: int = 100
) -> List[Dict]:
    """
    1. Normalize text (trim whitespace)
    2. Chunk text into <= max_chars pieces (do NOT split words)
    3. Attach metadata
    4. Generate stable chunk_ids
    """
    pass