import dataiku
from typing import List, Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger("retrieval_service")

class RetrievalService:
    """Service for enhanced retrieval with query rephrasing for RAG fusion."""
    
    def __init__(self, llm_id=None):
        """Initialize with LLM ID for query rephrasing."""
        self.llm_id = llm_id or "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"
        self._llm = None
    
    def get_llm(self):
        """Get or create the LLM connection."""
        if self._llm is None:
            client = dataiku.api_client()
            project = client.get_default_project()
            self._llm = project.get_llm(self.llm_id)
        return self._llm
    
    async def generate_query_variations(
        self, query: str, num_variations: int = 2
    ) -> List[str]:
        """Generate variations of the query for RAG Fusion."""
        if not query or not query.strip():
            return [query]
            
        prompt = (
            f"Your task is to rephrase the following query in {num_variations} different ways "
            f"to improve search results. Each rephrasing should express the same information need "
            f"but use different wording or perspectives. Format your response as a numbered list, "
            f"with just the rephrased queries.\n\nQuery: {query}\n\nRephrased queries:"
        )
        
        try:
            llm = self.get_llm()
            completion = llm.new_completion()
            completion.with_message(prompt)
            response = completion.execute()
            
            variations = []
            if response.success:
                # Parse the numbered list response
                lines = response.text.strip().split('\n')
                for line in lines:
                    # Extract queries from lines like "1. Query text"
                    if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, num_variations + 1)):
                        # Extract the query text after the number and period
                        parts = line.strip().split('.', 1)
                        if len(parts) > 1:
                            variations.append(parts[1].strip())
            
            # Always include original query and ensure we have enough variations
            variations = [query] + variations
            
            # Log the actual variations being used
            logger.info(f"Generated {len(variations)} query variations")
            for i, var in enumerate(variations):
                logger.info(f"  Variation {i}: '{var}'")
                
            return variations[:num_variations+1]  # Original + variations
            
        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}")
            return [query]  # Return original query if error
    
    def get_rrf_constant(self) -> int:
        """
        Get the constant k for Reciprocal Rank Fusion.
        Lower values give higher weight to top-ranked documents.
        Higher values flatten the ranking distribution.

        Default: 60 (standard in literature)
        """
        # You could make this configurable later
        return 60

    def rerank_results(self, results: List[Any], query: str) -> List[Any]:
        """
        Rerank combined results using Reciprocal Rank Fusion.

        RRF score for a document d is calculated as:
        RRF(d) = sum(1 / (rank_i + k)) for all ranks of d
        where k is a constant (typically 60)
        """
        if not results:
            return []

        try:
            # First, deduplicate results based on content
            content_to_docs = {}

            for doc in results:
                content = getattr(doc, "page_content", "")
                content_hash = hash(content)

                # Prepare source tracking for RRF
                if content_hash not in content_to_docs:
                    content_to_docs[content_hash] = {
                        "doc": doc,
                        "sources": [],
                        "rrf_score": 0.0
                    }

                # Track which source and rank this document came from
                source_kb = getattr(doc, "metadata", {}).get("source_kb", "unknown")
                query_var = getattr(doc, "metadata", {}).get("query_variation", "original")
                rank = getattr(doc, "metadata", {}).get("rank", 0)

                # If rank wasn't provided in metadata, use the position in the results list
                if rank == 0:
                    # Try to find position in original results
                    for i, r in enumerate(results):
                        if getattr(r, "page_content", "") == content:
                            rank = i + 1
                            break

                content_to_docs[content_hash]["sources"].append({
                    "source_kb": source_kb,
                    "query_variation": query_var,
                    "rank": rank
                })

            # Apply Reciprocal Rank Fusion with constant k from method
            k = self.get_rrf_constant()
            for doc_info in content_to_docs.values():
                # Calculate RRF score across all sources this document appeared in
                for source in doc_info["sources"]:
                    # RRF formula: 1 / (rank + k)
                    doc_info["rrf_score"] += 1.0 / (source["rank"] + k)

                # Store final RRF score in the document's metadata
                if not hasattr(doc_info["doc"], "metadata"):
                    doc_info["doc"].metadata = {}
                doc_info["doc"].metadata["rrf_score"] = doc_info["rrf_score"]

                # Also store source information for debugging/analysis
                doc_info["doc"].metadata["source_count"] = len(doc_info["sources"])
                doc_info["doc"].metadata["sources"] = doc_info["sources"]

            # Sort by RRF score (descending)
            unique_docs = [info["doc"] for info in content_to_docs.values()]
            ranked_results = sorted(unique_docs, key=lambda x: getattr(x, "metadata", {}).get("rrf_score", 0), reverse=True)

            logger.info(f"RRF ranking: {len(ranked_results)} unique documents")
            return ranked_results

        except Exception as e:
            logger.error(f"Error in RRF reranking: {str(e)}")
            return results
