import dataiku
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
import time

# Update imports to be compatible with current LangChain versions
from langchain.schema import Document

logger = logging.getLogger("knowledge_service")

class KnowledgeService:
    """Service for handling knowledge retrieval from multiple knowledge banks."""
    
    def __init__(self, kb_ids=None):
        self.kb_ids = kb_ids or ["F4mLJDV3", "p4wVMEbK"]
        self.kb_retrievers = {}
        self.kb_connections = {}
    
    def connect_to_kb(self, kb_id: str) -> Any:
        """Connect to a knowledge bank and get its retriever."""
        if kb_id in self.kb_retrievers:
            return self.kb_retrievers[kb_id]
        
        try:
            # Connect to Dataiku Knowledge Bank
            client = dataiku.api_client()
            project = client.get_default_project()
            kb_core = project.get_knowledge_bank(kb_id).as_core_knowledge_bank()
            
            # Convert to LangChain retriever
            retriever = kb_core.as_langchain_retriever()
            
            # Cache the connections
            self.kb_retrievers[kb_id] = retriever
            self.kb_connections[kb_id] = kb_core
            
            logger.info(f"Successfully connected to knowledge bank: {kb_id}")
            return retriever
        
        except Exception as e:
            logger.error(f"Failed to connect to knowledge bank {kb_id}: {e}")
            raise
    
    async def retrieve_from_kb(
        self, kb_id: str, query: str, top_results: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve results from a specific knowledge bank."""
        # Get the retriever
        retriever = self.connect_to_kb(kb_id)
        
        # Configure retriever parameters for search
        if hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs["k"] = top_results
        
        # Perform retrieval
        try:
            results = retriever.invoke(query)
        except AttributeError:
            # Fallback for older LangChain versions
            try:
                results = retriever.get_relevant_documents(query)
            except AttributeError:
                # Another fallback
                method = getattr(retriever, "retrieve", None) or getattr(retriever, "search", None)
                if method:
                    results = method(query, top_k=top_results)
                else:
                    raise ValueError(f"Could not find appropriate retrieval method for KB {kb_id}")
        
        # Add source KB to metadata for each document
        for doc in results:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata["source_kb"] = kb_id
            
        # Apply filters if provided
        if filter_criteria:
            results = self._apply_filters(results, filter_criteria)
            
        logger.info(f"Retrieved {len(results)} results from KB {kb_id} for query: '{query}'")
        return results
    
    async def retrieve_from_all_kbs(
        self, query: str, top_results: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve results from all knowledge banks in parallel."""
        # Create tasks for retrieving from each KB
        tasks = []
        for kb_id in self.kb_ids:
            tasks.append(self.retrieve_from_kb(kb_id, query, top_results, filter_criteria))
        
        # Execute all retrieval tasks concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Combine results from all KBs
        all_results = []
        for results in results_list:
            all_results.extend(results)
            
        return all_results
    
    def _apply_filters(self, results: List[Document], filter_criteria: Dict[str, Any]) -> List[Document]:
        """Apply filters to retrieval results."""
        filtered_results = []
        
        for doc in results:
            metadata = getattr(doc, "metadata", {})
            match = True
            
            for key, value in filter_criteria.items():
                if not value:
                    continue
                if metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(doc)
        
        return filtered_results
    
    def _deduplicate_results(self, results: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content hash."""
        seen_hashes = set()
        unique_results = []
        
        for doc in results:
            content = getattr(doc, "page_content", "")
            content_hash = hash(content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(doc)
        
        logger.info(f"Deduplicated {len(results)} results to {len(unique_results)} unique documents")
        return unique_results
    
    async def create_ensemble_retriever(
        self, query_variations: List[str], 
        filter_criteria: Optional[Dict[str, Any]] = None, 
        top_results: int = 10
    ) -> List[Document]:
        """Create an ensemble retrieval using multiple query variations."""
        all_results = []
        logger.info(f"Starting ensemble retrieval with {len(query_variations)} variations")
        
        for query_var in query_variations:
            for kb_id in self.kb_ids:
                logger.info(f"Searching KB {kb_id} with query: '{query_var}'")
                
                # Use standard retrieval
                results = await self.retrieve_from_kb(kb_id, query_var, top_results, filter_criteria)
                
                # Add metadata
                for doc in results:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata["query_variation"] = query_var
                    
                logger.info(f"Retrieved {len(results)} results from KB {kb_id}")
                all_results.extend(results)
        
        return all_results
