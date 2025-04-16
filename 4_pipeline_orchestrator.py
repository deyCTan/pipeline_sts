import asyncio
from typing import List, Dict, Any, Optional
import logging
import time

# Import your service classes - ensure these imports match your file names
from knowledge_service import KnowledgeService
from translation_service import TranslationService
from retrieval_service import RetrievalService

logger = logging.getLogger("pipeline_orchestrator")

class RAGPipeline:
    """Orchestrates the entire RAG pipeline flow."""
    
    def __init__(
        self,
        knowledge_service: KnowledgeService,
        translation_service: TranslationService,
        retrieval_service: RetrievalService
    ):
        """
        Initialize the RAG Pipeline with required services.
        
        Args:
            knowledge_service: Service for knowledge retrieval
            translation_service: Service for language translation
            retrieval_service: Service for query processing and ranking
        """
        self.knowledge_service = knowledge_service
        self.translation_service = translation_service
        self.retrieval_service = retrieval_service
    
    async def process_query(self, query, source_language="auto", target_language=None, search_type="global", project_name=None):
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's query string
            source_language: Language of the query (auto for detection)
            target_language: Preferred language for results (defaults to source_language)
            search_type: global or local search
            project_name: Optional project filter for local search
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        try:
            # Set target_language to source_language if not explicitly provided
            if target_language is None:
                target_language = source_language
                
            # Step 1: Detect language if auto-detection is requested
            detected_language = source_language
            if source_language == "auto":
                detection_result = await self.translation_service.detect_language(query)
                detected_language = detection_result.get("detected_language", "en")
                logger.info(f"Detected query language: {detected_language}")
            
            # Update target language if it wasn't explicitly set
            if target_language == "auto":
                target_language = detected_language
                
            # Store original language for later use
            original_language = detected_language
            
            # Step 2: Translate query to English if needed
            english_query = query
            if detected_language != "en" and detected_language != "english":
                translation_result = await self.translation_service.translate_text(
                    query, source_language=detected_language, target_language="en"
                )
                english_query = translation_result.get("translated_text", query)
                logger.info(f"Translated query to English: {english_query}")
                
            # Step 3: Generate query variations and retrieve results
            query_variations = await self.retrieval_service.generate_query_variations(
                english_query, query_language="en"
            )
            
            # Step 4: Retrieve results from all knowledge bases
            all_results = await self.knowledge_service.search_all_kbs(query_variations)
            
            # Step 5: Apply Reciprocal Rank Fusion to get unique documents
            unique_results = self.retrieval_service.rerank_results(
                all_results, english_query, target_language=target_language
            )
            
            # Step 6: Filter by project if needed
            if search_type == "local" and project_name:
                filtered_results = [
                    doc for doc in unique_results 
                    if getattr(doc, "metadata", {}).get("project", "") == project_name
                ]
                logger.info(f"Filtered to {len(filtered_results)} results for project: {project_name}")
            else:
                filtered_results = unique_results
                logger.info(f"Using all {len(filtered_results)} results (global search)")

            # Step 7: Extract fields for translation
            observation_texts = []
            solution_texts = []

            for result in filtered_results:
                # Ensure we have a metadata dictionary
                if not hasattr(result, "metadata"):
                    result.metadata = {}

                # Extract observation field with proper fallbacks
                obs_text = result.metadata.get("observation_final_translated", 
                           result.metadata.get("observation", 
                           getattr(result, "page_content", "")))
                
                observation_texts.append(obs_text if obs_text else "")

                # Extract solution field with proper fallbacks
                sol_text = result.metadata.get("solution_final_translated", 
                          result.metadata.get("solution", ""))
                
                solution_texts.append(sol_text if sol_text else "")

            # Step 8: Translate observations and solutions to target language if needed
            if target_language != "en" and target_language != "english":
                logger.info(f"Translating {len(filtered_results)} results to {target_language}")

                # Translate observations
                if observation_texts:
                    obs_translations = await self.translation_service.translate_batch(
                        observation_texts, source_language="en", target_language=target_language
                    )
                    translated_observations = obs_translations.get("translated_texts", observation_texts)
                else:
                    translated_observations = []

                # Translate solutions
                if solution_texts:
                    sol_translations = await self.translation_service.translate_batch(
                        solution_texts, source_language="en", target_language=target_language
                    )
                    translated_solutions = sol_translations.get("translated_texts", solution_texts)
                else:
                    translated_solutions = []
            else:
                # No translation needed, use original texts
                translated_observations = observation_texts
                translated_solutions = solution_texts

            # Step 9: Add translations back to results in metadata
            for i, result in enumerate(filtered_results):
                if i < len(translated_observations):
                    # Store in metadata, not as direct attributes
                    result.metadata["observation_final_translated"] = translated_observations[i]
                    
                if i < len(translated_solutions):
                    # Store in metadata, not as direct attributes 
                    result.metadata["solution_final_translated"] = translated_solutions[i]
                
                # Add language information to metadata
                result.metadata["language"] = target_language
            
            # Calculate total time
            total_time = time.time() - start_time
            
            return {
                "status": "success",
                "results": filtered_results,
                "total_time": total_time,
                "metadata": {
                    "detected_language": detected_language,
                    "target_language": target_language,
                    "english_query": english_query,
                    "query_variations": query_variations,
                    "result_count": len(filtered_results)
                }
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def _translate_results(self, results: List[Any], target_language: str) -> List[Dict[str, Any]]:
        """
        Translate results back to user's language.
        
        Args:
            results: List of retrieved documents
            target_language: Language to translate results to
            
        Returns:
            List of formatted results with translated content
        """
        # Skip translation if target is English
        if target_language == "en":
            logger.info("Target language is English, skipping translation")
            return [self._format_result(doc) for doc in results]
        
        # Extract content for translation
        texts_to_translate = []
        observation_indices = []
        solution_indices = []
        
        logger.info(f"Preparing to translate {len(results)} results to {target_language}")
        
        for doc in results:
            # Get metadata
            metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            
            # Get observation_final_translated for translation
            if "observation_final_translated" in metadata and metadata["observation_final_translated"]:
                texts_to_translate.append(metadata["observation_final_translated"])
                observation_indices.append(len(texts_to_translate) - 1)
                logger.debug(f"Added observation text for translation: {metadata['observation_final_translated'][:50]}...")
            else:
                observation_indices.append(None)
                
            # Get solution_final_translated for translation
            if "solution_final_translated" in metadata and metadata["solution_final_translated"]:
                texts_to_translate.append(metadata["solution_final_translated"])
                solution_indices.append(len(texts_to_translate) - 1)
                logger.debug(f"Added solution text for translation: {metadata['solution_final_translated'][:50]}...")
            else:
                solution_indices.append(None)
        
        # Filter out empty texts and translate
        if texts_to_translate:
            logger.info(f"Translating {len(texts_to_translate)} texts from English to {target_language}")
            translation_result = await self.translation_service.translate_batch(
                texts_to_translate, source_language="en", target_language=target_language
            )
            translated_texts = translation_result.get("translated_texts", [])
            logger.info(f"Received {len(translated_texts)} translated texts")
        else:
            logger.warning("No texts to translate")
            translated_texts = []
        
        # Map translations back to results
        formatted_results = []
        
        for i, doc in enumerate(results):
            result = self._format_result(doc)
            metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            
            # Add translated observation
            if observation_indices[i] is not None and observation_indices[i] < len(translated_texts):
                result["observation_final_translated"] = translated_texts[observation_indices[i]]
            else:
                result["observation_final_translated"] = metadata.get("observation_final_translated", "")
                
            # Add translated solution
            if solution_indices[i] is not None and solution_indices[i] < len(translated_texts):
                result["solution_final_translated"] = translated_texts[solution_indices[i]]
            else:
                result["solution_final_translated"] = metadata.get("solution_final_translated", "")
                
            formatted_results.append(result)
        
        logger.info(f"Formatted {len(formatted_results)} results with translations")
        return formatted_results
    
    def _format_result(self, doc: Any) -> Dict[str, Any]:
        """
        Format a document into a standardized result dictionary.
        
        Args:
            doc: Document object from retrieval
            
        Returns:
            Dictionary with formatted result data
        """
        # Handle different types of documents
        if hasattr(doc, "metadata"):
            metadata = doc.metadata
        elif isinstance(doc, dict) and "metadata" in doc:
            metadata = doc["metadata"]
        else:
            metadata = {}
            
        if hasattr(doc, "page_content"):
            content = doc.page_content
        elif isinstance(doc, dict) and "page_content" in doc:
            content = doc["page_content"]
        else:
            content = ""
        
        result = {
            "content": content,
            "source_kb": metadata.get("source_kb", "unknown")
        }
        
        # Add all metadata fields except internal ones
        for key, value in metadata.items():
            if key not in ["source_kb", "query_variation"]:
                result[key] = value
                
        return result
    
def format_results(results, filter_criteria, source_language, translated_observations, translated_solutions):
    """Format results into the desired JSON structure."""
    formatted_results = {"response": {"observation_results": []}}
    
    logger.info(f"Formatting {len(results)} results with filters: {filter_criteria}")
    logger.info(f"Translated obs count: {len(translated_observations)}, sols count: {len(translated_solutions)}")
    
    # Define observation keys to copy from metadata
    OBSERVATION_KEYS = ["field_name", "source", "category", "source_kb", "project"]
    
    obs_index = 0
    sol_index = 0
    
    for idx, result in enumerate(results, start=1):
        # Debug the result structure
        logger.debug(f"Processing result #{idx}: {type(result)}")
        
        # Extract metadata more robustly
        if hasattr(result, "metadata"):
            metadata = result.metadata
        elif isinstance(result, dict) and "metadata" in result:
            metadata = result["metadata"]
        elif isinstance(result, dict):
            # If result itself is a dictionary with fields, use it directly
            metadata = result
        else:
            logger.warning(f"Could not extract metadata from result #{idx}")
            metadata = {}
        
        # Debug available metadata keys
        logger.debug(f"Available metadata keys: {metadata.keys() if metadata else 'None'}")
            
        # Apply filters
        match = True
        for key, value in filter_criteria.items():
            if value and str(metadata.get(key, "")).lower() != str(value).lower():
                logger.info(f"Filtering out result #{idx} - {key}:{metadata.get(key,'')} != {value}")
                match = False
                break
                
        if not match:
            continue
        
        # Create observation with proper field extraction
        observation = {}
        
        # Copy directly available observation keys from metadata
        for key in OBSERVATION_KEYS:
            # Try multiple ways to get the field
            if key in metadata:
                observation[key] = metadata[key]
            elif hasattr(result, key):
                observation[key] = getattr(result, key)
            else:
                observation[key] = ""
                
        observation["ranking"] = idx
        
        # Identify the observation and solution content fields
        obs_content = None
        sol_content = None
        
        # Look for observation in multiple places (metadata-first approach)
        if "observation_final_translated" in metadata:
            obs_content = metadata["observation_final_translated"]
        elif "observation" in metadata:
            obs_content = metadata["observation"]
        elif hasattr(result, "observation_final_translated"):
            obs_content = result.observation_final_translated
        elif hasattr(result, "observation"):
            obs_content = result.observation
        elif hasattr(result, "page_content"):
            # Maybe the content itself is the observation
            obs_content = result.page_content
            
        # Look for solution in multiple places (metadata-first approach)
        if "solution_final_translated" in metadata:
            sol_content = metadata["solution_final_translated"]
        elif "solution" in metadata:
            sol_content = metadata["solution"]
        elif hasattr(result, "solution_final_translated"):
            sol_content = result.solution_final_translated
        elif hasattr(result, "solution"):
            sol_content = result.solution
            
        logger.debug(f"Result #{idx} - Found obs_content: {bool(obs_content)}, sol_content: {bool(sol_content)}")
            
        # Add translated observation if available
        if obs_content and obs_index < len(translated_observations):
            observation["observation"] = translated_observations[obs_index]
            obs_index += 1
        elif obs_content:
            observation["observation"] = obs_content
        else:
            observation["observation"] = ""
            
        # Add translated solution if available
        if sol_content and sol_index < len(translated_solutions):
            observation["solution"] = translated_solutions[sol_index]
            sol_index += 1
        elif sol_content:
            observation["solution"] = sol_content
        else:
            observation["solution"] = ""
            
        formatted_results["response"]["observation_results"].append(observation)
    
    logger.info(f"Formatted {len(formatted_results['response']['observation_results'])} results after filtering")
    return formatted_results
