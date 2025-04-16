import dataiku
import asyncio
import json
import logging
from datetime import datetime
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_pipeline_interactive")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Import the core components
from knowledge_service import KnowledgeService
from translation_service import TranslationService
from retrieval_service import RetrievalService
from pipeline_orchestrator import RAGPipeline

# Initialize services
knowledge_service = KnowledgeService(kb_ids=["F4mLJDV3", "p4wVMEbK"])
translation_service = TranslationService()
retrieval_service = RetrievalService()
pipeline = RAGPipeline(knowledge_service, translation_service, retrieval_service)

# Define observation keys for formatting
OBSERVATION_KEYS = [
    "category_id", "project", "country", "fleet", "subsystem", "database",
    "observation_category", "obs_id", "failure_class", "problem_code", "problem_cause",
    "problem_remedy", "functional_location", "notifications_number", "date",
    "solution_category", "pbs_code", "symptom_code", "root_cause", "document_link",
    "language", "resource", "min_resources_need", "max_resources_need",
    "the_most_frequent_value_for_resource", "time", "min_time_per_one_person",
    "max_time_per_one_person", "average_time", "frequency_obs", "frequency_sol",
    "min_resources_need_sol", "max_resources_need_sol",
    "the_most_frequent_value_for_resource_sol", "min_time_per_one_person_sol",
    "max_time_per_one_person_sol", "average_time_sol", "sol_category_id"
]

async def interactive_query(pipeline_instance):
    """Interactive query process with the RAG pipeline."""
    try:
        print("\n=== Multilingual RAG System ===\n")
        query = input("Enter your query: ")
        if not query.strip():
            print("Error: Query cannot be empty.")
            return
            
        source_lang = input("Enter source language (or press Enter for auto-detection): ").strip().lower()
        if not source_lang:
            source_lang = "auto"
            print("Using automatic language detection")
        else:
            print(f"Using specified language: {source_lang}")
        
        target_lang = input("Enter target language (or press Enter to use same as source): ").strip().lower()
        if not target_lang:
            target_lang = None
            print("Results will be returned in the same language as the query")
        else:
            print(f"Results will be translated to: {target_lang}")
        
        search_type = input("Search type (global/local): ").strip().lower()
        if search_type not in ["global", "local"]:
            print("Invalid search type. Defaulting to global.")
            search_type = "global"
            
        project_name = None
        project_filter = {}
        if search_type == "local":
            project_name = input("Enter project name: ").strip()
            if project_name:
                project_filter["project"] = project_name
                print(f"Filtering results for project: {project_name}")
            else:
                print("No project specified, using global search")
                search_type = "global"
        
        print("\nProcessing your query...")
        start_time = datetime.now()
        print(f"Started at: {start_time.strftime('%H:%M:%S')}")
        
        # Process query through pipeline
        response = await pipeline_instance.process_query(
            query=query,
            source_language=source_lang,
            target_language=target_lang,
            search_type=search_type,
            project_name=project_name
        )
        
        if response.get("status") != "success":
            print(f"\nError: {response.get('error', 'Unknown error')}")
            return
            
        # Extract results
        results = response.get("results", [])
        metadata = response.get("metadata", {})
        detected_language = metadata.get("detected_language", source_lang)
        target_language = metadata.get("target_language", detected_language)
        
        # Extract metadata fields from results
        translated_observations = []
        translated_solutions = []

        logger.info(f"Processing {len(results)} results for translations")
        for i, result in enumerate(results):
            logger.debug(f"Result {i+1} type: {type(result)}")
            
            # Access metadata with fallbacks
            result_metadata = {}
            if hasattr(result, "metadata") and result.metadata:
                result_metadata = result.metadata
            elif isinstance(result, dict) and "metadata" in result:
                result_metadata = result["metadata"]
            elif isinstance(result, dict):
                # Treat the result itself as metadata
                result_metadata = result
                
            # Extract observation using the observation_final_translated field
            obs_content = result_metadata.get("observation_final_translated", "")
            if obs_content:
                translated_observations.append(obs_content)
                logger.debug(f"Found observation in result metadata")
                
            # Extract solution using the solution_final_translated field  
            sol_content = result_metadata.get("solution_final_translated", "")
            if sol_content:
                translated_solutions.append(sol_content)
                logger.debug(f"Found solution in result metadata")

        logger.info(f"Extracted {len(translated_observations)} observations and {len(translated_solutions)} solutions")
        
        # Format results for display
        formatted_results = format_results(
            results=results,
            filter_criteria=project_filter,
            source_language=detected_language,
            translated_observations=translated_observations,
            translated_solutions=translated_solutions
        )
        
        # Add metadata to formatted results
        formatted_results["metadata"] = {
            "query": query,
            "detected_language": detected_language,
            "target_language": target_language,
            "search_type": search_type,
            "project_filter": project_name or "",
            "total_time_seconds": response.get("total_time", 0),
            "timestamp": datetime.now().isoformat(),
            "query_variations": metadata.get("query_variations", [])
        }
        
        # Display summary
        result_count = len(formatted_results["response"]["observation_results"])
        print(f"\nFound {result_count} results in {response.get('total_time', 0):.2f} seconds")
        
        if detected_language != "auto" and detected_language != source_lang:
            print(f"Detected language: {detected_language}")
                
        if "english_query" in metadata and metadata["english_query"] != query:
            print(f"Translated query: {metadata['english_query']}")
            
        if "query_variations" in metadata and metadata["query_variations"]:
            print(f"Query variations used:")
            for i, var in enumerate(metadata["query_variations"][:3], 1):
                print(f"  {i}. {var}")
            if len(metadata["query_variations"]) > 3:
                print(f"  ...and {len(metadata['query_variations'])-3} more")
        
        # Print results
        print("\nResults (JSON format):")
        print(json.dumps(formatted_results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Error in interactive query: {str(e)}", exc_info=True)

def format_results(results, filter_criteria, source_language, translated_observations, translated_solutions):
    """Format results into the desired JSON structure."""
    formatted_results = {"response": {"observation_results": []}}
    
    logger.info(f"Formatting {len(results)} results with filters: {filter_criteria}")
    logger.info(f"Translated obs count: {len(translated_observations)}, sols count: {len(translated_solutions)}")
    
    obs_index = 0
    sol_index = 0
    
    for idx, result in enumerate(results, start=1):
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
        
        # Get observation_final_translated and solution_final_translated  
        obs_content = metadata.get("observation_final_translated", "")
        sol_content = metadata.get("solution_final_translated", "")
            
        # Add observation content
        if obs_content and obs_index < len(translated_observations):
            observation["observation"] = translated_observations[obs_index]
            obs_index += 1
        else:
            observation["observation"] = obs_content
            
        # Add solution content
        if sol_content and sol_index < len(translated_solutions):
            observation["solution"] = translated_solutions[sol_index]
            sol_index += 1
        else:
            observation["solution"] = sol_content
            
        formatted_results["response"]["observation_results"].append(observation)
    
    logger.info(f"Formatted {len(formatted_results['response']['observation_results'])} results after filtering")
    return formatted_results
        
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(interactive_query(pipeline))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        logger.critical(f"Critical error in main loop: {str(e)}", exc_info=True)
    finally:
        print("\nThank you for using the Multilingual Retrieval System")
