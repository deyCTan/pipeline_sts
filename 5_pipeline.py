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
        
        print("\nProcessing your query...")
        
        # Process query through pipeline
        response = await pipeline_instance.process_query(
            query=query,
            source_language=source_lang,
            search_type=search_type,
            project_name=project_name
        )
        
        if response["status"] != "success":
            print(f"\nError: {response.get('error', 'Unknown error')}")
            return
            
        # Extract results
        results = response.get("results", [])
        metadata = response.get("metadata", {})
        detected_language = metadata.get("detected_language", source_lang)
        
        # Extract translated fields - improved extraction logic
        translated_observations = []
        translated_solutions = []

        logger.info(f"Processing {len(results)} results for translations")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1} type: {type(result)}")
            
            # Try multiple approaches to get the observations and solutions
            if isinstance(result, dict):
                # Direct dictionary access
                if "observation" in result:
                    translated_observations.append(result["observation"])
                    logger.info(f"Found observation in result dict")
                if "solution" in result:
                    translated_solutions.append(result["solution"])
                    logger.info(f"Found solution in result dict")
                    
                # Check in metadata if present
                if "metadata" in result:
                    meta = result["metadata"]
                    if "observation_final_translated" in meta:
                        obs = meta["observation_final_translated"]
                        if obs and not any(o == obs for o in translated_observations):
                            translated_observations.append(obs)
                    if "solution_final_translated" in meta:
                        sol = meta["solution_final_translated"]
                        if sol and not any(s == sol for s in translated_solutions):
                            translated_solutions.append(sol)
            else:
                # Object attribute access
                if hasattr(result, "observation"):
                    translated_observations.append(result.observation)
                    logger.info(f"Found observation attribute")
                if hasattr(result, "solution"):
                    translated_solutions.append(result.solution)
                    logger.info(f"Found solution attribute")
                    
                # Check metadata attribute if present
                if hasattr(result, "metadata"):
                    meta = result.metadata
                    if hasattr(meta, "observation_final_translated"):
                        obs = meta.observation_final_translated
                        if obs and not any(o == obs for o in translated_observations):
                            translated_observations.append(obs)
                    elif isinstance(meta, dict) and "observation_final_translated" in meta:
                        obs = meta["observation_final_translated"]
                        if obs and not any(o == obs for o in translated_observations):
                            translated_observations.append(obs)
                            
                    if hasattr(meta, "solution_final_translated"):
                        sol = meta.solution_final_translated
                        if sol and not any(s == sol for s in translated_solutions):
                            translated_solutions.append(sol)
                    elif isinstance(meta, dict) and "solution_final_translated" in meta:
                        sol = meta["solution_final_translated"]
                        if sol and not any(s == sol for s in translated_solutions):
                            translated_solutions.append(sol)

        logger.info(f"Extracted {len(translated_observations)} observations and {len(translated_solutions)} solutions")
        
        # Format results
        formatted_results = format_results(
            results=results,
            filter_criteria=project_filter,
            source_language=detected_language,
            translated_observations=translated_observations,
            translated_solutions=translated_solutions
        )
        
        # Add metadata
        formatted_results["metadata"] = {
            "query": query,
            "detected_language": detected_language,
            "search_type": search_type,
            "project_filter": project_name or "",
            "total_time_seconds": response.get("total_time", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Display summary
        result_count = len(formatted_results["response"]["observation_results"])
        print(f"\nFound {result_count} results in {response.get('total_time', 0):.2f} seconds")
        
        if "detected_language" in metadata:
            print(f"Detected language: {metadata['detected_language']}")
                
        if "english_query" in metadata and metadata["english_query"] != query:
            print(f"Translated query: {metadata['english_query']}")
        
        # Print results
        print("\nResults (JSON format):")
        print(json.dumps(formatted_results, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Error in interactive query: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(interactive_query(pipeline))        
