class DataikuRetrievalEvaluator:
    """SentRev-based evaluation framework for retrieval quality in Dataiku"""
    
    def __init__(
        self,
        retrieval_function: Callable,
        model_name: str = "openai/gpt-3.5-turbo-16k",
        kb_dataset_name: str = "knowledge_documents",
        languages: List[str] = ["en", "fr", "kk", "es", "sv", "ru"],
        api_key: Optional[str] = None,
        results_folder: str = "EVALUATIONS"
    ):
        """
        Initialize the retrieval evaluator
        
        Args:
            retrieval_function: Function that accepts query and returns ranked documents
            model_name: LLM to use for query generation and evaluation
            kb_dataset_name: Dataiku dataset name containing knowledge documents
            languages: List of languages to evaluate
            api_key: API key for accessing the LLM
            results_folder: Dataiku folder to store evaluation results
        """
        self.retrieval_function = retrieval_function
        self.kb_dataset_name = kb_dataset_name
        self.results_folder = results_folder
        
        # Initialize SentRev evaluator
        self.evaluator = SentRevEvaluator(
            model=model_name,
            api_key=api_key,
            options=EvalOptions(
                languages=languages,
                eval_dimensions=["relevance", "context_precision"],
                num_synthetic_queries_per_doc=3,
                batch_size=16,
                cache_results=True
            )
        )
        
        # Create results directory if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)
        
    async def load_documents_from_dataiku(
        self, 
        sample_size: int = 100, 
        project_filter: Optional[str] = None,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load documents from Dataiku dataset for evaluation
        
        Args:
            sample_size: Number of documents to sample
            project_filter: Optional filter for specific project
            language_filter: Optional filter for specific language
            
        Returns:
            List of document dictionaries
        """
        try:
            # Dataiku Python API import (only executed in Dataiku)
            import dataiku
            from dataiku.core.intercom import backend_json_call
            
            # Access the dataset
            dataset = dataiku.Dataset(self.kb_dataset_name)
            
            # Build filter conditions for SQL query
            conditions = []
            params = {}
            
            if project_filter:
                conditions.append("project = :project")
                params["project"] = project_filter
                
            if language_filter:
                conditions.append("language = :language")
                params["language"] = language_filter
                
            where_clause = " AND ".join(conditions) if conditions else ""
            
            # Sample documents from dataset
            sql_query = f"""
                SELECT 
                    id, 
                    observation_final_translated as content,
                    language,
                    project,
                    created_at,
                    source_kb
                FROM {self.kb_dataset_name}
                {f"WHERE {where_clause}" if where_clause else ""}
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """
            
            # Execute query
            df = dataset.sql_query(sql_query, params)
            
            # Convert to SentRev document format
            documents = []
            for _, row in df.iterrows():
                if not pd.isna(row['content']) and len(row['content'].strip()) > 50:
                    documents.append({
                        "document_id": str(row['id']),
                        "content": row['content'],
                        "metadata": {
                            "language": row['language'] or "en",
                            "project": row['project'] or "",
                            "source_kb": row['source_kb'] or "",
                            "created_at": row['created_at'] or datetime.now().isoformat()
                        }
                    })
            
            logger.info(f"Loaded {len(documents)} documents from Dataiku dataset")
            return documents
            
        except ImportError:
            # Fallback for non-Dataiku environments (for testing)
            logger.warning("Dataiku not available - using mock data")
            return self._generate_mock_documents(sample_size, language_filter)
            
    def _generate_mock_documents(self, sample_size: int, language_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate mock documents for testing outside Dataiku"""
        languages = [language_filter] if language_filter else ["en", "fr", "de", "es", "it"]
        mock_docs = []
        
        for i in range(sample_size):
            lang = languages[i % len(languages)]
            content = f"This is a sample document {i} in {lang} language. It contains some important information about a product or service."
            if lang == "fr":
                content = f"Ceci est un exemple de document {i} en français. Il contient des informations importantes sur un produit ou service."
            elif lang == "de":
                content = f"Dies ist ein Beispieldokument {i} in deutscher Sprache. Es enthält wichtige Informationen über ein Produkt oder eine Dienstleistung."
                
            mock_docs.append({
                "document_id": f"doc_{i}",
                "content": content,
                "metadata": {
                    "language": lang,
                    "project": f"project_{i % 3}",
                    "source_kb": f"kb_{i % 2}",
                    "created_at": datetime.now().isoformat()
                }
            })
            
        return mock_docs
        
    async def evaluate_retrieval(
        self, 
        documents: List[Dict[str, Any]], 
        query_strategy: str = "synthetic"
    ) -> EvalResult:
        """
        Evaluate retrieval quality using SentRev
        
        Args:
            documents: List of document dictionaries
            query_strategy: Strategy for query generation ('synthetic' or 'extract')
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting retrieval evaluation with {len(documents)} documents")
        
        # Define async wrapper for retrieval function
        async def retrieval_wrapper(query: str, **kwargs):
            # Call the retrieval function and format results for SentRev
            try:
                results = self.retrieval_function(query)
                formatted_results = []
                
                for i, result in enumerate(results[:10]):  # Top 10 results only
                    # Handle different result formats (dict or object)
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        doc_id = result.get("id", f"result_{i}")
                        score = result.get("score", 1.0 - (i * 0.1))
                    else:
                        # Assume object with attributes
                        content = getattr(result, "content", "") or getattr(result, "page_content", "")
                        metadata = getattr(result, "metadata", {}) or {}
                        doc_id = metadata.get("id", f"result_{i}")
                        score = metadata.get("score", 1.0 - (i * 0.1))
                        
                    formatted_results.append({
                        "content": content,
                        "document_id": doc_id,
                        "score": score
                    })
                    
                return formatted_results
            except Exception as e:
                logger.error(f"Error in retrieval function: {str(e)}")
                return []
        
        # Run evaluation
        eval_results = await self.evaluator.evaluate(
            documents=documents,
            retrieval_fn=retrieval_wrapper,
            query_generation_strategy=query_strategy
        )
        
        return eval_results
        
    def save_results_to_dataiku(self, eval_results: EvalResult, run_id: Optional[str] = None):
        """
        Save evaluation results to Dataiku
        
        Args:
            eval_results: Evaluation results from SentRev
            run_id: Optional identifier for this evaluation run
        """
        try:
            # Dataiku Python API import
            import dataiku
            
            # Get output folder
            folder = dataiku.Folder(self.results_folder)
            
            # Generate run ID if not provided
            if not run_id:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            # Save summary metrics
            metrics_path = f"retrieval_metrics_{run_id}.json"
            with folder.get_writer(metrics_path) as writer:
                writer.write(json.dumps(eval_results.metrics, indent=2))
                
            # Save detailed results
            details_path = f"retrieval_details_{run_id}.csv"
            df = eval_results.to_dataframe()
            with folder.get_writer(details_path) as writer:
                writer.write(df.to_csv(index=False))
                
            logger.info(f"Saved evaluation results to Dataiku folder: {self.results_folder}")
            
        except ImportError:
            # Fallback for non-Dataiku environments
            logger.warning("Dataiku not available - saving to local files")
            
            if not run_id:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            # Save to local files
            with open(f"{self.results_folder}/retrieval_metrics_{run_id}.json", "w") as f:
                json.dump(eval_results.metrics, f, indent=2)
                
            eval_results.to_dataframe().to_csv(
                f"{self.results_folder}/retrieval_details_{run_id}.csv", 
                index=False
            )
            
    def create_evaluation_dashboard(self, limit: int = 10):
        """
        Create evaluation dashboard in Dataiku
        
        Args:
            limit: Maximum number of recent evaluations to include
        """
        try:
            # Dataiku Python API import
            import dataiku
            from dataiku.notebooks import Notebook
            
            # Get output folder
            folder = dataiku.Folder(self.results_folder)
            
            # Find all metrics files
            metrics_files = [f for f in folder.list_paths_in_partition() if f.startswith("retrieval_metrics_") and f.endswith(".json")]
            metrics_files.sort(reverse=True)  # Most recent first
            
            # Load metrics from each file
            all_metrics = []
            for file_path in metrics_files[:limit]:
                with folder.get_download_stream(file_path) as f:
                    metrics = json.load(f)
                    run_id = file_path.replace("retrieval_metrics_", "").replace(".json", "")
                    metrics["run_id"] = run_id
                    metrics["date"] = run_id.split("_")[0]
                    all_metrics.append(metrics)
                    
            # Convert to DataFrame for easy charting
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                
                # Create insights notebook
                notebook = Notebook("Retrieval Evaluation Dashboard")
                
                # Add title and description
                notebook.add_markdown("# Retrieval System Evaluation Dashboard")
                notebook.add_markdown("## Performance metrics over time")
                
                # Add charts
                notebook.add_chart(
                    metrics_df, 
                    chart_type="line", 
                    x="date", 
                    y=["avg_relevance_score", "avg_context_precision", "mrr"],
                    title="Key Retrieval Metrics"
                )
                
                notebook.add_chart(
                    metrics_df,
                    chart_type="line",
                    x="date",
                    y="retrieval_success_rate",
                    title="Retrieval Success Rate"
                )
                
                # Add data table
                notebook.add_dataframe(metrics_df, title="Evaluation Runs")
                
                # Save notebook
                notebook.save("Retrieval_Evaluation_Dashboard")
                logger.info("Created evaluation dashboard in Dataiku")
            else:
                logger.warning("No evaluation data found to create dashboard")
                
        except ImportError:
            # Fallback for non-Dataiku environments
            logger.warning("Dataiku not available - cannot create dashboard")

    async def run_evaluation_job(
        self,
        sample_size: int = 100,
        project_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        query_strategy: str = "synthetic",
        create_dashboard: bool = True
    ):
        """
        Run complete evaluation workflow
        
        Args:
            sample_size: Number of documents to sample
            project_filter: Optional filter for specific project
            language_filter: Optional filter for specific language
            query_strategy: Strategy for query generation
            create_dashboard: Whether to create/update dashboard
            
        Returns:
            Evaluation results
        """
        try:
            start_time = datetime.now()
            run_id = start_time.strftime("%Y%m%d_%H%M%S")
            logger.info(f"Starting evaluation run {run_id}")
            
            # Load documents from Dataiku
            documents = await self.load_documents_from_dataiku(
                sample_size=sample_size,
                project_filter=project_filter,
                language_filter=language_filter
            )
            
            if not documents:
                logger.error("No documents found for evaluation")
                return None
                
            # Run evaluation
            eval_results = await self.evaluate_retrieval(
                documents=documents,
                query_strategy=query_strategy
            )
            
            # Save results
            self.save_results_to_dataiku(eval_results, run_id)
            
            # Create dashboard
            if create_dashboard:
                self.create_evaluation_dashboard()
                
            # Log summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Evaluation completed in {duration:.1f} seconds")
            logger.info(f"Results: MRR={eval_results.metrics['mrr']:.3f}, " +
                        f"Avg Relevance={eval_results.metrics['avg_relevance_score']:.3f}")
                
            return eval_results
            
        except Exception as e:
            logger.error(f"Error in evaluation job: {str(e)}")
            return None

# Example Dataiku recipe implementation
def process(input_datasets, output_datasets, output_folders):
    """Dataiku recipe entry point"""
    # Configure your retrieval function
    def retrieval_function(query):
        # Replace with your actual retrieval logic
        # This should call your retrieval_service or knowledge_service
        # Example:
        # return retrieval_service.search(query)
        return []  # Placeholder
    
    # Initialize evaluator
    evaluator = DataikuRetrievalEvaluator(
        retrieval_function=retrieval_function,
        kb_dataset_name=input_datasets[0],  # First input dataset
        results_folder=output_folders[0]     # First output folder
    )
    
    # Create event loop and run evaluation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        eval_results = loop.run_until_complete(evaluator.run_evaluation_job(
            sample_size=100,
            project_filter=None,  # Set to project name or None
            language_filter=None  # Set to language code or None
        ))
        
        if eval_results:
            print("Evaluation completed successfully")
        else:
            print("Evaluation failed")
            
    finally:
        loop.close()
