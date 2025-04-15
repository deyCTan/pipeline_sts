import dataiku
from typing import List, Optional
import asyncio
import logging

logger = logging.getLogger("translation_service")

class TranslationService:
    """Service for language detection and batch translation."""
    
    def __init__(self, llm_id=None):
        """Initialize with LLM ID for translation."""
        self.llm_id = llm_id or "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"
        self._llm = None
    
    def get_llm(self):
        """Get or create the LLM connection."""
        if self._llm is None:
            try:
                client = dataiku.api_client()
                project = client.get_default_project()
                self._llm = project.get_llm(self.llm_id)
                logger.info(f"Successfully connected to LLM: {self.llm_id}")
            except Exception as e:
                logger.error(f"Error connecting to LLM: {str(e)}")
                raise
        return self._llm
    
    async def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        if not text or not text.strip():
            logger.info("Empty text provided, defaulting to English")
            return "en"  # Default to English for empty text
        
        try:
            # Try langdetect first (faster)
            try:
                from langdetect import detect
                detected = detect(text)
                logger.info(f"Language detected using langdetect: {detected}")
                return detected
            except ImportError:
                logger.info("langdetect not available, using LLM for detection")
                return await self._detect_with_llm(text)
            except Exception as e:
                logger.warning(f"langdetect failed: {str(e)}, falling back to LLM")
                return await self._detect_with_llm(text)
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return "en"  # Default to English on error
    
    async def _detect_with_llm(self, text: str) -> str:
        """Detect language using LLM."""
        prompt = (
            f"Detect the language of the following text. "
            f"Respond with the language code only (e.g., 'en' for English):\n\n{text}"
        )
        
        try:
            llm = self.get_llm()
            completion = llm.new_completion()
            completion.with_message(prompt)
            response = completion.execute()
            
            if response.success:
                detected = response.text.strip().lower()
                logger.info(f"Language detected using LLM: {detected}")
                return detected
            else:
                logger.error(f"LLM language detection failed: {response.text}")
                return "en"  # Default to English on error
        except Exception as e:
            logger.error(f"Error in LLM language detection: {str(e)}")
            return "en"  # Default to English on error
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text between languages."""
        # Skip if source and target are the same
        if source_lang == target_lang:
            return text
        
        if not text or not text.strip():
            return text
            
        try:
            prompt = (
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"Return only the translated text without explanations.\n\n"
                f"Text: {text}"
            )
            
            llm = self.get_llm()
            completion = llm.new_completion()
            completion.with_message(prompt)
            response = completion.execute()
            
            if response.success:
                translated = response.text.strip()
                logger.debug(f"Translated text from {source_lang} to {target_lang}")
                return translated
            else:
                logger.error(f"Translation failed: {response.text}")
                return text  # Return original on error
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text  # Return original on error
    
    async def batch_translate(
        self,
        texts: List[str],
        source_language: str = "en",
        target_language: str = "en",
        batch_size: int = 5
    ) -> List[str]:
        """Translate a batch of texts in parallel."""
        # Skip if source and target are the same
        if source_language == target_language:
            return texts
        
        if not texts:
            return []
            
        logger.info(f"Batch translating {len(texts)} texts from {source_language} to {target_language}")
            
        try:
            # Filter out empty texts
            non_empty_texts = []
            empty_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    non_empty_texts.append((i, text))
                else:
                    empty_indices.append(i)
            
            # Split into batches
            batches = [non_empty_texts[i:i+batch_size] for i in range(0, len(non_empty_texts), batch_size)]
            all_translated = [""] * len(texts)  # Pre-fill with empty strings
            
            # Process each batch
            for batch_idx, batch in enumerate(batches):
                tasks = []
                batch_indices = []
                
                for idx, text in batch:
                    tasks.append(self.translate_text(text, source_language, target_language))
                    batch_indices.append(idx)
                
                # Translate batch in parallel
                batch_results = await asyncio.gather(*tasks)
                
                # Put results back in original order
                for i, result in enumerate(batch_results):
                    all_translated[batch_indices[i]] = result
                
                # Log progress
                logger.info(f"Translated batch {batch_idx+1}/{len(batches)} ({len(batch)} texts)")
                
                # Small delay between batches to avoid rate limits
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(0.5)
                
            return all_translated
            
        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return texts
