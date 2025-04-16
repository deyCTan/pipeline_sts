import dataiku
from typing import List, Dict, Any, Optional, Union
import asyncio
import logging
import time

logger = logging.getLogger("translation_service")

class TranslationService:
    """Service for language detection and batch translation."""
    
    def __init__(self, llm_id=None):
        """Initialize with LLM ID for translation."""
        self.llm_id = llm_id or "openai:Lite_llm_STS_Dev_GPT_4O:gpt-35-turbo-16k"
        self._llm = None
        self._translation_cache = {}  # Cache for translations to avoid redundant API calls
        self._language_name_to_code = {
            "english": "en",
            "french": "fr",
            "spanish": "es",
            "kazakh": "kk",
            "italian": "it",
            "swedish": "sv",
            "russian": "ru"
            # Add more mappings as needed
        }
    
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
    
    def normalize_language_code(self, language: str) -> str:
        """Normalize language input to standard ISO code."""
        if not language:
            return "en"
            
        # Convert to lowercase and trim
        language = language.lower().strip()
        
        # If it's already a 2-letter code, return it
        if len(language) == 2:
            return language
            
        # Check if it's a language name we know
        if language in self._language_name_to_code:
            return self._language_name_to_code[language]
            
        # Use first 2 chars if it looks like a language code
        if len(language) > 2 and language[:2].isalpha():
            return language[:2]
            
        # Default to English
        return "en"
    
    async def detect_language(self, text: str, metadata_lang: Optional[str] = None) -> Dict[str, str]:
        """
        Detect language of input text.
        
        Args:
            text: Text to detect language for
            metadata_lang: Language from metadata if available
            
        Returns:
            Dict with detected_language and confidence
        """
        # If metadata language is provided, prioritize it
        if metadata_lang:
            normalized_lang = self.normalize_language_code(metadata_lang)
            logger.info(f"Using language from metadata: {normalized_lang}")
            return {
                "detected_language": normalized_lang,
                "confidence": 1.0,
                "source": "metadata"
            }
            
        if not text or not text.strip():
            logger.info("Empty text provided, defaulting to English")
            return {
                "detected_language": "en",
                "confidence": 1.0,
                "source": "default"
            }
        
        try:
            # Try langdetect first (faster)
            try:
                from langdetect import detect, detect_langs
                detected = detect(text)
                # Get confidence too if available
                try:
                    langs = detect_langs(text)
                    confidence = next((l.prob for l in langs if l.lang == detected), 0.5)
                except:
                    confidence = 0.5
                    
                logger.info(f"Language detected using langdetect: {detected} (confidence: {confidence:.2f})")
                
                # If confidence is low, fall back to LLM
                if confidence < 0.6:
                    logger.info(f"Low confidence detection, falling back to LLM")
                    return await self._detect_with_llm(text)
                    
                return {
                    "detected_language": detected,
                    "confidence": confidence,
                    "source": "langdetect"
                }
                
            except ImportError:
                logger.info("langdetect not available, using LLM for detection")
                return await self._detect_with_llm(text)
                
            except Exception as e:
                logger.warning(f"langdetect failed: {str(e)}, falling back to LLM")
                return await self._detect_with_llm(text)
                
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return {
                "detected_language": "en",
                "confidence": 0.0,
                "source": "fallback"
            }
    
    async def _detect_with_llm(self, text: str) -> Dict[str, str]:
        """Detect language using LLM."""
        # Use a shorter sample for better efficiency
        text_sample = text[:500] if len(text) > 500 else text
        
        prompt = (
            f"Detect the language of the following text. "
            f"Respond with ONLY the language code (e.g., 'en' for English, 'fr' for French):\n\n{text_sample}"
        )
        
        try:
            llm = self.get_llm()
            completion = llm.new_completion()
            completion.with_message(prompt)
            response = completion.execute()
            
            if response.success:
                detected = response.text.strip().lower()
                # Normalize to standard 2-letter code
                detected = self.normalize_language_code(detected)
                logger.info(f"Language detected using LLM: {detected}")
                return {
                    "detected_language": detected,
                    "confidence": 0.8,  # LLM detection is usually reliable
                    "source": "llm"
                }
            else:
                logger.error(f"LLM language detection failed: {response.text}")
                return {
                    "detected_language": "en",
                    "confidence": 0.0,
                    "source": "fallback"
                }
        except Exception as e:
            logger.error(f"Error in LLM language detection: {str(e)}")
            return {
                "detected_language": "en",
                "confidence": 0.0,
                "source": "fallback"
            }
    
    async def translate_text(
        self, text: str, source_language: str = "en", target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate a single text between languages.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Dictionary with translated text and metadata
        """
        # Normalize language codes
        source_language = self.normalize_language_code(source_language)
        target_language = self.normalize_language_code(target_language)
        
        # Skip if source and target are the same
        if source_language == target_language:
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "skipped": True
            }
        
        if not text or not text.strip():
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "skipped": True
            }
            
        # Check cache first
        cache_key = f"{source_language}|{target_language}|{hash(text)}"
        if cache_key in self._translation_cache:
            logger.debug(f"Translation cache hit: {source_language} to {target_language}")
            return self._translation_cache[cache_key]
            
        try:
            prompt = (
                f"Translate the following text from {source_language} to {target_language}. "
                f"Return only the translated text without explanations.\n\n"
                f"Text: {text}"
            )
            
            llm = self.get_llm()
            completion = llm.new_completion()
            completion.with_message(prompt)
            response = completion.execute()
            
            if response.success:
                translated = response.text.strip()
                logger.debug(f"Translated text from {source_language} to {target_language}")
                
                result = {
                    "translated_text": translated,
                    "source_language": source_language,
                    "target_language": target_language,
                    "skipped": False
                }
                
                # Cache the result
                self._translation_cache[cache_key] = result
                
                return result
            else:
                logger.error(f"Translation failed: {response.text}")
                return {
                    "translated_text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "error": response.text,
                    "skipped": True
                }
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "error": str(e),
                "skipped": True
            }
    
    async def translate_batch(
        self,
        texts: List[str],
        source_language: str = "en",
        target_language: str = "en",
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Translate a batch of texts in parallel.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code  
            batch_size: Number of texts to translate in parallel
            
        Returns:
            Dictionary with translated texts and metadata
        """
        # Normalize language codes
        source_language = self.normalize_language_code(source_language)
        target_language = self.normalize_language_code(target_language)
        
        # Skip if source and target are the same
        if source_language == target_language:
            return {
                "translated_texts": texts,
                "source_language": source_language,
                "target_language": target_language,
                "skipped": True
            }
        
        if not texts:
            return {
                "translated_texts": [],
                "source_language": source_language,
                "target_language": target_language,
                "skipped": True
            }
            
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
            all_translated = [""] * len(texts)
            
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
                    all_translated[batch_indices[i]] = result.get("translated_text", "")
                
                # Log progress
                logger.info(f"Translated batch {batch_idx+1}/{len(batches)} ({len(batch)} texts)")
                
                # Small delay between batches to avoid rate limits
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(0.5)
                
            return {
                "translated_texts": all_translated,
                "source_language": source_language,
                "target_language": target_language,
                "skipped": False
            }
            
        except Exception as e:
            logger.error(f"Error in batch translation: {str(e)}")
            return {
                "translated_texts": texts,
                "source_language": source_language,
                "target_language": target_language,
                "error": str(e),
                "skipped": True
            }
    
    # Alias for backward compatibility
    batch_translate = translate_batch
