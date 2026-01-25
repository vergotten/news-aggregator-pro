# -*- coding: utf-8 -*-
# =============================================================================
# Путь: src/application/ai_services/orchestrator.py
# =============================================================================
"""
Enhanced AI Orchestrator with LangChain and Multi-Provider Support.

Coordinates all AI agents with:
- Configurable LLM providers (Ollama, OpenRouter)
- Quality validation and retries
- Performance metrics
- Graceful error handling
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from src.infrastructure.ai.llm_provider import (
    LLMProvider, LLMProviderFactory, LLMConfig, LLMProviderType
)
from src.application.ai_services.agents.classifier_agent import ClassifierAgent
from src.application.ai_services.agents.relevance_agent import RelevanceAgent
from src.application.ai_services.agents.summarizer_agent import SummarizerAgent
from src.application.ai_services.agents.rewriter_agent import RewriterAgent
from src.application.ai_services.agents.style_normalizer_agent import StyleNormalizerAgent
from src.application.ai_services.agents.quality_validator_agent import QualityValidatorAgent
from src.config.models_config import ModelsConfig, get_models_config
from src.domain.entities.article import Article

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for a single article processing."""
    article_id: str = ""
    start_time: float = 0.0
    processing_time: float = 0.0
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    relevance_score: Optional[int] = None
    validation_score: Optional[float] = None
    length_ratio: Optional[float] = None
    success: bool = False


class AIOrchestrator:
    """
    Orchestrator for AI article processing pipeline.

    Supports multiple LLM providers:
    - Ollama (local inference)
    - OpenRouter (cloud, OpenAI-compatible)

    Pipeline stages:
    1. Classification (news/article)
    2. Relevance scoring
    3. Summary/teaser generation
    4. Title improvement
    5. Style normalization
    6. Quality validation

    Example:
        >>> # Local Ollama
        >>> orchestrator = AIOrchestrator(provider="ollama")

        >>> # Cloud OpenRouter
        >>> orchestrator = AIOrchestrator(
        ...     provider="openrouter",
        ...     api_key="sk-or-..."
        ... )

        >>> # Process article
        >>> processed = orchestrator.process_article(article)
    """

    def __init__(
            self,
            provider: Optional[str] = None,
            api_key: Optional[str] = None,
            config_path: str = "config/models.yaml",
            enable_validation: bool = True,
            max_retries: int = 2
    ):
        """
        Initialize orchestrator.

        Args:
            provider: LLM provider ("ollama" or "openrouter").
                     If None, uses config file setting.
            api_key: API key for cloud providers (or use env var)
            config_path: Path to models config YAML
            enable_validation: Enable quality validation
            max_retries: Max retries on validation failure
        """
        # Load configuration
        self.config = ModelsConfig(config_path)

        # Override provider if specified
        if provider:
            import os
            os.environ["LLM_PROVIDER"] = provider
            # Reload config with new provider
            self.config = ModelsConfig(config_path)

        if api_key:
            import os
            os.environ["OPENROUTER_API_KEY"] = api_key

        self.enable_validation = enable_validation
        self.max_retries = max_retries

        # Log configuration
        profile = self.config.get_profile()
        logger.info(
            f"AIOrchestrator initializing: "
            f"profile={profile.name}, provider={profile.provider.value}"
        )

        # Initialize agents (they will use config automatically)
        self._init_agents()

        logger.info("AIOrchestrator initialized successfully")

    def _init_agents(self):
        """Initialize all AI agents."""
        self.classifier = ClassifierAgent(config=self.config)
        self.relevance = RelevanceAgent(config=self.config)
        self.summarizer = SummarizerAgent(config=self.config)
        self.rewriter = RewriterAgent(config=self.config)
        self.style_normalizer = StyleNormalizerAgent(config=self.config)

        if self.enable_validation:
            self.validator = QualityValidatorAgent(config=self.config)
            logger.info("QualityValidator enabled")
        else:
            self.validator = None
            logger.info("QualityValidator disabled")

    def process_article(
            self,
            article: Article,
            normalize_style: bool = True,
            validate_quality: bool = True,
            verbose: bool = False,
            min_relevance: int = 5
    ) -> Optional[Article]:
        """
        Full AI processing of an article.

        Args:
            article: Article to process
            normalize_style: Apply style normalization
            validate_quality: Validate normalization quality
            verbose: Print progress to console
            min_relevance: Minimum relevance score (lower scores are flagged)

        Returns:
            Processed article or None if processing failed
        """
        start_time = time.time()
        article_id = str(getattr(article, 'id', 'unknown'))

        logger.info(f"Processing article {article_id}: {article.title[:50]}...")

        if verbose:
            print(f"\n🤖 Processing: {article.title[:50]}...")

        metrics = ProcessingMetrics(
            article_id=article_id,
            start_time=start_time
        )

        try:
            # 1. Classification
            self._step_classify(article, metrics, verbose)

            # 2. Relevance scoring
            score = self._step_relevance(article, metrics, verbose, min_relevance)

            # 3. Summary/teaser
            self._step_summarize(article, metrics, verbose)

            # 4. Title improvement
            self._step_rewrite_title(article, metrics, verbose)

            # 5. Style normalization with validation
            if normalize_style and article.content:
                self._step_normalize(article, metrics, verbose, validate_quality)

            # 6. Update status
            article.embedding_status = "processed"

            # Final metrics
            metrics.processing_time = time.time() - start_time
            metrics.success = len(metrics.steps_failed) == 0

            logger.info(
                f"Article {article_id} processed in {metrics.processing_time:.2f}s - "
                f"Success: {metrics.success}, Failed: {metrics.steps_failed}"
            )

            if verbose:
                print(f"   ✅ Done! Score: {score}/10, Time: {metrics.processing_time:.1f}s")

            return article

        except Exception as e:
            logger.error(f"Article processing failed: {e}", exc_info=True)
            if verbose:
                print(f"   ❌ Processing error: {e}")
            return None

    def _step_classify(
            self,
            article: Article,
            metrics: ProcessingMetrics,
            verbose: bool
    ):
        """Classification step."""
        if verbose:
            print("   1️⃣ Classification...")

        try:
            result = self.classifier.classify_with_details(
                article.title, article.content or ""
            )
            article.is_news = result.is_news
            metrics.steps_completed.append('classification')

            logger.info(
                f"Classification: {'NEWS' if result.is_news else 'ARTICLE'} "
                f"(confidence: {result.confidence:.2f})"
            )
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            metrics.steps_failed.append('classification')
            article.is_news = False  # Default: article

    def _step_relevance(
            self,
            article: Article,
            metrics: ProcessingMetrics,
            verbose: bool,
            min_relevance: int
    ) -> int:
        """Relevance scoring step."""
        if verbose:
            print("   2️⃣ Relevance scoring...")

        try:
            result = self.relevance.score_with_details(
                article.title,
                article.content or "",
                article.tags
            )
            article.set_relevance(result.score, result.reason)
            metrics.steps_completed.append('relevance')
            metrics.relevance_score = result.score

            logger.info(f"Relevance: {result.score}/10 - {result.reason[:50]}...")

            if result.score < min_relevance:
                logger.info(
                    f"Low relevance: {result.score} < {min_relevance}"
                )
                if verbose:
                    print(f"   ⚠️ Low relevance ({result.score}/10)")

            return result.score

        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}", exc_info=True)
            metrics.steps_failed.append('relevance')
            article.set_relevance(5, "Scoring error")
            return 5

    def _step_summarize(
            self,
            article: Article,
            metrics: ProcessingMetrics,
            verbose: bool
    ):
        """Summary/teaser generation step."""
        if verbose:
            print("   3️⃣ Creating teaser...")

        try:
            result = self.summarizer.summarize_with_details(
                article.title,
                article.content or ""
            )
            article.editorial_teaser = result.teaser
            metrics.steps_completed.append('summarizer')

            logger.info(f"Teaser created: {len(result.teaser)} chars")

        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            metrics.steps_failed.append('summarizer')
            # Fallback: first 2 sentences
            sentences = (article.content or "")[:500].split('.')[:2]
            article.editorial_teaser = '. '.join(sentences) + '.'

    def _step_rewrite_title(
            self,
            article: Article,
            metrics: ProcessingMetrics,
            verbose: bool
    ):
        """Title improvement step."""
        if verbose:
            print("   4️⃣ Improving title...")

        try:
            result = self.rewriter.rewrite_with_details(
                article.title,
                article.content or ""
            )
            article.editorial_title = result.improved_title
            metrics.steps_completed.append('rewriter')

            logger.info(
                f"Title: '{article.title[:30]}' → '{result.improved_title[:30]}'"
            )

        except Exception as e:
            logger.error(f"Title rewriting failed: {e}", exc_info=True)
            metrics.steps_failed.append('rewriter')
            article.editorial_title = article.title

    def _step_normalize(
            self,
            article: Article,
            metrics: ProcessingMetrics,
            verbose: bool,
            validate: bool
    ):
        """Style normalization with optional validation."""
        if verbose:
            print(f"   5️⃣ Style normalization... ({self.style_normalizer.model})")

        normalized_content = self._normalize_with_retry(
            article.content,
            verbose=verbose,
            metrics=metrics,
            validate=validate
        )

        if normalized_content:
            article.editorial_rewritten = normalized_content
            metrics.steps_completed.append('normalization')

            logger.info(
                f"Normalization: {len(article.content)} → "
                f"{len(normalized_content)} chars"
            )
        else:
            logger.error("Normalization failed")
            metrics.steps_failed.append('normalization')
            article.editorial_rewritten = article.content

    def _normalize_with_retry(
            self,
            content: str,
            verbose: bool,
            metrics: ProcessingMetrics,
            validate: bool
    ) -> Optional[str]:
        """Normalization with retry on validation failure."""
        for attempt in range(self.max_retries + 1):
            try:
                # Normalize
                result = self.style_normalizer.normalize_with_details(content)
                normalized = result.normalized_text

                # Validate
                if validate and self.enable_validation and self.validator:
                    validation = self.validator.validate(
                        original=content,
                        normalized=normalized,
                        check_ai=False
                    )

                    metrics.validation_score = validation.score
                    metrics.length_ratio = validation.metrics.get('length_ratio', 0)

                    if validation.is_valid:
                        logger.info(
                            f"Validation passed: score={validation.score:.2f}"
                        )
                        return normalized
                    else:
                        logger.warning(
                            f"Validation failed (attempt {attempt + 1}): "
                            f"score={validation.score:.2f}, issues={validation.issues}"
                        )

                        if verbose:
                            print(f"      ⚠️ Validation failed (attempt {attempt + 1})")
                            for issue in validation.issues:
                                print(f"         - {issue}")

                        if attempt >= self.max_retries:
                            logger.error("All validation attempts failed")
                            if verbose:
                                print("      ❌ All attempts failed, using original")
                            return content
                else:
                    # Validation disabled
                    return normalized

            except Exception as e:
                logger.error(f"Normalization attempt {attempt + 1} failed: {e}")
                if attempt >= self.max_retries:
                    return content

        return content

    def check_health(self) -> Dict[str, Any]:
        """
        Check health of all components.

        Returns:
            Health status dictionary
        """
        health = {
            "orchestrator": "healthy",
            "provider": self.config.get_provider().value,
            "profile": self.config.active_profile,
            "agents": {}
        }

        # Check each agent
        agents = [
            ("classifier", self.classifier),
            ("relevance", self.relevance),
            ("summarizer", self.summarizer),
            ("rewriter", self.rewriter),
            ("style_normalizer", self.style_normalizer),
        ]

        if self.validator:
            agents.append(("validator", self.validator))

        for name, agent in agents:
            try:
                health["agents"][name] = {
                    "model": agent.model,
                    "status": "healthy",
                    "metrics": agent.get_metrics()
                }
            except Exception as e:
                health["agents"][name] = {
                    "status": "error",
                    "error": str(e)
                }

        return health

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        profile = self.config.get_profile()

        stats = {
            'active_profile': profile.name,
            'provider': profile.provider.value,
            'agents': {},
            'validation_enabled': self.enable_validation,
            'max_retries': self.max_retries,
        }

        agents = [
            ('classifier', self.classifier),
            ('relevance', self.relevance),
            ('summarizer', self.summarizer),
            ('rewriter', self.rewriter),
            ('style_normalizer', self.style_normalizer),
        ]

        if self.validator:
            agents.append(('validator', self.validator))

        for name, agent in agents:
            stats['agents'][name] = agent.get_metrics()

        return stats

    def reset_metrics(self):
        """Reset all agent metrics."""
        for agent in [
            self.classifier, self.relevance, self.summarizer,
            self.rewriter, self.style_normalizer
        ]:
            agent.reset_metrics()

        if self.validator:
            self.validator.reset_metrics()

        logger.info("All agent metrics reset")


# =============================================================================
# Backward compatibility alias
# =============================================================================
ContentOrchestrator = AIOrchestrator


# =============================================================================
# Convenience factory functions
# =============================================================================
def create_orchestrator(
        provider: str = "ollama",
        profile: str = "balanced",
        api_key: Optional[str] = None,
        **kwargs
) -> AIOrchestrator:
    """
    Create orchestrator with specified settings.

    Args:
        provider: LLM provider ("ollama" or "openrouter")
        profile: Config profile ("balanced", "fast", "quality", "cloud_balanced", etc.)
        api_key: API key for cloud providers
        **kwargs: Additional orchestrator options

    Returns:
        Configured AIOrchestrator

    Example:
        >>> # Local with Ollama
        >>> orch = create_orchestrator("ollama", "balanced")

        >>> # Cloud with OpenRouter
        >>> orch = create_orchestrator("openrouter", "cloud_balanced", api_key="sk-...")
    """
    import os

    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_PROFILE"] = profile

    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    return AIOrchestrator(provider=provider, api_key=api_key, **kwargs)


def create_local_orchestrator(**kwargs) -> AIOrchestrator:
    """Create orchestrator with local Ollama."""
    return create_orchestrator(provider="ollama", **kwargs)


def create_cloud_orchestrator(
        api_key: str,
        profile: str = "cloud_balanced",
        **kwargs
) -> AIOrchestrator:
    """Create orchestrator with OpenRouter cloud."""
    return create_orchestrator(
        provider="openrouter",
        profile=profile,
        api_key=api_key,
        **kwargs
    )