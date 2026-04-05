# -*- coding: utf-8 -*-
"""
Оркестратор AI-обработки статей v5.2

Изменения v5.2:
- Добавлен CatCommentatorAgent (шаг 11)
- Булево значение enable_cat_comment в process_article
- cat_comment сохраняется в article и добавляется в telegram_post_text
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from src.domain.entities.article import Article
from src.infrastructure.ai.llm_provider import LLMProviderFactory
from src.infrastructure.ai.qdrant_client import QdrantService

from src.infrastructure.skiplist import (
    get_skiplist_service,
    SkipReason,
)

from src.application.ai_services.agents import (
    ClassifierAgent,
    RelevanceAgent,
    SummarizerAgent,
    RewriterAgent,
    StyleNormalizerAgent,
    QualityValidatorAgent,
)
from src.application.ai_services.agents.telegram_formatter_agent import TelegramFormatterAgent
from src.application.ai_services.agents.telegraph_formatter_agent import TelegraphFormatterAgent
from src.application.ai_services.agents.image_prompt_agent import ImagePromptAgent
from src.application.ai_services.agents.image_transform_agent import ImageTransformAgent
from src.application.ai_services.agents.cat_commentator_agent import CatCommentatorAgent

logger = logging.getLogger(__name__)


@dataclass
class ModelAttempt:
    model_id: str
    success: bool
    error: Optional[str] = None
    attempt_time: float = 0.0
    response_time: float = 0.0

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"[{status}] {self.model_id} ({self.response_time:.2f}s)"


@dataclass
class ProcessingStats:
    article_id: str
    title: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    model_attempts: List[ModelAttempt] = field(default_factory=list)
    total_time: float = 0.0
    success: bool = False
    error: Optional[str] = None
    final_model: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class AIOrchestrator:
    """Оркестратор AI-обработки статей v5.2"""

    EXCLUDED_MODELS = [
        "nvidia/nemotron",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "meta-llama/llama-3.2-3b",
        "meta-llama/llama-3.1-8b",
        "google/gemma-2-9b",
        "openai/gpt-oss-120b:free",
    ]

    def __init__(
            self,
            provider: Optional[str] = None,
            strategy: Optional[str] = None,
            enable_fallback: bool = True,
            **kwargs
    ):
        try:
            self.provider = provider
            self.strategy = strategy
            self.enable_fallback = enable_fallback

            if provider:
                self.llm_provider = LLMProviderFactory.create_auto(provider=provider)
            else:
                self.llm_provider = LLMProviderFactory.create_auto()

            self.skiplist = get_skiplist_service()

            # Агенты (ленивая инициализация)
            self._classifier: Optional[ClassifierAgent] = None
            self._relevance: Optional[RelevanceAgent] = None
            self._summarizer: Optional[SummarizerAgent] = None
            self._rewriter: Optional[RewriterAgent] = None
            self._normalizer: Optional[StyleNormalizerAgent] = None
            self._validator: Optional[QualityValidatorAgent] = None
            self._telegram_formatter: Optional[TelegramFormatterAgent] = None
            self._telegraph_formatter: Optional[TelegraphFormatterAgent] = None
            self._image_prompt: Optional[ImagePromptAgent] = None
            self._image_transform: Optional[ImageTransformAgent] = None
            self._cat_commentator: Optional[CatCommentatorAgent] = None

            self.qdrant = QdrantService()

            logger.info(
                f"[Orchestrator] v5.2 Initialized: provider={provider}, "
                f"fallback={enable_fallback}, skiplist={len(self.skiplist.list_urls())} URLs"
            )
            self._log_available_models()

        except Exception as e:
            logger.error(f"[Orchestrator] Init error: {e}")
            raise

    # =========================================================================
    # Ленивая инициализация агентов
    # =========================================================================

    @property
    def classifier(self) -> ClassifierAgent:
        if self._classifier is None:
            self._classifier = ClassifierAgent()
        return self._classifier

    @property
    def relevance(self) -> RelevanceAgent:
        if self._relevance is None:
            self._relevance = RelevanceAgent()
        return self._relevance

    @property
    def summarizer(self) -> SummarizerAgent:
        if self._summarizer is None:
            self._summarizer = SummarizerAgent()
        return self._summarizer

    @property
    def rewriter(self) -> RewriterAgent:
        if self._rewriter is None:
            self._rewriter = RewriterAgent()
        return self._rewriter

    @property
    def normalizer(self) -> StyleNormalizerAgent:
        if self._normalizer is None:
            self._normalizer = StyleNormalizerAgent()
        return self._normalizer

    @property
    def validator(self) -> QualityValidatorAgent:
        if self._validator is None:
            self._validator = QualityValidatorAgent()
        return self._validator

    @property
    def telegram_formatter(self) -> TelegramFormatterAgent:
        if self._telegram_formatter is None:
            self._telegram_formatter = TelegramFormatterAgent()
        return self._telegram_formatter

    @property
    def telegraph_formatter(self) -> TelegraphFormatterAgent:
        if self._telegraph_formatter is None:
            self._telegraph_formatter = TelegraphFormatterAgent()
        return self._telegraph_formatter

    @property
    def image_prompt(self) -> ImagePromptAgent:
        if self._image_prompt is None:
            self._image_prompt = ImagePromptAgent()
        return self._image_prompt

    @property
    def image_transform(self) -> ImageTransformAgent:
        if self._image_transform is None:
            self._image_transform = ImageTransformAgent()
        return self._image_transform

    @property
    def cat_commentator(self) -> CatCommentatorAgent:
        if self._cat_commentator is None:
            self._cat_commentator = CatCommentatorAgent()
        return self._cat_commentator

    # =========================================================================
    # Основной метод обработки
    # =========================================================================

    def process_article(
            self,
            article: Article,
            verbose: bool = False,
            min_relevance: int = 5,
            enable_cat_comment: bool = True,
    ) -> Optional[Article]:
        """
        Полная обработка статьи.

        Args:
            article: Статья для обработки
            verbose: Подробный вывод
            min_relevance: Минимальная релевантность
            enable_cat_comment: Добавлять комментарий НейроКота (default: True)
        """
        start_time = time.time()
        stats = ProcessingStats(
            article_id=str(article.id),
            title=article.title[:50] if article.title else "Untitled"
        )

        try:
            # =========================================================
            # ПРОВЕРКА SKIPLIST
            # =========================================================
            if self.skiplist.should_skip(article.url):
                reason = self.skiplist.get_reason(article.url)
                logger.info(f"[Orchestrator] SKIPPED (skiplist): {article.url[:50]}... ({reason.value if reason else 'unknown'})")
                stats.skipped = True
                stats.skip_reason = reason.value if reason else "in skiplist"
                return None

            logger.info(f"[Orchestrator] Processing: {article.title[:50]}...")

            # =========================================================
            # ШАГ 1: Классификация
            # =========================================================
            logger.info("[Orchestrator] Step 1: Classification...")
            step_start = time.time()

            classification = self.classifier.classify_with_details(
                title=article.title,
                content=(article.content or "")[:1000]
            )
            article.is_news = classification.is_news
            step_time = time.time() - step_start
            stats.model_attempts.append(ModelAttempt(
                model_id=self.classifier.model, success=True, response_time=step_time
            ))
            logger.info(f"[Orchestrator] Classification: {'NEWS' if article.is_news else 'ARTICLE'} ({classification.confidence:.0%})")

            # =========================================================
            # ШАГ 2: Релевантность
            # =========================================================
            logger.info("[Orchestrator] Step 2: Relevance...")
            step_start = time.time()

            relevance_result = self.relevance.score_with_details(
                title=article.title,
                content=(article.content or "")[:1500],
                tags=article.tags
            )
            article.relevance_score = float(relevance_result.score)
            article.relevance_reason = relevance_result.reason
            step_time = time.time() - step_start
            stats.model_attempts.append(ModelAttempt(
                model_id=self.relevance.model, success=True, response_time=step_time
            ))
            logger.info(f"[Orchestrator] Relevance: {article.relevance_score}/10")

            if article.relevance_score < min_relevance:
                logger.info(f"[Orchestrator] Low relevance ({article.relevance_score} < {min_relevance}), skipping further processing")
                stats.success = True
                stats.total_time = time.time() - start_time
                return article

            # =========================================================
            # ШАГ 3: Тизер
            # =========================================================
            logger.info("[Orchestrator] Step 3: Teaser...")
            step_start = time.time()

            summary_result = self.summarizer.summarize_with_details(
                title=article.title,
                content=(article.content or "")[:3000]
            )
            article.editorial_teaser = summary_result.teaser
            step_time = time.time() - step_start
            stats.model_attempts.append(ModelAttempt(
                model_id=self.summarizer.model, success=True, response_time=step_time
            ))
            logger.info(f"[Orchestrator] Teaser: {len(article.editorial_teaser)} chars")

            # =========================================================
            # ШАГ 4: Заголовок
            # =========================================================
            logger.info("[Orchestrator] Step 4: Title rewrite...")
            step_start = time.time()

            title_result = self.rewriter.rewrite_with_details(
                title=article.title,
                content=(article.content or "")[:500]
            )
            article.editorial_title = title_result.improved_title
            step_time = time.time() - step_start
            stats.model_attempts.append(ModelAttempt(
                model_id=self.rewriter.model, success=True, response_time=step_time
            ))
            logger.info(f"[Orchestrator] Title: {article.editorial_title[:50]}...")

            # =========================================================
            # ШАГ 5: Нормализация стиля
            # =========================================================
            logger.info("[Orchestrator] Step 5: Style normalization...")
            step_start = time.time()

            content_length = len(article.content or "")
            normalization_result = self.normalizer.normalize_with_details(
                content=article.content or "",
                url=article.url
            )
            step_time = time.time() - step_start
            stats.model_attempts.append(ModelAttempt(
                model_id=self.normalizer.model, success=True, response_time=step_time
            ))

            validation = self.validator.validate(
                original=article.content or "",
                normalized=normalization_result.normalized_text
            )

            if validation.is_valid:
                article.editorial_rewritten = normalization_result.normalized_text
                logger.info(
                    f"[Orchestrator] Normalized: {content_length} -> {len(normalization_result.normalized_text)} "
                    f"(ratio: {normalization_result.length_ratio:.2f}, chunks: {normalization_result.chunks_processed})"
                )
            else:
                logger.warning(f"[Orchestrator] Normalization failed validation: {validation.issues}")
                article.editorial_rewritten = None
                if normalization_result.used_fallback and content_length > 50000:
                    self.skiplist.add(
                        article.url,
                        SkipReason.CONTEXT_TOO_LONG,
                        content_length=content_length,
                        error_message="Normalization failed"
                    )

            # =========================================================
            # ШАГ 6: Теги
            # =========================================================
            logger.info("[Orchestrator] Step 6: Tags...")
            if not article.tags:
                article.tags = relevance_result.categories or []
            logger.info(f"[Orchestrator] Tags: {article.tags[:5]}")

            # =========================================================
            # ШАГ 7: Telegram форматирование
            # =========================================================
            logger.info("[Orchestrator] Step 7: Telegram formatting...")
            step_start = time.time()

            try:
                telegram_post = self.telegram_formatter.format_for_telegram(
                    title=article.editorial_title or article.title,
                    content=article.editorial_rewritten or article.content or "",
                    source_url=article.url,
                    tags=article.tags or [],
                    images=getattr(article, 'images', [])
                )
                article.telegram_post_text = telegram_post.text
                article.telegram_cover_image = telegram_post.cover_image
                if telegram_post.telegraph_needed:
                    article.telegraph_content_html = telegram_post.telegraph_content

                step_time = time.time() - step_start
                stats.model_attempts.append(ModelAttempt(
                    model_id=self.telegram_formatter.model, success=True, response_time=step_time
                ))
                logger.info(f"[Orchestrator] Telegram: {len(telegram_post.text)} chars")

            except Exception as e:
                logger.warning(f"[Orchestrator] Telegram formatting failed: {e}")

            # =========================================================
            # ШАГ 8: Telegraph форматирование
            # =========================================================
            logger.info("[Orchestrator] Step 8: Telegraph formatting...")
            step_start = time.time()

            try:
                telegraph_content = article.editorial_rewritten or article.content or ""
                if len(telegraph_content) > 500:
                    telegraph_result = self.telegraph_formatter.format_for_telegraph(
                        content=telegraph_content
                    )
                    article.telegraph_content_html = telegraph_result.formatted_text
                    step_time = time.time() - step_start
                    stats.model_attempts.append(ModelAttempt(
                        model_id=self.telegraph_formatter.model, success=True, response_time=step_time
                    ))
                    logger.info(
                        f"[Orchestrator] Telegraph: {telegraph_result.original_length} → "
                        f"{telegraph_result.final_length} chars"
                    )
                else:
                    logger.info("[Orchestrator] Telegraph: контент слишком короткий, пропуск")

            except Exception as e:
                logger.warning(f"[Orchestrator] Telegraph formatting failed: {e}")

            # =========================================================
            # ШАГ 9: Генерация промпта для обложки
            # =========================================================
            logger.info("[Orchestrator] Step 9: Image prompt generation...")
            step_start = time.time()

            try:
                prompt_result = self.image_prompt.generate_prompt(
                    title=article.editorial_title or article.title,
                    content=article.editorial_rewritten or article.content or "",
                    tags=article.tags or [],
                )
                # Кошачий промпт для генерации обложки с НейроКотом
                cat_prompt_result = self.image_prompt.generate_cat_prompt(
                    title=article.editorial_title or article.title,
                    content=article.editorial_rewritten or article.content or "",
                    tags=article.tags or [],
                )
                metadata = getattr(article, 'metadata', {}) or {}
                metadata['image_prompt'] = prompt_result.prompt
                metadata['image_prompt_success'] = prompt_result.success
                metadata['cat_image_prompt'] = cat_prompt_result.prompt
                metadata['cat_image_prompt_success'] = cat_prompt_result.success
                article.metadata = metadata

                step_time = time.time() - step_start
                stats.model_attempts.append(ModelAttempt(
                    model_id=self.image_prompt.model if hasattr(self.image_prompt, 'model') else 'image_prompt',
                    success=True,
                    response_time=step_time
                ))
                logger.info(f"[Orchestrator] Image prompt: {len(prompt_result.prompt)} chars")

            except Exception as e:
                logger.warning(f"[Orchestrator] Image prompt failed: {e}")

            # =========================================================
            # ШАГ 10: Трансформация изображений
            # =========================================================
            logger.info("[Orchestrator] Step 10: Image transformation...")
            step_start = time.time()

            try:
                images = getattr(article, 'images', []) or []
                if images:
                    transform_result = self.image_transform.transform_images(images)
                    if transform_result.success_count > 0:
                        article.images = transform_result.transformed_urls
                        logger.info(
                            f"[Orchestrator] Images: {transform_result.success_count}/"
                            f"{len(images)} transformed"
                        )
                    else:
                        logger.warning("[Orchestrator] Images: no transforms succeeded, keeping originals")
                else:
                    logger.info("[Orchestrator] Images: нет изображений, пропуск")

            except Exception as e:
                logger.warning(f"[Orchestrator] Image transformation failed: {e}")

            # =========================================================
            # ШАГ 11: НейроКот комментарий
            # =========================================================
            if enable_cat_comment:
                logger.info("[Orchestrator] Step 11: Cat comment...")
                step_start = time.time()

                try:
                    cat_comment = self.cat_commentator.comment(
                        title=article.editorial_title or article.title,
                        teaser=article.editorial_teaser or ""
                    )

                    if cat_comment:
                        article.cat_comment = cat_comment

                        # Добавляем в конец telegram_post_text если он есть
                        if article.telegram_post_text:
                            cat_block = (
                                "\n\n"
                                "🐱 <i>" + cat_comment + "</i>"
                            )
                            article.telegram_post_text = article.telegram_post_text + cat_block

                        step_time = time.time() - step_start
                        stats.model_attempts.append(ModelAttempt(
                            model_id=self.cat_commentator.model,
                            success=True,
                            response_time=step_time
                        ))
                        logger.info(f"[Orchestrator] Cat comment: {len(cat_comment)} chars")

                except Exception as e:
                    logger.warning(f"[Orchestrator] Cat comment failed: {e}")
            else:
                logger.info("[Orchestrator] Step 11: Cat comment — отключён")

            # =========================================================
            # Завершение
            # =========================================================
            stats.success = True
            stats.total_time = time.time() - start_time
            stats.final_model = self.llm_provider.model

            self._log_final_stats(article, stats)
            return article

        except Exception as e:
            stats.error = str(e)
            stats.total_time = time.time() - start_time
            logger.error(f"[Orchestrator] Error processing {article.id}: {e}")

            error_msg = str(e).lower()
            if 'context' in error_msg or 'maximum' in error_msg:
                self.skiplist.add(
                    article.url,
                    SkipReason.CONTEXT_TOO_LONG,
                    content_length=len(article.content or ""),
                    error_message=str(e)[:200]
                )
            elif 'rate' in error_msg or '429' in error_msg:
                self.skiplist.add_temporary(
                    article.url,
                    SkipReason.RATE_LIMITED,
                    minutes=5,
                    error_message=str(e)[:200]
                )
            return None

    # =========================================================================
    # Вспомогательные методы
    # =========================================================================

    def _log_available_models(self) -> None:
        try:
            if hasattr(self.llm_provider, '_discovery'):
                models = self.llm_provider._discovery.get_free_models()
                logger.info(f"[Orchestrator] Available models: {len(models)}")
                for i, model in enumerate(models[:5], 1):
                    logger.info(f"  {i}. {model.id} (ctx: {model.context_length})")
                if len(models) > 5:
                    logger.info(f"  ... and {len(models) - 5} more")
        except Exception as e:
            logger.warning(f"[Orchestrator] Could not get models: {e}")

    def _log_final_stats(self, article: Article, stats: ProcessingStats) -> None:
        logger.info(f"[Orchestrator] DONE: {article.id}")
        logger.info(f"  Title: {article.title[:50]}...")
        logger.info(f"  Type: {'NEWS' if article.is_news else 'ARTICLE'}")
        logger.info(f"  Relevance: {article.relevance_score}/10")
        logger.info(f"  Rewritten: {'Yes' if article.editorial_rewritten else 'No'}")
        logger.info(f"  Cat comment: {'Yes' if article.cat_comment else 'No'}")
        logger.info(f"  Time: {stats.total_time:.2f}s")
        logger.info(f"  Models used:")
        for attempt in stats.model_attempts:
            logger.info(f"    {attempt}")

    def get_skiplist_stats(self) -> dict:
        return self.skiplist.get_stats()

    def clear_skiplist(self) -> int:
        return self.skiplist.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "strategy": self.strategy,
            "fallback_enabled": self.enable_fallback,
            "skiplist": self.skiplist.get_stats(),
            "agents_initialized": {
                "classifier": self._classifier is not None,
                "relevance": self._relevance is not None,
                "summarizer": self._summarizer is not None,
                "rewriter": self._rewriter is not None,
                "normalizer": self._normalizer is not None,
                "validator": self._validator is not None,
                "telegram_formatter": self._telegram_formatter is not None,
                "telegraph_formatter": self._telegraph_formatter is not None,
                "cat_commentator": self._cat_commentator is not None,
            }
        }


def create_orchestrator(provider: str = "openrouter", enable_fallback: bool = True, **kwargs) -> AIOrchestrator:
    return AIOrchestrator(provider=provider, enable_fallback=enable_fallback, **kwargs)


def create_local_orchestrator(**kwargs) -> AIOrchestrator:
    return AIOrchestrator(provider="ollama", enable_fallback=False, **kwargs)


def create_cloud_orchestrator(provider: str = "openrouter", **kwargs) -> AIOrchestrator:
    return AIOrchestrator(provider=provider, enable_fallback=True, **kwargs)


ContentOrchestrator = AIOrchestrator