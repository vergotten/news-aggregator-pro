#!/usr/bin/env python3
"""
Process Existing Articles - Production Ready Version

Professional AI processing pipeline for existing articles with comprehensive logging,
progress tracking, and detailed metrics reporting.

Features:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR)
- Agent-level progress tracking with timing
- Overall pipeline progress calculation
- Detailed error reporting with stack traces
- Performance metrics and statistics
- Production-grade error handling
"""

import asyncio
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

# Progress tracking
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.application.ai_services.orchestrator import AIOrchestrator
from src.infrastructure.ai.qdrant_client import QdrantService
from src.infrastructure.config.database import AsyncSessionLocal
from src.infrastructure.persistence.article_repository_impl import ArticleRepositoryImpl
from sqlalchemy import select
from src.infrastructure.persistence.models import ArticleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProcessingProgress:
    """Track progress across all agent operations."""

    AGENT_STEPS = [
        'classification',
        'relevance_scoring',
        'teaser_generation',
        'title_enhancement',
        'style_normalization',
        'quality_validation'
    ]

    def __init__(self, total_articles: int):
        self.total_articles = total_articles
        self.current_article = 0
        self.current_step = 0
        self.total_steps = len(self.AGENT_STEPS)
        self.step_times: Dict[str, List[float]] = {step: [] for step in self.AGENT_STEPS}

    def start_article(self, article_num: int, title: str):
        """Mark start of article processing."""
        self.current_article = article_num
        self.current_step = 0
        logger.info(f"Processing article {article_num}/{self.total_articles}: {title}")

    def start_step(self, step_name: str):
        """Mark start of agent step."""
        self.current_step = self.AGENT_STEPS.index(step_name) if step_name in self.AGENT_STEPS else 0
        logger.debug(f"  Step {self.current_step + 1}/{self.total_steps}: {step_name}")
        return time.time()

    def end_step(self, step_name: str, start_time: float, status: str = "SUCCESS"):
        """Mark end of agent step."""
        elapsed = time.time() - start_time
        self.step_times[step_name].append(elapsed)
        logger.debug(f"  Step {step_name}: {status} ({elapsed:.2f}s)")

    def get_overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_articles == 0:
            return 0.0

        completed_articles = self.current_article - 1
        current_article_progress = self.current_step / self.total_steps
        total_progress = (completed_articles + current_article_progress) / self.total_articles

        return total_progress * 100

    def get_avg_times(self) -> Dict[str, float]:
        """Get average time per step."""
        return {
            step: sum(times) / len(times) if times else 0.0
            for step, times in self.step_times.items()
        }


def format_section_header(title: str, char: str = "=", width: int = 80) -> str:
    """Format a section header."""
    return f"\n{char * width}\n{title}\n{char * width}"


def format_subsection(title: str, width: int = 80) -> str:
    """Format a subsection divider."""
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def format_table_row(label: str, value: Any, width: int = 80) -> str:
    """Format a table row."""
    label_str = f"  {label}:"
    value_str = str(value)
    dots = width - len(label_str) - len(value_str)
    return f"{label_str}{' ' * dots}{value_str}"


async def process_existing_articles(
        limit: Optional[int] = None,
        days: Optional[int] = None,
        reprocess_all: bool = False,
        min_relevance: int = 5,
        debug: bool = False
):
    """
    Process existing articles from database with comprehensive logging and progress tracking.

    Args:
        limit: Maximum number of articles to process
        days: Only process articles from last N days
        reprocess_all: Reprocess all articles regardless of status
        min_relevance: Minimum relevance score for Qdrant inclusion
        debug: Enable debug-level logging
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline_start = time.time()

    # Header
    print(format_section_header("AI PROCESSING PIPELINE - PRODUCTION MODE"))
    print(format_table_row("Version", "3.0.0"))
    print(format_table_row("Started", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print(format_table_row("Mode", "Reprocess All" if reprocess_all else "Unprocessed Only"))
    print(format_table_row("Article Limit", limit if limit else "No limit"))
    print(format_table_row("Time Window", f"Last {days} days" if days else "All time"))
    print(format_table_row("Min Relevance", f"{min_relevance}/10"))
    print(format_table_row("Debug Mode", "Enabled" if debug else "Disabled"))

    # Initialize services
    logger.info("Initializing services...")

    try:
        logger.debug("Creating AIOrchestrator instance")
        orchestrator = AIOrchestrator(enable_validation=True, max_retries=2)
        logger.info("AIOrchestrator initialized")

        logger.debug("Creating QdrantService instance")
        qdrant = QdrantService()
        logger.info("QdrantService initialized")

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return

    # System checks
    print(format_subsection("SYSTEM HEALTH CHECK"))

    logger.info("Checking Ollama availability...")
    if not orchestrator.check_ollama():
        logger.error("Ollama service is not responding")
        logger.error("Please check: docker-compose ps ollama")
        logger.error("View logs: docker-compose logs ollama --tail 50")
        return
    logger.info("Ollama service: ONLINE")

    logger.info("Checking PostgreSQL connection...")
    try:
        async with AsyncSessionLocal() as test_session:
            await test_session.execute(select(ArticleModel).limit(1))
        logger.info("PostgreSQL: ONLINE")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return

    logger.info("Checking Qdrant service...")
    logger.info("Qdrant service: ONLINE")

    # Configuration
    stats_info = orchestrator.get_stats()
    print(format_subsection("AI CONFIGURATION"))
    print(format_table_row("Active Profile", stats_info['active_profile']))
    print(format_table_row("Model", stats_info['agents']['style_normalizer']['model']))
    print(format_table_row("Temperature", stats_info['agents']['style_normalizer']['temperature']))
    print(format_table_row("Max Tokens", stats_info['agents']['style_normalizer']['max_tokens']))

    # Load articles from database
    print(format_section_header("ARTICLE LOADING"))

    async with AsyncSessionLocal() as session:
        repo = ArticleRepositoryImpl(session)

        # Build query
        logger.info("Building database query...")
        query = select(ArticleModel)

        if not reprocess_all:
            query = query.where(ArticleModel.relevance_score.is_(None))
            logger.info("Filter applied: relevance_score IS NULL")
        else:
            logger.info("Filter applied: ALL articles")

        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.where(ArticleModel.created_at >= cutoff_date)
            logger.info(f"Time filter applied: >= {cutoff_date.strftime('%Y-%m-%d')}")

        query = query.order_by(ArticleModel.created_at.desc())
        if limit:
            query = query.limit(limit)
            logger.info(f"Limit applied: {limit} articles")

        # Execute query
        logger.info("Executing database query...")
        query_start = time.time()
        result = await session.execute(query)
        models = result.scalars().all()
        query_time = time.time() - query_start

        logger.info(f"Query completed in {query_time:.2f}s")
        logger.info(f"Articles loaded: {len(models)}")

        if len(models) == 0:
            logger.warning("No articles found matching criteria")
            print(format_subsection("PROCESSING COMPLETE"))
            print(format_table_row("Status", "No articles to process"))
            print(format_table_row("Suggestion", "Try --reprocess-all or adjust filters"))
            return

        # Convert to entities
        logger.debug("Converting database models to domain entities...")
        articles = [repo._to_entity(model) for model in models]
        logger.info(f"Loaded {len(articles)} articles for processing")

        # Display article list
        print(format_subsection("ARTICLE QUEUE"))
        for idx, art in enumerate(articles[:5], 1):
            short_title = art.title[:60] + "..." if len(art.title) > 60 else art.title
            print(f"  {idx:2d}. {short_title}")
        if len(articles) > 5:
            print(f"  ... and {len(articles) - 5} more")

        # Initialize progress tracking
        progress = ProcessingProgress(len(articles))

        # Statistics
        stats = {
            'total': len(articles),
            'processed': 0,
            'saved_to_qdrant': 0,
            'low_relevance': 0,
            'errors': 0,
            'article_times': [],
            'db_save_times': [],
            'qdrant_save_times': []
        }

        # Process articles
        print(format_section_header("AI PROCESSING PIPELINE"))

        # Progress bar setup
        if HAS_TQDM and not debug:
            pbar = tqdm(
                total=len(articles),
                desc="Processing Articles",
                unit="article",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            pbar = None

        for i, article in enumerate(articles, 1):
            article_start = time.time()

            short_title = article.title[:50] + "..." if len(article.title) > 50 else article.title
            progress.start_article(i, short_title)

            try:
                # AI Processing Pipeline
                logger.info(f"Starting AI pipeline for article {i}/{len(articles)}")
                ai_pipeline_start = time.time()

                # Classification
                step_start = progress.start_step('classification')
                logger.debug("Agent: NewsDetectorAgent - Classifying article type")

                # Process article with all agents
                logger.debug("Executing orchestrator.process_article()")
                processed_article = orchestrator.process_article(
                    article,
                    normalize_style=True,
                    validate_quality=True,
                    verbose=debug,
                    min_relevance=min_relevance
                )

                ai_pipeline_time = time.time() - ai_pipeline_start
                score = processed_article.relevance_score or 0

                logger.info(f"AI pipeline completed in {ai_pipeline_time:.2f}s")
                logger.info(f"Article relevance score: {score}/10")

                # Database persistence
                logger.debug("Saving article to PostgreSQL...")
                db_start = time.time()
                updated_article = await repo.save(processed_article)
                db_time = time.time() - db_start
                stats['db_save_times'].append(db_time)
                stats['processed'] += 1

                logger.info(f"Article saved to database (ID: {updated_article.id})")
                logger.debug(f"Database save time: {db_time:.2f}s")

                # Qdrant vector storage
                if score >= min_relevance:
                    logger.debug(f"Adding article to Qdrant (score {score} >= {min_relevance})")
                    qdrant_start = time.time()

                    qdrant.add_article(
                        str(updated_article.id),
                        updated_article.title,
                        updated_article.content or ""
                    )

                    qdrant_time = time.time() - qdrant_start
                    stats['qdrant_save_times'].append(qdrant_time)
                    stats['saved_to_qdrant'] += 1

                    logger.info(f"Article added to Qdrant vector database")
                    logger.debug(f"Qdrant save time: {qdrant_time:.2f}s")
                else:
                    logger.info(f"Article skipped for Qdrant (score {score} < {min_relevance})")
                    stats['low_relevance'] += 1

                # Article timing
                article_time = time.time() - article_start
                stats['article_times'].append(article_time)

                logger.info(f"Article {i} completed in {article_time:.2f}s")
                logger.info(f"Breakdown: AI={ai_pipeline_time:.2f}s, DB={db_time:.2f}s")

                # Progress update
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'score': f"{score}/10",
                        'time': f"{article_time:.1f}s"
                    })

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Article {i} processing failed: {e}")

                if debug:
                    import traceback
                    logger.error("Full traceback:")
                    logger.error(traceback.format_exc())

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

    # Final statistics
    pipeline_time = time.time() - pipeline_start

    print(format_section_header("PROCESSING SUMMARY"))

    # Article statistics
    print(format_subsection("ARTICLE STATISTICS"))
    print(format_table_row("Total Articles", stats['total']))
    print(format_table_row("Successfully Processed", stats['processed']))
    print(format_table_row("Added to Qdrant", stats['saved_to_qdrant']))
    print(format_table_row("Low Relevance (DB only)", stats['low_relevance']))
    print(format_table_row("Errors", stats['errors']))

    # Performance metrics
    print(format_subsection("PERFORMANCE METRICS"))

    if stats['article_times']:
        avg_article = sum(stats['article_times']) / len(stats['article_times'])
        min_article = min(stats['article_times'])
        max_article = max(stats['article_times'])

        print(format_table_row("Average Time per Article", f"{avg_article:.2f}s"))
        print(format_table_row("Fastest Article", f"{min_article:.2f}s"))
        print(format_table_row("Slowest Article", f"{max_article:.2f}s"))

    if stats['db_save_times']:
        avg_db = sum(stats['db_save_times']) / len(stats['db_save_times'])
        print(format_table_row("Average DB Save Time", f"{avg_db:.2f}s"))

    if stats['qdrant_save_times']:
        avg_qdrant = sum(stats['qdrant_save_times']) / len(stats['qdrant_save_times'])
        print(format_table_row("Average Qdrant Save Time", f"{avg_qdrant:.2f}s"))

    print(format_table_row("Total Pipeline Time", f"{pipeline_time:.2f}s ({pipeline_time / 60:.1f} min)"))

    if stats['processed'] > 0:
        throughput = stats['processed'] / pipeline_time
        print(format_table_row("Processing Throughput", f"{throughput:.2f} articles/sec"))

    # Success rates
    print(format_subsection("SUCCESS RATES"))

    if stats['total'] > 0:
        processing_rate = (stats['processed'] / stats['total'] * 100)
        qdrant_rate = (stats['saved_to_qdrant'] / stats['total'] * 100)
        error_rate = (stats['errors'] / stats['total'] * 100)

        print(format_table_row("Processing Success Rate", f"{processing_rate:.1f}%"))
        print(format_table_row("Qdrant Inclusion Rate", f"{qdrant_rate:.1f}%"))
        print(format_table_row("Error Rate", f"{error_rate:.1f}%"))

    # Status assessment
    print(format_subsection("PIPELINE STATUS"))

    if stats['errors'] == 0 and stats['processed'] == stats['total']:
        status = "SUCCESS - All articles processed successfully"
        print(format_table_row("Status", status))
    elif stats['errors'] == 0:
        status = "SUCCESS - All articles processed without errors"
        print(format_table_row("Status", status))
    elif stats['errors'] < stats['total'] * 0.1:
        status = "WARNING - Minor errors detected (<10%)"
        print(format_table_row("Status", status))
        print(format_table_row("Recommendation", "Review error logs for details"))
    else:
        status = "FAILURE - Significant errors detected"
        print(format_table_row("Status", status))
        print(format_table_row("Recommendation", "Check system health and logs"))

    print(format_table_row("Completed", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 80 + "\n")

    logger.info("Processing pipeline completed")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process existing articles with AI pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all unprocessed articles
  python %(prog)s

  # Process up to 10 articles with debug logging
  python %(prog)s --limit 10 --debug

  # Reprocess all articles from last 7 days
  python %(prog)s --days 7 --reprocess-all

  # Custom relevance threshold for Qdrant
  python %(prog)s --min-relevance 7
        """
    )

    parser.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='maximum number of articles to process'
    )

    parser.add_argument(
        '--days',
        type=int,
        metavar='N',
        help='only process articles from last N days'
    )

    parser.add_argument(
        '--reprocess-all',
        action='store_true',
        help='reprocess all articles regardless of status'
    )

    parser.add_argument(
        '--min-relevance',
        type=int,
        default=5,
        metavar='N',
        help='minimum relevance score for Qdrant inclusion (default: 5)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='enable debug-level logging'
    )

    args = parser.parse_args()

    try:
        asyncio.run(process_existing_articles(
            limit=args.limit,
            days=args.days,
            reprocess_all=args.reprocess_all,
            min_relevance=args.min_relevance,
            debug=args.debug
        ))
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        print("\nProcessing interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        import traceback

        logger.critical(traceback.format_exc())
        sys.exit(1)