#!/bin/bash
# =============================================================================
# git_push.sh — Коммит всех изменений логическими группами и push
# =============================================================================
# Запуск: bash git_push.sh
# =============================================================================

set -e

echo "🔍 Проверяем статус..."
git status --short
echo ""

# ---------------------------------------------------------
# 0. Конфиг и инфраструктура
# ---------------------------------------------------------
echo "📦 Commit 0: Config & Infrastructure"
git add .dockerignore
git add .env.example
git add .gitignore
git add docker-compose.yml
git add docker/init-db.sql
git add requirements.txt
git commit -m "chore: update Docker config, deps, .dockerignore

- Add .dockerignore (exclude ollama_models/ from build context)
- Update docker-compose.yml
- Update requirements.txt
- Update .env.example, .gitignore, init-db.sql" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 1. Domain & Persistence
# ---------------------------------------------------------
echo "📦 Commit 1: Domain & Persistence"
git add src/domain/entities/article.py
git add src/domain/entities/n8n_workflow.py
git add src/domain/services/n8n_service.py
git add src/infrastructure/persistence/article_repository_impl.py
git add src/infrastructure/persistence/models.py
git add src/infrastructure/config/settings.py
git add migrate_db.py
git commit -m "feat: update domain entities and persistence layer

- Update Article entity with editorial fields
- Add n8n workflow entity and service
- Update article repository and DB models
- Update settings, add migrate_db.py" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 2. LLM Provider & Models Config (core fix)
# ---------------------------------------------------------
echo "📦 Commit 2: LLM Provider — per-agent model support"
git add config/
git add src/config/__init__.py
git add src/config/models_config.py
git add src/infrastructure/ai/__init__.py
git add src/infrastructure/ai/llm_provider.py
git add src/infrastructure/ai/ollama_client.py
git add src/infrastructure/ai/openrouter_models.py
git add docs/MODELS.md
git commit -m "feat: per-agent model routing via models.yaml

- Add agent_models section to models.yaml (per-agent Ollama model override)
- Add get_ollama_model(agent_name) to ModelsConfig
- Add get_ollama_base_url(), get_ollama_context_length() helpers
- Remove hardcoded qwen defaults from llm_provider.py, ollama_client.py
- LLMProviderFactory.create_auto() now reads env vars
- Priority: agent_models.X → env OLLAMA_MODEL → ollama.model → default" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 3. AI Agents
# ---------------------------------------------------------
echo "📦 Commit 3: AI Agents update"
git add src/application/ai_services/agents/__init__.py
git add src/application/ai_services/agents/base_agent.py
git add src/application/ai_services/agents/classifier_agent.py
git add src/application/ai_services/agents/relevance_agent.py
git add src/application/ai_services/agents/summarizer_agent.py
git add src/application/ai_services/agents/rewriter_agent.py
git add src/application/ai_services/agents/style_normalizer_agent.py
git add src/application/ai_services/agents/quality_validator_agent.py
git add src/application/ai_services/agents/telegram_formatter_agent.py
git commit -m "feat: update all AI agents

- Update base_agent with improved retry logic
- Update classifier, relevance, summarizer, rewriter agents
- StyleNormalizer v9.4 with chunking for long texts
- Add TelegramFormatterAgent
- QualityValidator improvements" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 4. Orchestrator (agents no longer receive shared llm_provider)
# ---------------------------------------------------------
echo "📦 Commit 4: Orchestrator — agents self-resolve models"
git add src/application/ai_services/orchestrator.py
git commit -m "fix: orchestrator no longer passes shared llm_provider to agents

- Remove llm_provider= from all agent constructors
- Each agent now calls get_llm_config(agent_name) independently
- This enables per-agent model routing from models.yaml
- Integrate SkiplistService for problematic URLs" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 5. Skiplist
# ---------------------------------------------------------
echo "📦 Commit 5: Skiplist service"
git add src/infrastructure/skiplist/
git commit -m "feat: add SkiplistService for problematic URLs

- Skip articles that cause context overflow or rate limits
- Temporary and permanent skip support
- Auto-add on repeated failures" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 6. Telegram & Telegraph
# ---------------------------------------------------------
echo "📦 Commit 6: Telegram & Telegraph publishers"
git add src/infrastructure/telegram/telegraph_publisher.py
git add src/infrastructure/telegram/telegram_publisher.py
git commit -m "feat: Telegraph and Telegram publishing

- TelegraphPublisher: create pages with images
- TelegramPublisher: send formatted posts to channel
- Update telegram_publisher with new post format" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 7. API routes
# ---------------------------------------------------------
echo "📦 Commit 7: API routes"
git add src/api/routes/pipeline.py
git add src/api/routes/webhooks.py
git add src/main.py
git commit -m "feat: add pipeline API route and webhooks

- POST /api/v1/pipeline/run — start pipeline via HTTP
- GET /api/v1/pipeline/status — check status
- POST /api/v1/pipeline/stop — force stop
- Add webhooks route
- Update main.py with new routes" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 8. Pipeline scripts
# ---------------------------------------------------------
echo "📦 Commit 8: Pipeline scripts"
git add run_full_pipeline.py
git add run_scraper.py
git add process_existing_articles.py
# Remove the deleted fixed file if staged
git rm --cached run_full_pipeline_fixed.py 2>/dev/null || true
git commit -m "feat: update pipeline scripts

- run_full_pipeline: use ModelsConfig for ollama model resolution
- Remove hardcoded qwen default from check_llm_provider()
- Update run_scraper.py, process_existing_articles.py
- Remove obsolete run_full_pipeline_fixed.py" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 9. Scrapers
# ---------------------------------------------------------
echo "📦 Commit 9: Scrapers"
git add src/scrapers/habr/habr_scraper.py
git add src/scrapers/habr/scraper_service.py
git commit -m "fix: update Habr scraper

- Improve scraper reliability
- Update scraper service" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 10. n8n integration
# ---------------------------------------------------------
echo "📦 Commit 10: n8n integration"
git add src/infrastructure/n8n/
git add tests/test_n8n_integration.py
git commit -m "feat: add n8n workflow integration

- n8n infrastructure module
- Integration tests" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 11. Streamlit UI
# ---------------------------------------------------------
echo "📦 Commit 11: Streamlit UI"
git add streamlit_app/
git add streamlit_n8n_manager.py
git commit -m "feat: add Streamlit management UI

- Streamlit app for pipeline management
- n8n manager interface" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 12. Tests & examples
# ---------------------------------------------------------
echo "📦 Commit 12: Tests & examples"
git add tests/test_openrouter_system.py
git add examples/
git commit -m "test: add OpenRouter system tests and examples" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 13. README
# ---------------------------------------------------------
echo "📦 Commit 13: README"
git add README.md
git commit -m "docs: rewrite README with per-agent models, Docker setup, troubleshooting

- Add per-agent model routing documentation
- Add Docker quick start with .dockerignore
- Add GPU recommendations table
- Add Telegram/Telegraph setup
- Update architecture diagrams
- Simplify troubleshooting with real-world issues" || echo "  (nothing to commit)"

# ---------------------------------------------------------
# 14. Cache (если нужно)
# ---------------------------------------------------------
if [ -d "cache/" ]; then
    echo "⚠️  cache/ directory found — skipping (should be in .gitignore)"
fi

# ---------------------------------------------------------
# 15. Sensitive files check
# ---------------------------------------------------------
echo ""
echo "🔒 Проверяем чувствительные файлы..."
if git diff --cached --name-only | grep -qE "(\.env$|id_ed25519|id_rsa|\.pem$)"; then
    echo "⚠️  ВНИМАНИЕ: обнаружены чувствительные файлы в коммите!"
    echo "   Проверьте: git diff --cached --name-only | grep -E '\.env|ed25519|rsa|\.pem'"
    echo "   Отмените если нужно: git reset HEAD <file>"
    exit 1
fi

# Skip ollama_models sensitive file
git checkout -- ollama_models/id_ed25519.pub 2>/dev/null || true

# ---------------------------------------------------------
# Push
# ---------------------------------------------------------
echo ""
echo "📊 Итого коммитов:"
git log --oneline -15
echo ""

read -p "🚀 Push to origin/main? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    echo "✅ Pushed!"
else
    echo "⏸️  Не пушим. Коммиты сохранены локально."
    echo "   Для push: git push origin main"
fi