#!/usr/bin/env python3
"""
Показать текущую конфигурацию моделей.
"""

import sys
sys.path.insert(0, '.')

from src.config.models_config import ModelsConfig


def main():
    """Показать конфиг."""
    config = ModelsConfig()
    config.print_config()
    
    print("💡 Чтобы изменить профиль:")
    print("   1. Отредактируйте config/models.yaml")
    print("   2. Измените active_profile на:")
    print("      - balanced (12-16 GB RAM)")
    print("      - high_quality (40+ GB RAM, GPT-OSS 20B)")
    print("      - low_ram (8 GB RAM)")
    print("      - fastest (минимальное время)")
    print("   3. Перезапустите: scripts/ops/docker_restart_quick.sh")
    print()


if __name__ == "__main__":
    main()
