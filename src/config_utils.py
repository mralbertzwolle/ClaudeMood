#!/usr/bin/env python3
"""
Configuration utilities for ClaudeMood
"""
import json
import subprocess
from pathlib import Path


def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("📋 Creating default config from template...")

        template_path = config_path.parent / "config.json.example"
        if template_path.exists():
            import shutil
            shutil.copy(template_path, config_path)
            print(f"✅ Config created: {config_path}")
        else:
            raise FileNotFoundError(f"Config template not found: {template_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Expand ~ in paths
    for key in ['conversations_dir', 'data_dir', 'export_dir', 'model_cache_dir']:
        if key in config:
            config[key] = Path(config[key]).expanduser()

    return config


def save_config(config):
    """Save configuration to config.json"""
    config_path = Path(__file__).parent.parent / "config.json"

    # Convert Path objects back to strings
    config_dict = config.copy()
    for key in ['conversations_dir', 'data_dir', 'export_dir', 'model_cache_dir']:
        if key in config_dict and isinstance(config_dict[key], Path):
            config_dict[key] = str(config_dict[key])

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"💾 Config saved to: {config_path}")


def find_conversations_directories():
    """Search for Claude conversations directories (legacy sync version)"""
    print("🔍 Searching for conversations directories...")

    search_paths = [
        Path.home() / "Library" / "Application Support" / "Claude",
        Path.home() / "Library" / "Application Support" / "Claude Code",
        Path.home() / ".config" / "Claude",
        Path.home() / ".config" / "Claude Code",
    ]

    found_dirs = []

    # Quick search in common locations
    for base_path in search_paths:
        if base_path.exists():
            conversations_dir = base_path / "conversations"
            if conversations_dir.exists() and conversations_dir.is_dir():
                # Check if it has JSON files
                json_files = list(conversations_dir.glob("*.json"))
                if json_files:
                    found_dirs.append(conversations_dir)
                    print(f"✅ Found: {conversations_dir} ({len(json_files)} conversations)")

    # If not found, do deeper search (slower)
    if not found_dirs:
        print("🔍 Doing deep search (may take 10-20 seconds)...")
        try:
            result = subprocess.run(
                ['find', str(Path.home()), '-name', 'conversations', '-type', 'd', '-maxdepth', '5'],
                capture_output=True,
                text=True,
                timeout=30
            )

            for line in result.stdout.strip().split('\n'):
                if line:
                    conv_dir = Path(line)
                    if conv_dir.exists():
                        json_files = list(conv_dir.glob("*.json"))
                        if json_files and len(json_files) > 0:
                            found_dirs.append(conv_dir)
                            print(f"✅ Found: {conv_dir} ({len(json_files)} conversations)")
        except subprocess.TimeoutExpired:
            print("⚠️  Search timeout after 30 seconds")
        except Exception as e:
            print(f"⚠️  Search error: {e}")

    return found_dirs
