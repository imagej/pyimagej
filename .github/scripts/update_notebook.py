#!/usr/bin/env python3
"""
Script to automatically update PyImageJ AI Guide notebook cells based on
personas and rulesets files.
"""

import os
import re
from pathlib import Path
from typing import Dict

import jinja2
import nbformat


def scan_persona_files(personas_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Scan personas directory and build category mappings.
    Expected format: activities/{category}_{level}.md
    """
    activities_dir = personas_dir / "activities"
    category_mappings = {}

    if not activities_dir.exists():
        return category_mappings

    # Scan all activity files and group by category
    for file_path in sorted(activities_dir.glob("*.md")):
        stem = file_path.stem
        if "_" in stem:
            category, level = stem.rsplit("_", 1)
            if category not in category_mappings:
                category_mappings[category] = {}
            category_mappings[category][level] = file_path.name

    return category_mappings


def scan_ruleset_files(rulesets_dir: Path) -> Dict[str, str]:
    """
    Scan rulesets directory for environment files.
    Expected format: environments/env_{environment}.md
    """
    env_dir = rulesets_dir / "environments"
    environment_mapping = {}

    if not env_dir.exists():
        return environment_mapping

    # Scan environment files
    for file_path in sorted(env_dir.glob("env_*.md")):
        env_name = file_path.stem
        if env_name.startswith("env_"):
            # Convert env_colab -> Google Colab, env_scripting -> Fiji Script Editor, etc.
            display_name = format_environment_name(env_name[4:])  # Remove "env_" prefix
            # Store the full relative path from rulesets directory
            relative_path = f"environments/{env_name}"
            environment_mapping[display_name] = relative_path

    return environment_mapping


def format_environment_name(env_key: str) -> str:
    """Convert environment key to display name."""
    mapping = {
        "colab": "Google Colab",
        "interactive": "Interactive Desktop",
        "headless": "True Headless",
        "scripteditor": "Fiji Script Editor"
    }
    return mapping.get(env_key, env_key.replace("_", " ").title())


def build_persona_template_data(category_mappings: Dict[str, Dict[str, str]]) -> Dict:
    """Build template data for persona cell."""

    # Create categories with their options
    categories = {}
    experience_levels = {}

    level_mapping = {
        "beginner": "beginner",
        "intermediate": "intermediate",
        "advanced": "advanced"
    }

    # Build categories and experience level mappings in desired order
    # Define preferred order for known categories, but include any new ones automatically
    preferred_order = ["colab", "coding", "pyimagej"]

    # Start with preferred categories in order, then add any new ones alphabetically
    ordered_categories = []
    for category in preferred_order:
        if category in category_mappings:
            ordered_categories.append(category)

    # Add any categories not in preferred_order (future categories)
    remaining_categories = sorted([cat for cat in category_mappings.keys() if cat not in preferred_order])
    ordered_categories.extend(remaining_categories)


def order_environments(environment_mapping: Dict[str, str]) -> list:
    """Return environment names in preferred order: Colab > Headless > Desktop > Script Editor."""
    # Define preferred order based on user requirements
    preferred_order = [
        "Google Colab",
        "True Headless",
        "Interactive Desktop",
        "Fiji Script Editor"
    ]

    # Start with preferred environments in order
    ordered_environments = []
    for env in preferred_order:
        if env in environment_mapping:
            ordered_environments.append(env)

    # Add any environments not in preferred_order (future environments)
    remaining_environments = sorted([env for env in environment_mapping.keys() if env not in preferred_order])
    ordered_environments.extend(remaining_environments)

    return ordered_environments


def build_persona_template_data(category_mappings: Dict[str, Dict[str, str]]) -> Dict:
    """Build template data for persona cell."""

    # Create categories with their options
    categories = {}
    experience_levels = {}

    level_mapping = {
        "beginner": "beginner",
        "intermediate": "intermediate",
        "advanced": "advanced"
    }

    # Build categories and experience level mappings in desired order
    # Define preferred order for known categories, but include any new ones automatically
    preferred_order = ["colab", "coding", "pyimagej"]

    # Start with preferred categories in order, then add any new ones alphabetically
    ordered_categories = []
    for category in preferred_order:
        if category in category_mappings:
            ordered_categories.append(category)

    # Add any categories not in preferred_order (future categories)
    remaining_categories = sorted([cat for cat in category_mappings.keys() if cat not in preferred_order])
    ordered_categories.extend(remaining_categories)

    for category in ordered_categories:
        levels = category_mappings[category]
        category_key = f"{category}"

        # Create display options for this category in specific order
        level_order = ["beginner", "intermediate", "advanced"]  # Define desired order
        options = []
        for level in level_order:  # Use specific order instead of sorted()
            if level in levels:
                display_name = f"{'New to' if level == 'beginner' else 'Some' if level == 'intermediate' else category.title() + ' expert'} {category.replace('_', ' ')}"
                if category == "coding":
                    display_name = f"{'New to programming' if level == 'beginner' else 'Some programming experience' if level == 'intermediate' else 'Advanced programmer'}"
                elif category == "pyimagej":
                    display_name = f"{'New to PyImageJ' if level == 'beginner' else 'Some PyImageJ experience' if level == 'intermediate' else 'PyImageJ expert'}"
                elif category == "colab":
                    display_name = f"{'New to Google Colab' if level == 'beginner' else 'Some Colab experience' if level == 'intermediate' else 'Colab expert'}"

                options.append(display_name)
                experience_levels[display_name] = level_mapping[level]

        if options:
            categories[category_key] = options

    return {
        "categories": categories,
        "experience_levels": experience_levels,
        "category_mappings": category_mappings
    }


def update_notebook_cell(notebook: nbformat.NotebookNode, cell_id: str, new_content: str) -> bool:
    """Update a specific cell in the notebook by ID or title pattern."""
    # First try to find by ID (for GitHub Actions)
    for cell in notebook.cells:
        if cell.get("id") == cell_id:
            cell["source"] = new_content
            return True

    # Fall back to finding by title pattern (for local testing)
    title_patterns = {
        "#VSC-672bc454": "🤖 Personalize Gemini",
        "#VSC-382943dc": "⚖️ Set Coding Rules",
        "#VSC-b7af7e7c": "📦 Download PyImageJ Source",
        "#VSC-8ce7bebd": "colab.research.google.com"  # Colab badge cell
    }

    if cell_id in title_patterns:
        pattern = title_patterns[cell_id]
        for cell in notebook.cells:
            if pattern in cell.source:
                cell["source"] = new_content
                return True

    return False

def update_colab_badge_cell(notebook: nbformat.NotebookNode, branch_name: str, notebook_filename: str) -> bool:
    """Update the Colab badge cell to point to the correct branch and filename."""
    badge_cell_id = "#VSC-8ce7bebd"  # The badge cell ID from the notebook

    # First try to find by ID (for GitHub Actions)
    for cell in notebook.cells:
        if cell.get("id") == badge_cell_id:
            source = cell["source"]
            # Update the Colab badge URL to use the correct branch and filename
            # Pattern matches the full Colab URL structure
            updated_source = re.sub(
                r'https://colab\.research\.google\.com/github/imagej/pyimagej/blob/[^/]+/doc/llms/[^"]+',
                f'https://colab.research.google.com/github/imagej/pyimagej/blob/{branch_name}/doc/llms/{notebook_filename}',
                source
            )
            cell["source"] = updated_source
            return True

    # Fall back to finding by content pattern (for local testing)
    for cell in notebook.cells:
        if "colab.research.google.com" in cell.source:
            source = cell["source"]
            updated_source = re.sub(
                r'https://colab\.research\.google\.com/github/imagej/pyimagej/blob/[^/]+/doc/llms/[^"]+',
                f'https://colab.research.google.com/github/imagej/pyimagej/blob/{branch_name}/doc/llms/{notebook_filename}',
                source
            )
            cell["source"] = updated_source
            return True
    return False

def update_download_cell(notebook: nbformat.NotebookNode, commit_sha: str, branch_name: str) -> bool:
    """Update the download cell to checkout the specific commit."""
    download_cell_id = "#VSC-b7af7e7c"  # The download cell ID from the notebook

    # First try to find by ID (for GitHub Actions)
    for cell in notebook.cells:
        if cell.get("id") == download_cell_id:
            source = cell["source"]
            # Replace any existing git checkout with the specific commit SHA
            updated_source = re.sub(
                r'!cd /content/pyimagej && git checkout \S+',
                f'!cd /content/pyimagej && git checkout {commit_sha}',
                source
            )
            cell["source"] = updated_source
            return True

    # Fall back to finding by content pattern (for local testing)
    for cell in notebook.cells:
        if "📦 Download PyImageJ Source" in cell.source:
            source = cell["source"]
            updated_source = re.sub(
                r'!cd /content/pyimagej && git checkout \S+',
                f'!cd /content/pyimagej && git checkout {commit_sha}',
                source
            )
            cell["source"] = updated_source
            return True
    return False

def main():
    """Main script execution."""

    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    notebook_path = repo_root / "doc" / "llms" / "custom-pyimagej-ai-guide.ipynb"
    personas_dir = repo_root / "doc" / "llms" / "personas"
    rulesets_dir = repo_root / "doc" / "llms" / "rulesets"
    templates_dir = script_dir.parent / "templates"  # .github/templates, not .github/scripts/templates

    # Get commit SHA and branch name from environment
    commit_sha = os.environ.get("COMMIT_SHA", "main")
    branch_name = os.environ.get("BRANCH_NAME", "main")

    # Determine commit message prefix based on branch
    if branch_name == "main":
        commit_prefix = "Auto-update"
    else:
        commit_prefix = "WIP: Auto-update"

    print(f"Updating notebook: {notebook_path}")
    print(f"Using commit SHA: {commit_sha}")
    print(f"From branch: {branch_name}")
    print(f"Commit prefix: {commit_prefix}")

    # Check if notebook file exists
    if not notebook_path.exists():
        print(f"❌ Notebook file not found: {notebook_path}")
        print("This might be expected if the notebook doesn't exist on this branch yet.")
        print("Exiting gracefully - no updates needed.")
        return

    # Load notebook
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"❌ Failed to read notebook file: {e}")
        print("Exiting gracefully - cannot process invalid notebook.")
        return

    # Setup Jinja environment
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )

    # Scan and update persona cell
    category_mappings = scan_persona_files(personas_dir)
    if category_mappings:
        print(f"Found persona categories: {list(category_mappings.keys())}")

        template_data = build_persona_template_data(category_mappings)
        template = jinja_env.get_template("personalize_gemini_cell.py.j2")
        new_content = template.render(**template_data)

        # Update the persona cell
        persona_cell_id = "#VSC-672bc454"
        if update_notebook_cell(notebook, persona_cell_id, new_content):
            print("✅ Updated Personalize Gemini cell")
        else:
            print("❌ Failed to find Personalize Gemini cell")

    # Scan and update ruleset cell
    environment_mapping = scan_ruleset_files(rulesets_dir)
    if environment_mapping:
        print(f"Found environments: {list(environment_mapping.keys())}")

        template_data = {
            "environments": order_environments(environment_mapping),
            "environment_mapping": environment_mapping
        }
        template = jinja_env.get_template("set_coding_rules_cell.py.j2")
        new_content = template.render(**template_data)

        # Update the rules cell
        rules_cell_id = "#VSC-382943dc"
        if update_notebook_cell(notebook, rules_cell_id, new_content):
            print("✅ Updated Set Coding Rules cell")
        else:
            print("❌ Failed to find Set Coding Rules cell")

    # Update download cell with commit SHA
    if update_download_cell(notebook, commit_sha, branch_name):
        print(f"✅ Updated Download cell to use commit {commit_sha}")
    else:
        print("❌ Failed to find Download cell")

    # Update Colab badge cell with correct branch and filename
    notebook_filename = notebook_path.name
    if update_colab_badge_cell(notebook, branch_name, notebook_filename):
        print(f"✅ Updated Colab badge to use branch {branch_name}")
    else:
        print("❌ Failed to find Colab badge cell")

    # Save updated notebook
    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f, version=4)
        print("✅ Notebook update complete!")
    except Exception as e:
        print(f"❌ Failed to save notebook: {e}")
        return


if __name__ == "__main__":
    main()
