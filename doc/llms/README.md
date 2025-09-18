# PyImageJ LLM Support

This directory contains a complete framework for learning PyImageJ with Large Language Models (LLMs) like Gemini, ChatGPT, and Claude. The system uses modular persona and ruleset files to customize AI responses for different experience levels and environments.

## 🚀 Quick Start with Google Colab

The easiest way to get started is with our interactive Colab notebook:

**➡️ [Open PyImageJ AI Guide in Colab](pyimagej-ai-guide.ipynb)**

This notebook will:
1. 📦 Download PyImageJ source and AI templates
2. 🤖 Configure Gemini based on your experience level
3. ⚖️ Set coding rules for your target environment
4. 🏝️ Set up a complete Fiji bundle for analysis
5. 🚀 Guide you through personalized learning activities

The setup takes ~2-3 minutes on first run, then you're ready to learn with AI assistance!

### Secondary Colab Notebook Uses

* As this notebook contains robust PyImageJ initialization, coupled with extensive PyImageJ LLM rulesets, it can be used as a starting point for notebook development even by advanced developers

## 📁 Framework Components

### `pyimagej-ai-guide.ipynb`
Interactive Colab notebook that orchestrates the entire learning experience. Features:
- Experience-based persona selection (beginner/intermediate/advanced)
- Environment-specific coding rules (Colab/Desktop/Headless/Script Editor)
- Notebook-safe Fiji setup and PyImageJ integration
- Easy transfer of persona text to other LLMs

### `personas/` Directory
Modular AI personality guides which can be combined and used to establish tailored persona and training roles.

- **`base_persona.md`** - Core PyImageJ teaching approach
- **Activity Files** - Experience-specific learning paths:
  - `activities/coding_*.md` - Core programming concepts
  - `activities/pyimagej_*.md` - Image procesing with PyImageJ
  - `activities/colab_*.md` - Useful Google Colab skills

### `rulesets/` Directory
Technical guidelines for how AI should write PyImageJ code. Select the ruleset appropriate for the environment you will be running your code.

- **`pyimagej_core.md`** - Universal PyImageJ best practices
- **Environment-specific rules** in `environments/` subdirectory:
  - `environments/env_colab.md` - Google Colab considerations
  - `environments/env_interactive.md` - Desktop GUI environments
  - `environments/env_headless.md` - Server/cluster environments
  - `environments/env_script_editor.md` - Fiji Script Editor integration

## 🔧 Using with Other LLMs

While designed for Gemini in Colab, you can use these files with any LLM:

### With ChatGPT, Claude, etc.

1. **Run the Colab notebook** to generate your personalized content
2. **Copy the output** using the provided copy buttons in the persona and rules cells
3. **Paste into your preferred LLM** as context

Or manually combine files:

```bash
# Example: Advanced PyImageJ user in headless environment
cat personas/base_persona.md \
    personas/activities/pyimagej_advanced.md \
    rulesets/pyimagej_core.md \
    rulesets/environments/env_headless.md > my_context.md
```

### Prompt Template
```
I'm learning PyImageJ for scientific image analysis. Please use this context to guide our conversation:

[PASTE PERSONA/RULESET CONTENT HERE]

I'm ready to start! What activities do you recommend based on my experience level?
```

## 🤝 Contributing

To extend this framework:

- **Add new personas** in `personas/activities/` using the naming pattern `{category}_{level}.md` (e.g., `statistics_beginner.md`)
- **Create environment rules** in `rulesets/environments/` using the pattern `env_{environment}.md` (e.g., `env_docker.md`)
- **Update the notebook** to fix bugs, improve the documentation, or add more features

### 🔄 Automated Updates

The framework includes GitHub Actions automation to keep the notebook synchronized with persona and ruleset files:

**How it works:**
1. **Auto-detection**: GitHub Action triggers on commits to `main` that modify `personas/` or `rulesets/` files
2. **Smart updates**: Uses Jinja templates to regenerate the "Personalize Gemini" and "Set Coding Rules" cells
3. **Commit tracking**: Updates the download cell to use the exact commit SHA that triggered the update
4. **Zero maintenance**: Persona dropdowns and environment options automatically reflect new files
5. **Branch safety**: Builds on development branches must be triggered manually. The commit is generated as a `WIP`, which is blocked from merging by PR

**File naming requirements:**
- **Personas**: `personas/activities/{category}_{level}.md` where level is `beginner`, `intermediate`, or `advanced`
  - You can create just one level (e.g., only `statistics_beginner.md`) or any combination
  - Missing levels are automatically skipped - no need to create all three
- **Rulesets**: `rulesets/environments/env_{environment}.md` for new target environments
- **Ordering**: Categories appear in order: `colab` → `coding` → `pyimagej` → (new categories alphabetically)

**To add new categories:**
1. Create activity files: `newcategory_beginner.md` (and optionally `_intermediate.md`, `_advanced.md`)
2. Commit to trigger the automation
3. The notebook will automatically include the new category with only the levels you created

**Template locations:**
- Automation scripts: `.github/workflows/update-notebook.yml`
- Jinja templates: `.github/templates/`
- Update logic: `.github/scripts/update_notebook.py`

### 🌿 Branch Development Workflow

**Automatic updates only run on `main`** to keep the main branch history clean. For development:

**Working on branches:**
1. Edit persona/ruleset files on your feature branch
2. Manually test automation by going to **Actions** → **Update PyImageJ AI Guide Notebook** → **Run workflow** → select your branch
3. This creates `WIP: Auto-update...` commits on your branch (clearly marked as temporary)
4. Continue development normally

**Merging to main:**
1. Before creating a PR, ensure no `WIP:` commits will be merged:
   - Option A: Rebase/squash to remove WIP commits  
   - Option B: Manually change commit messages to remove `WIP:` prefix
2. Create PR to `main` - the **Check for WIP commits** workflow will verify no temporary commits remain
3. After merge, automation runs automatically and creates a clean `Auto-update...` commit on `main`

**Emergency options:**
- Add `[skip-automation]` to any commit message to prevent automation
- Use `workflow_dispatch` with "force update" to regenerate the notebook anytime

## 📚 Learn More

- [ImageJ Community Forum](https://forum.image.sc/)
- [PyImageJ Documentation](https://pyimagej.readthedocs.io/)