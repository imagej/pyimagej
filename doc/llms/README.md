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
- **Environment-specific rules:**
  - `env_colab.md` - Google Colab considerations
  - `env_interactive.md` - Desktop GUI environments
  - `env_headless.md` - Server/cluster environments
  - `env_script_editor.md` - Fiji Script Editor integration

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
    activities/pyimagej_advanced.md \
    rulesets/pyimagej_core.md \
    rulesets/env_headless.md > my_context.md
```

### Prompt Template
```
I'm learning PyImageJ for scientific image analysis. Please use this context to guide our conversation:

[PASTE PERSONA/RULESET CONTENT HERE]

I'm ready to start! What activities do you recommend based on my experience level?
```

## 🤝 Contributing

To extend this framework:

- **Add new personas** in `personas/activities/` for specialized domains
- **Create environment rules** in `rulesets/` for new platforms
- **Update the notebook** to fix bugs, improve the documentation, or add more features

## 📚 Learn More

- [ImageJ Community Forum](https://forum.image.sc/)
- [PyImageJ Documentation](https://pyimagej.readthedocs.io/)