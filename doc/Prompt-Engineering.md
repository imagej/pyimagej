# Prompt Engineering

**Prompt engineering** is the act of adjusting the text you send (your prompt) to an Large Language Model (LLM) to give the AI the best chance at producing output that is useful to you. This is a fundamental skill independent of the specific LLM you are interacting with (Gemini, ChatGPT, Claude, etc...). The goal of this page is to provide you with some basic guidelines to consider when crafting your prompts.

**Remember**: Expect any interaction with a LLM to be an **iterative back-and-forth**. It is normal not to get the right answer at first! A fundamental step in prompt engineering is simply *"try again"*. If an LLM's output isn't useful to you, that is a good indicator to *take a critical look at your own text*.

## Getting the Most from AI Assistants

**Be Specific, Include as Much Context as Possible**
- ❌ "This doesn't work"
- ✅ "I'm getting a 'NameError: name 'dataset' is not defined' when trying to run ij.py.show(dataset). Here's my code: [paste code]"

**Avoid Bias Leading Towards a Particular Answer**
- ❌ "Why is Colab better than other Jupyter notebook environments?"
- ✅ "Give me an unbiased assessment of the current options for running Jupyter notebooks in a table with pros and cons"

**Ask for Explanations, Not Just Code**
- ❌ "Write code to filter an image"
- ✅ "Explain the difference between Gaussian blur and median filtering, explain the options for both in PyImageJ and their pros & cons"

**Request Step-by-Step Breakdowns**
- ❌ "How do I segment cells?"
- ✅ "What steps are needed for segmenting the cells in a DAPI-stained image?"
- ✅ "Explain this process of cell segmentation, creating well-documented cells using PyImageJ for each step: preprocessing → thresholding → watershed"

**Ask Gemini to Defend its Choices**
- ❌ "Thanks for this notebook, now I'm going to publish it!"
- ✅ "Why did you convert these images to numpy for processing? Could it be done in PyImageJ directly?"

## Understanding Hallucinations

**Hallucinations** are when an LLM generates false outputs. Fundamentally, this underscores a core principle of *all* LLMs:

**⚠️LLMs are *Statistical Constructs* that DO NOT understand meaning⚠️**

It is important to understand that LLMs:
- Work by generating the "most likely" responses, based on their context and training data
- Are tuned to be sensitive to *your words*: they can pick up on subtle bias and reflect that back to you
- Tend to be overly positive, optimistic, and self-confident

Common areas you may run into hallucinations include:
- Specific code usage: methods or packages that don't exist
- Recent facts: LLM's knowledge is frozen at their time of training and rely on internal web searches for the latest information
- Obscure facts: particulars from publications, legal documents, etc...
- In references themselves: made-up names, URLs, paper titles

In Google Colab we have also noticed hallucinations in terms of understanding incorrect but not *failing* code cell output. For example, imagine a cell that segments an image and reports the number of objects found: Gemini may hallucinate a particular number of cells ("You found 67 cells. Segmentation works!") when in reality segmentation failed and only one or zero cells were found. This may be a sign that the Gemini chatbot has *access* to all cell outputs, but doesn't necessarily *understand* all formats of output.

General approaches for handling hallucinations include:
- Be skeptical of verbatim LLM output. Ask for sources and verify they exist (e.g. using your own searching, or other LLMs).
- Rely on your prompt engineering skills. Try again, be specific, and tell the LLM what it got wrong and what you expected.

## More Conversation Starters

Gemini is your tutor in [the AI tutorial notebook](AI-Tutorial-Notebook.md): the hope is to have a natural conversation about your goals, so that Gemini can guide you to success.

Here are some ideas for more useful interactions:

**Real-world Use:**
> I work with [microscopy type] images of [sample type] - what image processing skills would be most useful to me?

> I currently use [other software] for [task] - how would I do this in PyImageJ?

**Focus on Specific Areas:**
> What coding fundamentals should I practice for scientific computing?

> What Colab features can help with my research?

**Objective Navigation:**
> Show me a learning pathway for cell segmentation

> What should I try next based on my current skill level?

**Context Switching:**
> I want to focus on real research applications

> What can I practice that will help with reproducible workflows?

**Best Practices:**
> Do you have any thoughts on my approach to [previous activity]

> What common mistakes should I watch out for?

**Theory & Understanding:**
> Explain the theory behind [concept I just practiced]

> Break down [complex topic] into smaller steps

**Environment-Specific:**
> What should I know about running this locally instead of in Colab?

> How would this work in a headless environment?
