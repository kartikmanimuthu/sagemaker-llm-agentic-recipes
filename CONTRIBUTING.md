# Contributing to SageMaker LLM Recipes

First off, thank you for considering contributing to this repository! It's people like you who make the open-source community an amazing place to learn, inspire, and create.

## How Can I Contribute?

### Reporting Bugs
*   Check the [Issues](https://github.com/your-username/sagemaker-llm-recipes/issues) to see if it has already been reported.
*   If not, open a new issue. Clearly describe the problem and include steps to reproduce it.

### Suggesting Enhancements
*   Open an issue with the "enhancement" label.
*   Provide a clear explanation of why this feature would be useful.

### Pull Requests
1.  Fork the repo and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  Ensure your code adheres to standard Python (PEP 8) style.
4.  Issue that PR!

## Style Guide
*   **Scripts**: Use `snake_case.py`.
*   **Notebooks**: Use `numbers_and_kebab-case.ipynb`.
*   **Commit Messages**: Use concise, descriptive messages (e.g., `feat: add gemma-2b deployment recipe`).

## Adding New Models
We love new model recipes! If you're adding a new model:
1.  Add a notebook in `notebooks/`.
2.  Add a corresponding deployment script in `scripts/` (if applicable).
3.  Update the **Model Support Matrix** in `README.md`.
