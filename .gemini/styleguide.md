# Gemini Code Assist PR Review Styleguide
**Repository:** `prostate-cancer` (RationAI)
**Context:** This is a research-focused machine learning repository dedicated to predicting carcinoma in prostate WSIs in digital pathology. 

## 🎯 Primary Review Focus
- **Ignore formatting and linting:** We use `ruff` for formatting/linting and `uv` for package management. Assume CI/CD will catch styling issues. Do not comment on line length, quotes, or basic PEP-8 formatting.
- **Focus on ML logic and correctness:** Look for off-by-one errors in array slicing, incorrect tensor device allocations, tensor shape mismatches, and data leakage between train/val/test splits.
- **Research context over production validation:** This is research code. Do not suggest adding heavy input validation, complex exception handling, or enterprise-grade defensive programming unless the current logic will explicitly crash the pipeline. Prioritize readability.
- **Testing:** Do not block PRs or aggressively request unit tests. We do not enforce strict unit testing for this repository.

## 📝 General Comment Style
- Keep comments **short and actionable**.
- Prefer **bullet points** over long paragraphs.
- Point to specific lines or sections when possible.
- Suggest improvements, not rewrite entire snippets.
- Avoid repetition of what the code already clearly states.
- Defer to the repo’s existing conventions unless there’s a clear bug or inconsistency.

## 🔬 Domain-Specific Guidance (RationAI & ratiopath)
- **Use `ratiopath`:** This project relies on our library `ratiopath`. 
  - If you see custom tiling logic, suggest using `ratiopath.tiling`.
  - If you see custom annotation parsing (ASAP/GeoJSON), suggest using `ratiopath.parsers`.
  - Check if Ray-based distributed processing in `ratiopath` is being used efficiently for large-scale WSI tasks.
- **WSI Handling:** Verify that `openslide` or `ratiopath` calls use the correct downsample levels and that tile offsets are calculated correctly.

## 🏗️ Architecture & Reproducibility 
- **Hydra Configs (`configs/`):** Reproducibility is paramount. If a PR introduces a new module, model architecture, or preprocessing step, check if the author has updated or created the corresponding YAML configuration. Remind them if it seems missing.
- **Experiment Tracking (MLflow):** When PRs add new loss functions, evaluation metrics, or training loops in `project_name/` (or `ml/`), ensure that these new metrics are properly logged to MLflow.
- **Repository Structure:**
  - `preprocessing/`: Ensure data transformations (tiling, QC, tissue masks) are logically sound.
  - `project_name/` (future `ml/`): Focus on training loops, PyTorch Lightning modules, and model definitions.
  - `postprocessing/`: Focus on ensembling and final prediction logic.
  - `scripts/`: These are job submission templates. Do not review them as strictly as core Python code.

## 📚 Types & Documentation
- **Type Hinting:** We use strict `mypy`, but it is *not required* for PRs to be merged. Gently suggest type hints for complex function signatures, but do not nitpick missing `Any` types or incomplete typing.
- **Docstrings:** Docstrings are not strictly required everywhere. However, if a docstring *is* provided, ensure it generally follows the **Google Docstring Style**.

## 💻 Libraries & Best Practices
- **PyTorch & PyTorch Lightning:** Suggest idiomatic PyTorch Lightning constructs (e.g., using `self.log` correctly). Watch out for detached tensors or memory leaks in custom training steps.
- **Data Processing (NumPy, Pandas, OpenSlide):** Suggest vectorized operations over `for` loops where applicable for performance. Ensure OpenSlide WSI coordinate extractions are logical (e.g., matching the correct level/downsample).