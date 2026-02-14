## Abstract
Reference-based metrics for Machine Translation (MT) offer limited semantic coverage and interpretability, especially in biomedical settings. We study AskQE, a question-answering (QA) formulation of MT evaluation, and assess whether it remains effective in reproducible, low-cost configurations based on open-source LLMs.
We replicate the AskQE pipeline replacing proprietary models with three compact instruction-tuned open-source LLMs, Meta-Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, and Gemma-2-9B-IT, runnable on consumer hardware. We evaluate on two benchmarks: ContraTICO, a controlled synthetic test suite with eight perturbation types at two severity levels, and BioMQM, a real-world biomedical MT dataset annotated under the MQM framework.

Beyond single-model baselines, we propose The Champions Trio, a Multi-LLM ensemble that selects consensus answers via semantic centroid aggregation, stabilizing segment-level evaluation signals. The ensemble achieves a decision accuracy of 67.1\% on ContraTICO (+15.0 pp over Qwen) and 73.4\% on BioMQM, outperforming the 63.77\% reported by the original SBERT-based AskQE approach.
To move beyond scalar scores, we introduce an LLM-as-a-Judge module (Qwen2.5-7B) that assigns structured error categories from a predefined taxonomy and produces a natural-language explanation for each translation, making evaluation transparent and actionable. The judge is validated on ContraTICO before application to biomedical data.

Overall, our study demonstrates that AskQE-style evaluation can be made cheaper, open, and more interpretable while retaining competitive performance against human judgments.

<p align="center">
  <img src="pipeline.png" alt="Pipeline overview" width="600">
</p>

_Figure: Overview of the proposed pipeline_

## Repo Structure
The repository is organized as:
```
DNLP-askQE/
    ├── Answers/                                # Model-generated answers for each dataset
        └── results_gemma_biomqm.jsonl
        └── results_gemma_contra.jsonl
        └── results_llama3_biomqm.jsonl
        └── results_llama3_contra.jsonl
        └── results_qwen_biomqm.jsonl
        └── results_qwen_contra.jsonl
    ├── Plot/                                   # evaluation plots and visualizations
        └── biomqm_askqe_by_perturbation.png
        └── confidence_bt_biomqm.png
        └── confidence_bt_contra.png
        └── contra_askqe_by_perturbation.png
    ├── AskQE_finalVersion.ipynb                # notebook
    ├── DNLP_paper                              # paper
    ├── readme.md
```

## Get ready
**Prerequisites**
- GPU to run the 3 models that we will use
- Hugging face account to access:
    - `meta-llama/Llama-3.1-8B-Instruct`
    - `Qwen/Qwen2.5-7B-Instruct`
    - `google/gemma-2-9b-it`
- Main libraries:
  - `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `tqdm`, `matplotlib`, `seaborn`

**Setup libraries**
```
!pip install -q bitsandbytes accelerate peft sentence-transformers jsonlines datasets einops
```

**Option 1 - Full pipeline**\
Running the full pipeline takes about 8 hours, mainly due to sections 2.5 2.6 2.7, where the three LLMs answer at all questions for both datasets.
- **Section 1 – Environment Setup & Data Loading** (~7 min): Set up the environment (Colab/Drive), clone the AskQE repository, and load/serialize ContraTICO and BioMQM in .pkl format (a subset of about 500 rows per dataset is used)

- **Section 2 – Question Answering for Each Model** (~7 h): Define the prompt, the ModelEngine class, and the inference loop to make the 3 LLMs answer all questions (both source and back‑translation).

- **Section 3 – Extension 1: LLM Ensemble** (~5 min): Combine the responses from the 3 LLMs and, for each question, select the most semantically consistent “centroid” answer. Finally, compute the AskQE score and detect potential hallucinations.

- **Section 4 – Results** (~5 min): Comprehensive evaluation of the AskQE metric on both datasets:
  - **ContraTICO Analysis**: Computes mean AskQE scores per perturbation type and achieves decision accuracy (Accept/Reject) using a Gaussian Mixture Model (GMM) classifier.
  - **BioMQM Alignment**: Merges model predictions with human ratings, computing Kendall's Tau correlation, decision accuracy, disagreement analysis across error severity levels, and mean scores per severity category.
  - **Overall Assessment**: Validates that AskQE scores correlate with human perception and correctly identify translation errors, demonstrating the metric's effectiveness in automatic MT evaluation.

- **Section 5 – Plots** (~2 min): Three visualizations: (a) back‑translation confidence boxplot per model, (b) AskQE score boxplot per perturbation_type and model, (c) rate of possible hallucination (table).

- **Section 6 – Extension 2: Error Categorization** (~35 min): The LLM‑as‑a‑Judge (Qwen2.5) assigns each translation to one of the ContraTICO perturbation_type labels by analyzing the source, back‑translation, and QA discrepancies, then compares the predicted labels with the true ones.

**Option 2 - Only reproduce extensions**\
This second option is designed to be lightweight: it skips the expensive QA inference on GPUs, use the pre‑computed model answers, and only runs the ensemble, analysis, and error‑categorization steps, taking around 40 minutes in total.

- Use the answer files provided in the `Answers/` folder, which contain the responses generated by each model for each dataset
- Run sections 1, 2.1, 2.2 and 2.4 to set the enviroment
- Run sections 3, 4, 5 and 6 to compute the results and execute the two extensions

## Team
Team Number 17
- Andrea Cauda s343386 - s343386@studenti.polito.it
- Roberto Cozzone s336155 - s336155@studenti.polito.it
- Pietro Giancristofaro s341870 - s341870@studenti.polito.it
- Davide Tonetti s334297 - s334297@studenti.polito.it
- Antonio Visciglia s346837 - s346837@studenti.polito.it
<hr style="height: 3px; border: 0; background-color: #808080; margin-top: 40px;">

**Research Paper:** This implementation is described in detail in:\
_"ASKQE: Question Answering as Automatic Evaluation for Machine Translation"_\
Cauda A., Cozzone R., Giancristofaro P., Tonetti D., Visciglia A. (2025)