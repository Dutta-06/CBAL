# CBAL: Context-Based Agentic Learning for Speaker Diarization Segmentation Refinement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: ACL SRW 2026](https://img.shields.io/badge/Paper-ACL_SRW_2026-blue.svg)](#) <!-- Add paper link here when available -->

**CBAL** is a training-free, post-processing framework designed to refine segmentation boundaries in diarized transcripts (such as those from Pyannote or NeMo). It repairs common structural errors like false splits, acoustic confusion, and short turn ambiguities by combining acoustic heuristics, LLM-based linguistic reasoning, and strict signal-level constraints to guarantee no transcript degradation (0% cpWER degradation).

## 🚀 Features
- **Training-Free**: Operates on standard RTTM outputs; no model retraining required.
- **Interpretable**: Uses a lightweight LLM (`Gemma-3-4b`) to provide explicit reasoning for every structural change.
- **Safe Execution**: Employs a strict `GapClear` constraint based on word-level ASR timestamps to prevent hallucinated merges across genuine speaker transitions.
- **High Accuracy**: Achieves 93.4% fix accuracy on the AMI corpus.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dutta-06/CBAL.git
   cd CBAL
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## ⚙️ Configuration

The primary configuration is managed in `configs/base_config.yaml`.
- Set your WavLM model (`microsoft/wavlm-base-plus-sv`).
- Set your LLM backend (`google/gemma-3-4b-it`).
- Adjust hyperparameter thresholds for acoustic similarity and maximum gap durations as needed.

## 🏃 Usage

To run the full CBAL pipeline (scanning, reasoning, and validation) on a specific meeting:

```bash
python run_cbal_full.py --meeting_id ES2004a
```

To run on the full batch defined in the script:
```bash
python run_cbal_full.py
```

The pipeline will:
1. Load the baseline RTTM and audio from `results/` and `data/audio/`.
2. Extract WavLM embeddings to identify conflict candidates.
3. Prompt Gemma to reason over the transcripts.
4. Validate merges against Whisper word-timestamps.
5. Save the refined diarization to `results/fixed_{meeting_id}.rttm`.
6. Save detailed decision logs to `results/logs/decisions_{meeting_id}.log`.

## 📊 Evaluation

The repository includes scripts to evaluate the safety and utility of the refined segmentations:
- `evaluate_fix.py`: Calculates Fix Accuracy against ground truth.
- `evaluate_cpwer.py`: Verifies perfect transcript integrity.
- `evaluate_der.py`: Calculates Diarization Error Rate (DER).

## 📜 Citation

If you use CBAL in your research, please cite our ACL SRW 2026 paper:

```bibtex
@inproceedings{dutta-vishwakarma-2026-cbal,
    title = "{CBAL}: Context-Based Agentic Learning for Speaker Diarization Segmentation Refinement",
    author = "Dutta, Odwitiyo and Vishwakarma, Dinesh Kumar",
    booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics: Student Research Workshop (ACL SRW)",
    year = "2026",
    publisher = "Association for Computational Linguistics",
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
