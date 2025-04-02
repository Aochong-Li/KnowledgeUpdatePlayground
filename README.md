# Knowledge Update Playground (KUP)

[![Hugging Face Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/aochongoliverli/KUP)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)
[![Build Status](https://img.shields.io/badge/Status-Work%20In%20Progress-orange)]()

---

Welcome to **Knowledge Update Playground (KUP)** — an automatic framework for generating **realistic knowledge update/conflict datasets** and evaluating how well Large Language Models (LLMs) adapt to knowledge changes during **continued pre-training**.

## 🚀 Overview

KUP helps researchers and practitioners:
- **Generate** realistic **knowledge update pairs** to simulate real-world knowledge shifts and conflicts.
- **Evaluate** LLMs’ adaptability to knowledge updates during fine-tuning or continued pre-training.
- **Train** LLMs using both **continued pre-training** and **supervised fine-tuning** following the setup in [Synthetic Continued Pre-training](https://github.com/ZitongYang/Synthetic_Continued_Pretraining).

This playground is designed to benchmark how well LLMs handle **incremental knowledge**, especially in dynamic environments.
> **Note:** The `main` branch is fully functional. However, we are actively working on improving code readability, structure, and usability to make the project more production-ready in `prod` branch.

---

## 📄 Dataset

The KUP dataset contains **5,000 high-quality knowledge update/conflict pairs**, automatically synthesized and verified to represent realistic knowledge shifts.

🔗 **Hugging Face Dataset:**  
[https://huggingface.co/datasets/aochongoliverli/KUP](https://huggingface.co/datasets/aochongoliverli/KUP)

---

## 📥 Installation

```bash
git clone https://github.com/your-username/KUP.git
cd KUP
pip install -r requirements.txt
