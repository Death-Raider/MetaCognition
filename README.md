# HiPO: Hierarchical Preference Optimization

## Introduction

This repository contains the implementation for training language models using HiPO (Hierarchical Preference Optimization) format. HiPO enhances model responses by structuring outputs into three components:

1. **Refined Query (Rq)**: A clarified and improved version of the user's original query
2. **Meta Thinking (Mt)**: The model's reasoning process and approach to solving the problem
3. **Answer (Ra)**: The final response to the query

By training models with DPO on HiPO-formatted data, we encourage the model to not only provide accurate answers but also demonstrate transparent reasoning and query understanding.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ GPU memory (for 7B models)

### Setup

## Dataset Format

## General Usage

### 1. Standard DPO Training (Using `output_a/b`)

### 2. HiPO Training (Using `Rq + Mt + Ra`)

### 3. Configuration

### 4. Training Hyperparameters

### 5. Using Trained Models

## Project Structure

## HiPO Instructions File

## Citation

*(To be added upon publication)*

If you use this code or method in your research, please cite:

```bibtex
@article{yourname2024hipo,
  title={HiPO-DPO: Hierarchical Prompt Optimization with Direct Preference Optimization},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Paper

[Link to paper will be added here]

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: kachroo.darsh@gmail.com
