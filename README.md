# llama-sql-fine-tuning
Fine-tuning Llama 2 model for SQL query generation from natural language using LoRA and PEFT techniques

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VKewUPYJ2228wcNB7zsAlDPo0po-Jcos)

## Overview

This project demonstrates fine-tuning the Llama 2 model to generate SQL queries from natural language instructions. The implementation uses Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to efficiently adapt the model for the SQL generation task.

## Features

- **Efficient Training**: Uses LoRA and 4-bit quantization to fine-tune large language models on limited hardware
- **Dataset**: Fine-tuned on the Llama-2-SQL-Dataset from HuggingFace
- **Model**: Based on Meta's Llama-2-7b model with optimized configurations
- **Production-Ready**: Includes best practices for LLM fine-tuning and inference

## Technical Stack

- **Framework**: Transformers, PEFT, TRL
- **Quantization**: BitsAndBytes 4-bit quantization
- **Training**: Supervised Fine-Tuning (SFT) with custom parameters
- **Hardware**: Optimized for Google Colab with GPU acceleration

## Implementation Details

### Model Configuration
- Base Model: `meta-llama/Llama-2-7b`
- LoRA rank (r): 16
- LoRA alpha: 32
- Target modules: Query and Value projection layers
- Quantization: 4-bit with NF4 type

### Training Parameters
- Dataset: `ChrisHayduk/Llama-2-SQL-Dataset`
- Max sequence length: 512 tokens
- Batch size and gradient accumulation optimized for memory efficiency
- Mixed precision training with fp16

## Getting Started

### Run in Google Colab

Click the "Open in Colab" badge above to run the notebook directly in your browser. No local setup required!

### Key Steps

1. **Install Dependencies**
   - TRL for efficient training
   - BitsAndBytes for quantization
   - PEFT for LoRA adaptation

2. **Load and Quantize Model**
   - Configure 4-bit quantization
   - Load Llama 2 model with optimized settings

3. **Configure LoRA**
   - Set up low-rank adaptation parameters
   - Target specific model layers for efficiency

4. **Fine-Tune**
   - Train on SQL dataset
   - Monitor training metrics

5. **Inference**
   - Generate SQL queries from natural language
   - Evaluate model performance

## Results

The fine-tuned model can generate accurate SQL queries from natural language descriptions, demonstrating:
- Understanding of SQL syntax and semantics
- Ability to handle complex query structures
- Improved performance on domain-specific tasks

## Applications

- Natural language to SQL interfaces
- Database query automation
- Business intelligence tools
- Data analytics platforms

## Skills Demonstrated

- Large Language Model fine-tuning
- Parameter-efficient training techniques (LoRA, QLoRA)
- Model quantization and optimization
- ML pipeline development
- Production ML best practices

## About

This project was created as part of exploring state-of-the-art LLM fine-tuning techniques based on the GPT Learning Hub course on Fine-Tuning LLMs.

## License

MIT License - see LICENSE file for details
