# Fine-Tuning Showdown: Project Instructions

## 1. Project Summary

In this project, you will build and compare two fine-tuned Large Language Models trained on the **same dataset**:

1. A **proprietary model** fine-tuned using the OpenAI API
2. An **open-source Llama 3.2 3B model** fine-tuned using **QLoRA**

You will evaluate both models using identical test data and analyze trade-offs across performance, cost, latency, resource usage, and deployment complexity.

The primary objective is not to declare a universal winner, but to develop a structured framework for deciding **when to use proprietary fine-tuning versus open-source fine-tuning** in real-world scenarios.

## 2. Learning Objectives

By completing this assignment, you will:

- Fine-tune a proprietary LLM using OpenAI’s fine-tuning API
- Fine-tune an open-source LLM using QLoRA
- Build evaluation pipelines for both approaches
- Benchmark performance on identical test sets
- Analyze cost, latency, and deployment trade-offs

## 3. Dataset

### Dataset Name

**MedQA (Medical Question Answering)**

### Source

OpenLifeScience AI MedQA dataset on Hugging Face.

### Description

- Medical board exam–style multiple-choice questions
- Approximately 12,000 questions
- Topics include Anatomy, Pharmacology, Pathology, Surgery, and Internal Medicine
- Each example includes:
  - Question
  - Four answer options (A/B/C/D)
  - Correct answer
  - Explanation

### Dataset Split Requirements

You must use:

- Training set
- Validation set
- Held-out test set

**Important:**  
The **same test set must be used for both models** to ensure a fair comparison.

## 4. Environment Setup

This assignment uses two different fine-tuning approaches with different infrastructure needs.

- **OpenAI fine-tuning** can be completed locally on CPU.
- **Open-source fine-tuning with QLoRA** requires a GPU and is strongly recommended to be run on Google Colab.

You may use other environments if you are confident in your setup.

### Option A: Local Environment (OpenAI Fine-Tuning)

This environment is sufficient for:

- Dataset preparation
- OpenAI fine-tuning
- OpenAI model evaluation
- Comparative analysis

#### Prerequisites

- Python 3.9 or higher
- No GPU required

#### Setup

Create and activate a virtual environment:

```bash
conda create -n finetuning-openai python=3.10
conda activate finetuning-openai
```

Install required dependencies:

```bash
pip install -U \
  openai python-dotenv \
  pandas numpy jsonlines \
  scikit-learn \
  matplotlib plotly tqdm
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

This environment will be used only for the proprietary fine-tuning track.

### Option B: Google Colab (Open-Source QLoRA Fine-Tuning)

This environment is required for:

- QLoRA training
- GPU-based inference
- Resource usage measurement

#### Setup

1. Open a new Google Colab notebook
2. Set runtime type to **GPU**
3. Install dependencies:

```bash
pip install -U \
  torch transformers datasets accelerate \
  bitsandbytes peft trl \
  huggingface-hub \
  scipy tqdm
```

4. Authenticate with Hugging Face:

```bash
huggingface-cli login
```

You must ensure that:

- The model is loaded in 4-bit
- LoRA adapters are used
- Only adapter weights are trained

### Notes

- Do not attempt full fine-tuning of the open-source model.
- If running QLoRA locally, you are responsible for installing PyTorch with CUDA support and ensuring `bitsandbytes` works correctly.
- You may use separate environments for the two tracks.

## 5. Part 1: Proprietary Model Fine-Tuning (OpenAI)

### Objectives

- Prepare data in OpenAI’s JSONL message format
- Fine-tune a proprietary model using the OpenAI API
- Evaluate the fine-tuned model
- Track training and inference costs

### Data Format

Each training example must follow OpenAI’s chat message format:

```json
{
  "messages": [
    { "role": "system", "content": "You are a medical expert assistant." },
    {
      "role": "user",
      "content": "Question: <question text>\nA) <option A>\nB) <option B>\nC) <option C>\nD) <option D>\n\nAnswer:"
    },
    {
      "role": "assistant",
      "content": "<correct letter>. Explanation: <explanation>"
    }
  ]
}
```

### Fine-Tuning

- Upload training and validation files
- Launch a fine-tuning job using OpenAI’s API
- Monitor job status and completion
- Record training cost

### Evaluation Metrics

You must measure:

- Accuracy
- Precision and recall
- Average inference latency
- Cost per 1,000 inferences

### Deliverables

- Data preparation script
- Fine-tuning script
- Evaluation script
- Results file containing metrics and cost information

## 6. Part 2: Open-Source Model Fine-Tuning (QLoRA)

### Objectives

- Prepare data for instruction fine-tuning
- Fine-tune an open-source model using QLoRA
- Track GPU memory usage and training time
- Evaluate using the same test set as Part 1

### Data Format

Each training example must be converted to an instruction-following format:

```text
### Instruction:
Answer this medical question by selecting the correct option.

### Question:
<question>
A) <option A>
B) <option B>
C) <option C>
D) <option D>

### Response:
The correct answer is <letter>. <explanation>
```

### QLoRA Configuration Requirements

- 4-bit quantization using bitsandbytes
- LoRA adapters with documented hyperparameters
- Parameter-efficient fine-tuning only (full fine-tuning is not allowed)

### Training

- Load the open-source model with quantization
- Attach LoRA adapters
- Train using a supervised fine-tuning trainer
- Save adapter weights
- Log training time and GPU memory usage

### Evaluation Metrics

You must measure:

- Accuracy
- Precision and recall
- Average inference latency
- GPU memory usage
- Total training cost (if applicable)

### Deliverables

- Data preparation script
- Training notebook or script
- Evaluation script
- Results file containing metrics and resource usage

## 7. Evaluation and Comparison

You must compare both approaches across the following dimensions:

### Performance

- Accuracy
- Precision
- Recall

### Latency

- Average latency

### Cost

- Training cost
- Inference cost per 1,000 requests
- Break-even analysis where applicable

### Resource Utilization

- GPU memory usage
- Training time
- Model or adapter size

### Deployment Considerations

- Time to deploy
- Scalability
- Maintenance burden
- Privacy implications
- Offline capability

## 8. Comparative Analysis

You must do a comparative analysis that includes:

- Performance comparison
- Cost analysis
- Deployment trade-offs
- Clear recommendations for when each approach should be used
