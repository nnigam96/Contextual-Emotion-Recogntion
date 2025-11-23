# Contextual Emotion Recognition

A PyTorch implementation of the Emotic CNN methodology for recognizing human emotions in images using both body pose and contextual scene information. This project extends emotion recognition beyond facial expressions by incorporating full-body context and environmental cues.

## Problem Statement

Traditional emotion recognition systems rely primarily on facial expressions, which can be limited when faces are occluded, distant, or not clearly visible. The Emotic approach addresses this by:
- **Multi-Modal Input**: Combining body pose features with scene context
- **Discrete + Continuous Emotions**: Predicting both categorical emotions (26 classes) and continuous valence-arousal-dominance (VAD) dimensions
- **Context-Aware**: Leveraging scene information to improve emotion understanding

## Features

- **Dual-Branch Architecture**: Separate ResNet encoders for body and context features
- **Emotic Dataset Support**: Full implementation for the Emotic emotion recognition dataset
- **Facial Feature Integration**: Optional facial tagging for enhanced recognition
- **YOLO Integration**: Person detection and bounding box extraction
- **Flexible Training**: Supports ResNet18 and ResNet50 backbones
- **Loss Functions**: Combined categorical and continuous loss with dynamic weighting

## Project Structure

```
Contextual-Emotion-Recogntion/
├── main.py                    # Entry point for training/testing/inference
├── train.py                   # Training loop implementation
├── test.py                    # Evaluation on test set
├── inference.py               # Single image inference
├── emotic.py                  # Emotic model architecture
├── emotic_dataset.py          # Dataset loading and preprocessing
├── loss.py                    # Combined discrete + continuous loss
├── yolo_inference.py          # YOLO-based person detection
└── facial_tagging_exploration.ipynb  # Facial feature analysis
```

## Technical Details

### Architecture

- **Body Model**: ResNet18/50 for extracting body pose features
- **Context Model**: ResNet18/50 for extracting scene context features
- **Fusion**: Concatenated features passed through MLP heads
- **Outputs**: 
  - 26 discrete emotion categories
  - 3 continuous VAD dimensions (Valence, Arousal, Dominance)

### Loss Function

- **Categorical Loss**: Weighted cross-entropy for discrete emotions
- **Continuous Loss**: Smooth L1 or L2 for VAD regression
- **Dynamic Weighting**: Adaptive loss weight policy

## Quick Start

### Prerequisites

```bash
pip install torch torchvision
# Additional dependencies for YOLO and facial features
```

### Training

```bash
python main.py \
    --mode train \
    --data_path /path/to/emotic/data \
    --experiment_path /path/to/experiments \
    --context_model resnet18 \
    --body_model resnet18 \
    --epochs 15 \
    --batch_size 52
```

### Inference

```bash
python main.py \
    --mode inference \
    --inference_file /path/to/image/list.txt \
    --experiment_path /path/to/experiments
```

## Dataset

This implementation uses the [Emotic dataset](https://github.com/Tandon-A/emotic), which contains:
- Images with annotated bounding boxes
- 26 discrete emotion categories
- Continuous VAD annotations

## Citation

If you use this implementation, please cite the original Emotic paper:
```
@inproceedings{kosti2017emotion,
  title={Emotion recognition in context},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2017}
}
```

## License

See [LICENSE](LICENSE) file for details.
