# Object Detection Learning Project with YOLO

This repository is designed to help students and beginners get started with machine learning and computer vision, specifically focusing on object detection using YOLO (You Only Look Once). The project demonstrates both using pre-trained models and fine-tuning them with custom datasets.

## 🎯 Project Overview

This project covers three main aspects of object detection:
1. Using pre-trained YOLO models for basic object detection
2. Labeling custom image datasets using Label Studio
3. Fine-tuning YOLO models with custom data

## 🚀 Getting Started

### Prerequisites

- Python (recommended version 3.8+)
- Docker (for running Label Studio)
- GPU recommended for training (but not required)

### Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd object-detection
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start Label Studio for image annotation:
```bash
docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest
```

## 📂 Project Structure

```
├── base/                  # Base model inference
│   ├── score_base_in/     # Input images for base model
│   └── score_base_out/    # Output detections from base model
├── custom/                # Custom model training
│   ├── images/            # Custom dataset images
│   ├── labels/            # YOLO format labels
│   └── fine-tune/         # Training outputs
├── mydata/                # Label Studio data
└── showcase.ipynb         # Main tutorial notebook
```

## 📚 Tutorial Content

The `showcase.ipynb` notebook guides you through:

1. **Setup and Imports**: Basic configuration and library imports
2. **Pre-trained Model Usage**: Using YOLO model pre-trained on COCO dataset
3. **Fine-tuning Process**: Transfer learning with custom labeled data
4. **Custom Inference**: Using your fine-tuned model on new images

## 🏷️ Data Labeling

This project uses Label Studio for image annotation:

1. Access Label Studio at http://localhost:8080 after starting the Docker container
2. Create a new project
3. Import your images
4. Label objects in your images
5. Export annotations in YOLO format

## 🔧 Model Training

The project demonstrates two key workflows:

1. **Using Pre-trained Models**:
   - Direct inference using COCO-trained YOLO model
   - Supports 80 common object classes

2. **Custom Model Training**:
   - Fine-tune pre-trained model on your data
   - Transfer learning with custom object classes
   - Configuration via YAML files

## 📊 Results

Training results and model weights are saved in:
- Base model results: `base/score_base_out/`
- Fine-tuned model results: `custom/fine-tune/`
- Best weights: `custom/fine-tune/weights/best.pt`

## 📖 Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Label Studio Documentation](https://labelstud.io/guide/)
- [Object Detection Basics](https://www.tensorflow.org/tutorials/images/object_detection)

## 🤝 Contributing

Contributions to improve the educational content or add new features are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## 📜 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
