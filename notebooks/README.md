# SaMPiGe-Distill Notebooks

This directory contains Jupyter notebooks for running SaMPiGe-Distill on Kaggle and Google Colab.

## 📁 Available Notebooks

### 1. `sampige_distill.ipynb`
**Complete Implementation for Kaggle & Google Colab**

This notebook provides a **full, end-to-end implementation** of the SaMPiGe-Distill pipeline with:

- ✅ **Setup & Installation**: Automatic package installation
- ✅ **Configuration**: Centralized settings for all components
- ✅ **Data Loading**: KITTI dataset with preprocessing
- ✅ **Model Architecture**: DINOv2 teacher + YOLO student
- ✅ **Knowledge Distillation**: All 7 distillation signals
- ✅ **Training Pipeline**: Complete training with PyTorch Lightning
- ✅ **Validation & Testing**: Comprehensive evaluation
- ✅ **Inference & Visualization**: Prediction visualization
- ✅ **Ablation Studies**: Configuration templates for systematic evaluation
- ✅ **Results & Analysis**: Training metrics visualization

## 🚀 Quick Start

### On Kaggle:

1. **Create a new notebook** on [Kaggle](https://www.kaggle.com/)
2. **Add the SaMPiGe-Distill repository** as a dataset:
   - Click "Add Data" → "GitHub"
   - Enter: `prismairesearchlabs-cell/Sampige`
3. **Upload this notebook** (`sampige_distill.ipynb`)
4. **Run all cells**

### On Google Colab:

1. **Open Colab**: [https://colab.research.google.com/](https://colab.research.google.com/)
2. **Upload the notebook**: File → Upload notebook
3. **Mount Google Drive**: For data storage
4. **Run all cells**

### Local Jupyter:

```bash
# Clone the repository
git clone https://github.com/prismairesearchlabs-cell/Sampige.git
cd Sampige/notebooks

# Install dependencies
pip install -r ../requirements.txt

# Start Jupyter
jupyter notebook sampige_distill.ipynb
```

## 📋 Notebook Structure

The notebook is organized into **10 main sections**:

### 1. Setup & Installation 🛠️
- Detects environment (Kaggle/Colab/Local)
- Installs required packages automatically
- Verifies package versions

### 2. Configuration 🎛️
- Imports centralized configuration
- Displays all settings (device, model, training parameters)
- Environment-specific path setup

### 3. Data Loading & Preprocessing 📁
- Loads KITTI dataset
- Sets up train/val/test splits
- Displays dataset statistics

### 4. Model Architecture 🏗️
- Initializes DINOv2 teacher (frozen)
- Initializes YOLO-based student
- Sets up projection heads
- Initializes prototype module

### 5. Knowledge Distillation Components 🔄
- Creates knowledge distiller
- Sets up detection loss
- Configures dynamic weight scheduler
- Displays initial loss weights

### 6. Training Pipeline 🚀
- Configures PyTorch Lightning trainer
- Sets up callbacks (checkpoint, LR monitor, early stopping)
- Configures loggers (TensorBoard, CSV)
- Starts training

### 7. Validation & Testing 📊
- Loads best checkpoint
- Runs validation
- Runs testing
- Displays results

### 8. Inference & Visualization 🎨
- Runs inference on validation images
- Visualizes predictions with bounding boxes
- Shows confidence scores
- Displays class labels

### 9. Ablation Studies 🔬
- Defines configurations for systematic evaluation
- Shows how to isolate each component's contribution
- Template for comparing different setups

### 10. Results & Analysis 📈
- Loads training logs
- Plots training curves (loss, mAP)
- Visualizes metrics progression
- Provides insights for improvement

## 🎯 Key Features

### Automatic Environment Detection
```python
IN_KAGGLE = 'KAGGLE_URL' in os.environ
IN_COLAB = 'google.colab' in sys.modules
```

### Environment-Specific Paths
```python
if IN_KAGGLE:
    config.path.IMAGE_DIR = '/kaggle/input/kitti-dataset/training/image_2'
    config.path.LABEL_DIR = '/kaggle/input/kitti-dataset-yolo-format/labels'
elif IN_COLAB:
    config.path.IMAGE_DIR = '/content/drive/MyDrive/KITTI/training/image_2'
    config.path.LABEL_DIR = '/content/drive/MyDrive/KITTI/labels'
```

### Complete Training Pipeline
```python
trainer = pl.Trainer(
    accelerator=config.system.ACCELERATOR,
    devices=config.system.NUM_DEVICES,
    strategy=config.system.STRATEGY,
    max_epochs=config.training.EPOCHS,
    precision=config.system.PRECISION,
    callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    logger=loggers
)

trainer.fit(distiller, datamodule=datamodule)
```

### Visualization
```python
visualizer = DetectionVisualizer(class_names)
fig, axes = plt.subplots(1, num_visualize, figsize=(16, 4))

for i in range(num_visualize):
    ax = axes[i] if num_visualize > 1 else axes
    img = images[i].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)
    # Draw bounding boxes and labels
```

## 📊 Expected Output

### Training Progress
```
Epoch 0: Train Loss = 2.456, Val Loss = 1.892, mAP50 = 0.123
Epoch 5: Train Loss = 1.234, Val Loss = 1.123, mAP50 = 0.456
Epoch 10: Train Loss = 0.892, Val Loss = 0.876, mAP50 = 0.567
...
Epoch 99: Train Loss = 0.345, Val Loss = 0.321, mAP50 = 0.789
```

### Validation Results
```
mAP50: 0.78
mAP50-95: 0.61
Precision: 0.85
Recall: 0.82
F1: 0.83
```

### Visualization
- Images with bounding boxes
- Class labels and confidence scores
- Color-coded by class
- Ground truth vs predictions overlay

## 🔧 Customization

### Change Dataset Paths
Edit the path configuration in Section 2:
```python
if IN_KAGGLE:
    config.path.IMAGE_DIR = '/your/kaggle/path/images'
    config.path.LABEL_DIR = '/your/kaggle/path/labels'
```

### Adjust Training Parameters
Modify the configuration in Section 2:
```python
config.training.BATCH_SIZE = 32
config.training.EPOCHS = 200
config.training.LEARNING_RATE = 2e-4
```

### Change Model Architecture
Edit the model initialization in Section 4:
```python
# Use different teacher model
config.model.TEACHER_MODEL = 'dinov2_vits14'

# Use different student backbone
config.model.STUDENT_BACKBONE = 'yolov8m'
```

### Modify Loss Weights
Adjust the scheduler in Section 5:
```python
initial_weights = {
    'detection': 1.0,
    'feature': 0.5,
    'attention': 0.3,
    'relation': 0.2,
    'prototype': 0.2,
    'patch': 0.1,
    'consistency': 0.1
}
```

## 📈 Ablation Study Template

The notebook includes a template for running ablation studies:

```python
configurations = {
    'YOLO Only': {'feature': 0, 'attention': 0, 'relation': 0, 'prototype': 0, 'patch': 0, 'consistency': 0},
    'YOLO + DINO Feature': {'feature': 1, 'attention': 0, 'relation': 0, 'prototype': 0, 'patch': 0, 'consistency': 0},
    'YOLO + DINO + Attention': {'feature': 1, 'attention': 1, 'relation': 0, 'prototype': 0, 'patch': 0, 'consistency': 0},
    'Full SaMPiGe': {'feature': 1, 'attention': 1, 'relation': 1, 'prototype': 1, 'patch': 1, 'consistency': 1}
}
```

To run a complete ablation study:
1. Train each configuration separately
2. Record the results
3. Compare performance metrics

## 🎨 Visualization Examples

### Training Curves
- Loss vs Epoch (Train & Val)
- mAP vs Epoch
- Learning Rate vs Epoch

### Prediction Visualization
- Bounding boxes with class labels
- Confidence scores
- Ground truth vs predictions
- Feature maps (optional)

### Metrics Comparison
- Precision-Recall curves
- Confusion matrix
- Per-class performance

## 🚀 Performance Tips

### Kaggle Optimization
- **Use GPU**: Enable GPU acceleration in notebook settings
- **Dataset**: Add KITTI dataset as input
- **Memory**: Monitor GPU memory usage
- **Checkpoints**: Save to `/kaggle/working/`

### Colab Optimization
- **GPU**: Runtime → Change runtime type → GPU
- **Drive**: Mount Google Drive for data storage
- **TPU**: Consider using TPU for larger models
- **TensorBoard**: Use `%tensorboard` for real-time monitoring

### General Tips
- **Batch Size**: Adjust based on GPU memory
- **Mixed Precision**: Use `precision='16-mixed'` for faster training
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Monitors validation metrics

## 📚 Additional Resources

- **[GitHub Repository](https://github.com/prismairesearchlabs-cell/Sampige)**: Complete source code
- **[DINOv2 Paper](https://arxiv.org/abs/2304.07193)**: Self-supervised vision transformer
- **[YOLOv8 Documentation](https://docs.ultralytics.com/)**: Object detection backbone
- **[PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)**: Training framework
- **[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)**: Benchmark dataset

## 🙏 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision

2. **Missing Packages**
   - Run `!pip install package-name` in a notebook cell
   - Check the requirements.txt file

3. **Dataset Not Found**
   - Verify paths in configuration
   - Check file permissions
   - Ensure dataset is properly mounted

4. **Slow Training**
   - Use GPU acceleration
   - Enable mixed precision
   - Reduce image size

### Debug Mode

Add debug prints to track progress:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📝 License

This notebook is part of the **SaMPiGe-Distill** project, licensed under the **MIT License**.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 🙏 Acknowledgments

- **DINOv2 Team** (Facebook Research) - Self-supervised vision transformer
- **Ultralytics** - YOLOv8 object detection
- **PyTorch Lightning Team** - Training framework
- **KITTI Dataset** - Benchmark dataset

---

**SaMPiGe-Distill: Where Self-Supervised Learning Meets Object Detection** 🚀

*Built with ❤️ for the research community*
