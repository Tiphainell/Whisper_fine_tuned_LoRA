Python pipeline for fine-tuning Whisper using LoRA techniques.

## Project Structure

```
Whisper_fine_tuned_LoRA/
│
├─ Bash/                        # Bash scripts:
│    ├─ audio_to_mel.sh         # Transform audio to mel-spectrograms
│    ├─ train.sh                # Launch training
│    └─ inference.sh            # Run inference
│
├─ src/                         # Source code
│    ├─ Features_on_disk.py     # Convert audio to mel-spectrograms using Whisper processor
│    ├─ Training.py             # Train Whisper using LoRA techniques
│    ├─ Inference.py            # Inference script
│    └─ utils/                  # Utility functions
│
├─ config/                      # Configuration files
│    ├─ config_features.yaml    # Settings for audio processing (Step 1)
│    ├─ config_training.yaml    # Settings for training LoRA (Step 2)
│    └─ config_inference.yaml   # Settings for inference (Step 3)


```

## Step 1: Transform Audio to Tensors (Mel-Spectrograms)

- **Script**: ``Features_on_disk.py``
- **Purpose**: Converts audio to mel-spectrograms and transcripts into tokens using Whisper processor.
- **Configuration**: Edit ``config_features.yaml`` to point to your dataset and define:
Whisper architecture, language, and task
- **Data format** : 3 .json files for training, validation, and testing with columns:
``file_name, task (audio task: reading, free speech), start_segment, end_segment, transcript``
- **Output**: Mel-spectrograms stored in the specified output path
- **Note**: This step is done once to save time during training.

## Step 2: Training Whisper using LoRA 


- **Script**: ``Training.py`` 
- **Purpose**: Fine-tune Whisper with LoRA techniques using pretrained models from Hugging Face and the PEFT library.
- **Configuration**: ``config_training.yaml`` : allows you to customize:
  - LoRA parameters: rank, alpha, dropout, target_modules
  - Training hyperparameters: batch_size, learning_rate, num_epochs, seed 
  - Callbacks: early stopping patience & threshold
  - Integration with Weights & Biases for experiment tracking
- **Input** = segments processed as in Step 1
- **Output** : LoRa matrices in peft format


## Step 3: Inference
- Script: ``inference.py``
- Purpose: Run the fine-tuned model on new data.
- Output: CSV file with:
     - Segment identifier
     - Original transcript
     - Transcript from base Whisper
     - Transcript from fine-tuned Whisper
     - Word Error Rate (WER) calculation



# Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:Tiphainell/Whisper_fine_tuned_LoRA.git
cd Whisper_fine_tuned_LoRA
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

# Usage Example

```Bash
# Step 1: Transform audio
python src/Features_on_disk.py --config configs/config_features.yaml

# Step 2: Train LoRA
python src/Training.py --config configs/config_training.yaml

#you can also overwrite some parameters outside config files to train on different ranks:
python src/Training.py --config configs/config_training.yaml --lora_rank 2

# Step 3: Inference
python src/inference.py --config configs/config_inference.yaml

```

💡 Tips
- Ensure your JSON files are formatted correctly for Step 1.
- Track experiments using Weights & Biases
- Adjust LoRA hyperparameters carefully for best performance