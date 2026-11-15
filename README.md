# BanglaLekha EDA on macOS

Start with `01_dataset_eda.ipynb`, then use `02_model_training.ipynb` for model training. Both use local files, short text, commented code and small cells. The EDA notebook does not train a model.

Your existing environment is `/Users/sakibsmacbookm5/Documents/Python work/.venv`. Keep using it; no new host environment is needed. The Dockerfile below is for you to create and understand. No Dockerfile or Docker image has been created for you.

## 1. Activate your existing environment

In your terminal:

```bash
cd "/Users/sakibsmacbookm5/Documents/Python work/research-project/SICIP - Assignment 4"
source "../../.venv/bin/activate"
python -c "import sys; print(sys.executable)"
```

The printed interpreter should end with `Python work/.venv/bin/python`.

## 2. Install the EDA dependencies

Your existing environment was created with `uv` and does not currently contain pip. Use `uv` to install directly into it:

```bash
uv pip install --python "../../.venv/bin/python" -r requirements-eda.txt
```

This installs into the existing environment without creating another one. The Docker image has its own pip installation, so the later Dockerfile uses pip there.

## 3. Run the notebook locally first

In VS Code, open `01_dataset_eda.ipynb`, choose **Select Kernel → Python Environments**, and select the interpreter from step 1. Run the cells from top to bottom. The first cell prints the actual interpreter so you can verify the selection.

Alternatively, start JupyterLab from the activated terminal:

```bash
python -m jupyterlab
```

Open the local link printed in the terminal, then open the notebook. Launch Jupyter from this project directory so its relative dataset paths work.

The current folder should contain:

```text
SICIP - Assignment 4/
├── 01_dataset_eda.ipynb
├── requirements-eda.txt
├── README.md
├── BanglaLekha-Isolated.zip
└── BanglaLekha-Isolated/
    ├── Readme.txt
    ├── Form-based Marking.xlsx
    └── Images/
        ├── 1/
        ├── 2/
        └── ... 84/
```

Your dataset is already extracted. If you later keep only the zip, extract it here before running the notebook. Change `DATASET_DIR` in the notebook only if you move the data.

By default, the notebook decodes **every image**. Set `FULL_SCAN = False` for a faster sample-based image audit. Metadata and class counts always cover the full dataset; image-quality and duplicate findings cover only the scanned images. Reports are saved under `reports/eda/`, and the source images are not changed.

## 4. Start Docker Desktop

Open Docker Desktop from Applications and wait until its engine is running. In the terminal:

```bash
docker version
docker info
```

`docker version` should show both **Client** and **Server** information. The initial check found the command-line client installed, with no running engine connection. If needed, use the Apple silicon installer for your arm64 Mac. [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)

## 5. Create your Dockerfile

Create a file named exactly `Dockerfile` in the project root, without a `.txt` extension. Type these instructions into it:

```dockerfile
# 1. Start with a Linux image containing Python 3.12.
FROM python:3.12-slim

# 2. Use this directory when the container starts.
WORKDIR /workspace

# 3. Avoid Python cache files and display logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. Install dependencies before adding or mounting project files.
COPY requirements-eda.txt /tmp/requirements-eda.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-eda.txt

# 5. Document the port on which Jupyter will listen.
EXPOSE 8888

# 6. Start Jupyter inside this local development container.
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/workspace"]
```

Read each instruction before proceeding:

| Instruction | Purpose |
|---|---|
| `FROM` | Supplies Linux and Python inside the image; your Mac's `.venv` stays on the Mac. |
| `WORKDIR` | Sets the container's default directory. |
| `ENV` | Configures Python's cache and log behavior. |
| `COPY` | Copies only the dependency list into the image. |
| `RUN` | Installs packages while building the image. |
| `EXPOSE` | Documents a port; the later `-p` argument makes it reachable. |
| `CMD` | Starts Jupyter when you run a container from the image. |

The image contains Python and its dependencies. Your notebook and dataset will be mounted at runtime. Docker installs Linux packages in the image rather than copying macOS packages from `.venv`. The official Python image supports arm64, so no `--platform=linux/amd64` flag is needed for this local build. [Official Python image](https://hub.docker.com/_/python)

## 6. Create `.dockerignore`

Create a second file named `.dockerignore` next to `Dockerfile`:

```text
.venv/
**/.venv/
.git/
__pycache__/
.ipynb_checkpoints/
.DS_Store
BanglaLekha-Isolated/
*.zip
reports/
```

This keeps the large dataset, generated reports and environment files out of the build context. They remain available through the runtime mount. Copying the dependency list before other files also lets Docker reuse the dependency layer when that list is unchanged. [Docker build practices](https://docs.docker.com/build/building/best-practices/)

## 7. Build your image

Run this from the project directory containing `Dockerfile`:

```bash
docker build -t bangla-eda:0.1 .
```

`-t` names the image `bangla-eda` with tag `0.1`. The final `.` is the build context: the current directory. The first build downloads the base image and packages.

Check that the image exists:

```bash
docker image ls bangla-eda
```

An **image** is the built template. The next step starts a **container**, a running instance of it. [Running containers](https://docs.docker.com/engine/containers/run/)

## 8. Run Jupyter in your container

Stop your local Jupyter server first if it is using port 8888, then run:

```bash
docker run --rm -it \
  --name bangla-eda \
  -p 127.0.0.1:8888:8888 \
  --mount "type=bind,source=$(pwd),target=/workspace" \
  bangla-eda:0.1
```

| Argument | Purpose |
|---|---|
| `--rm` | Removes the container after it stops; the image remains. |
| `-it` | Displays the server output in your terminal. |
| `--name` | Names this container `bangla-eda`. |
| `-p 127.0.0.1:8888:8888` | Connects your Mac's localhost port to the Jupyter port in the container. |
| `--mount` | Shares the current project folder with `/workspace` inside the container. |

The quotes around the mount argument are necessary because your project path contains spaces. Notebook edits and reports written under `/workspace` are saved to the same project folder on your Mac. [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)

## 9. Open and verify the container notebook

Visit `http://127.0.0.1:8888/lab`. Use the login token printed in the terminal; authentication remains enabled. Open `01_dataset_eda.ipynb` and select **Python 3 (ipykernel)**. [Jupyter server access](https://jupyter-server.readthedocs.io/en/latest/operators/public-server.html)

Run the first two code cells. They should report **Linux**, a container Python interpreter and `/workspace` as the project directory. On native macOS, those same cells report **Darwin** and your existing `.venv`. This confirms which Python is executing the code.

Now run the remaining cells. Images come from the mounted dataset, and reports appear in your Mac's `reports/eda/` folder.

## 10. Stop and rerun

From another terminal:

```bash
docker stop bangla-eda
```

Use the step 8 command again to start a fresh container from the same image. Notebook edits are immediately visible through the mount; rebuild the image after changing dependencies or `Dockerfile`.

If port 8888 is occupied, change only the host side to `-p 127.0.0.1:8889:8888`, then open `http://127.0.0.1:8889/lab`. If Docker says it cannot connect to its daemon, return to step 4.

## 11. Train models locally on macOS

Use `02_model_training.ipynb` after running the full EDA. From the project directory, install the training dependencies into your existing environment:

```bash
source "../../.venv/bin/activate"
uv pip install --python "../../.venv/bin/python" -r requirements-training.txt
```

Select that same `.venv` as the notebook kernel. Restart the kernel after installing packages, then run the notebook from the top. Initial ImageNet weights download to the project's `cache/torch/` directory. Subsequent training uses local files.

The notebook selects **MPS** when PyTorch reports Apple's GPU backend available, otherwise **CPU**. It uses FP32 and zero loader workers for macOS notebook compatibility. The Docker setup above uses CPU because its Python runs inside Linux. Native macOS is the route provided for MPS training. [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

The notebook compares two model families:

| Stage | ResNet-18 | EfficientNet-B0 |
|---|---|---|
| Character pretraining | ImageNet encoder + 84-class head | ImageNet encoder + 84-class head |
| Optional whole-word training | Character encoder + BiLSTM/CTC | Character encoder + BiLSTM/CTC |

Both character models fine-tune all encoder layers using AdamW, a smaller encoder learning rate, warmup/cosine decay, mild handwriting augmentation, dropout, label smoothing, gradient clipping and early stopping. These are practical candidates, not a guarantee of the best possible architecture. [Torchvision model API](https://docs.pytorch.org/vision/stable/models.html)

Data preparation uses the full EDA reports to exclude filename/class inconsistencies, failed decodes, blank flags and conflicting exact duplicates. It keeps one copy of each remaining pixel pattern and creates approximately 70/15/15 splits by collection form. The source images are unchanged. Rerun EDA if you replace or modify the dataset. A collection form is a proxy for handwriting identity; the supplied README does not guarantee globally unique writer IDs.

Each model has separate training, test-evaluation and plotting cells. Outputs include:

- Classification reports, per-class precision/recall/F1 and held-out predictions/errors.
- Training/validation loss and accuracy curves; character confusion matrices and frequent confusion pairs.
- Parameter counts, model sizes, best epoch, batch-1 median/p95 latency and throughput.
- MLflow parameters, epoch metrics, environment details, reports and saved model files.
- For word models: exact joined-word accuracy, NFC code-point CER, grapheme CER, one-word WER and held-out results for **মানুষ** and **জল**.

The winning character checkpoint maximizes **validation macro F1**; the winning word checkpoint minimizes **validation CER**. Test results do not select the winner. Training metrics use augmented inputs and dropout, while validation uses clean inputs. `training_seconds` sums measured training-step computation and excludes data loading, validation and checkpoint writes. Timing uses 10 warmup passes and the configured number of repetitions on one held-out image; end-to-end timing includes preprocessing/transfer/decoding but excludes disk reads.

## 12. Resume training after stopping the kernel

Leave `EXPERIMENT_NAME`, `CFG`, data and library versions unchanged, then run from the top. Completed training is loaded; interrupted training continues from the next batch after the last committed checkpoint. Work since the last checkpoint is repeated.

Checkpoints save every 500 batches or five minutes, plus every epoch boundary. Two alternating atomic slots provide fallback if the latest is unreadable. Saved state includes the optimizer, model, random generators, batch position, partial metrics, best weights and early-stopping count. Only load full resume checkpoints from your own runs. Inference artifacts load tensor state dictionaries with `weights_only=True`. [PyTorch checkpoint documentation](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

Use a new experiment name if you change data or hyperparameters. Reduce the batch size before starting if the Mac has insufficient memory. The optional uint8 image cache is memory mapped on disk and reused across restarts. Different devices can still produce different floating point results.

## 13. Supply real word images for whole-word recognition

BanglaLekha-Isolated supplies isolated characters, without whole-word transcripts. Character pretraining alone cannot establish that a connected drawing of **মানুষ** or **জল** will be recognized correctly. The notebook refuses to serve a character artifact through its word-prediction method.

To train the provided word stage, prepare real connected word crops and a UTF-8 CSV:

```text
word_dataset/
├── labels.csv
└── images/
    └── ...
```

```csv
path,text,writer_id,split
images/train_001.png,মানুষ,writer001,train
images/train_002.png,জল,writer002,train
images/val_001.png,মানুষ,writer101,val
images/val_002.png,জল,writer102,val
images/test_001.png,মানুষ,writer201,test
images/test_002.png,জল,writer202,test
```

These rows illustrate the schema, not a sufficient training corpus. Use many correctly transcribed examples from independent writers and the drawing styles expected in the future application. Each writer and exact duplicate must stay in a single split. If you omit `split`, the notebook creates grouped partitions and requires at least seven writers.

Set these values in the notebook's settings cell:

```python
TRAIN_WORDS = True
WORD_CSV = PROJECT_DIR / 'word_dataset' / 'labels.csv'
```

Each word recognizer transfers its matching character encoder and learns alignment from whole word images using CTC. It returns one joined NFC-normalized string, with no character segmentation or forced dictionary replacement. CTC loss runs on CPU and autograd carries gradients back to an MPS encoder. Held-out metrics measure actual recognition; training a model does not guarantee correct predictions for every drawing.

## 14. Saved models and MLflow

The default experiment directory is `training_runs/macos_resnet18_efficientnetb0_v1/`:

```text
character/<architecture>/     # Checkpoints, history, weights, architecture and test reports
word/<architecture>/          # Present when word training is enabled
selected/character/           # Selected character-pretraining weights
selected/word/                # Selected word-supervised weights, when enabled
mlruns/                      # Local MLflow experiment records
selected_models.zip          # Selected weights, metadata, requirements and inference source
training_status.json          # Explicit word-training status
```

For the future Streamlit/Docker application, copy a selected model directory and use its generated `bangla_inference.py`:

```python
from bangla_inference import Predictor

recognizer = Predictor('best_model.pt', device='cpu')
prediction = recognizer.predict_word('my_word_drawing.png')
print(prediction)
```

Use `selected/word` for this example. `predict_character_id()` is available separately for character model inspection. Inference does not download pretrained weights.

MLflow tracking needs no running server. To open the optional browser UI, install the full package into the same existing environment:

```bash
uv pip install --python "../../.venv/bin/python" "mlflow==2.22.5"
python -m mlflow ui --backend-store-uri ./training_runs/macos_resnet18_efficientnetb0_v1/mlruns --port 5000
```

Open `http://127.0.0.1:5000`. Change the path in that command if you changed the experiment name. [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)

The Docker guide's EDA image remains focused on EDA. When you choose to containerize training yourself, change its dependency file to `requirements-training.txt` and include `requirements-eda.txt` before installing it, since the training requirements reference that file. Container training in the provided setup will use CPU.
