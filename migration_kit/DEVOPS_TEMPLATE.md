# DevOps & Hardware Protocols 🖥️☁️

**"The Right Tool for the Right Job."**

This document defines the computational infrastructure and the workflows for moving between environments.

---

## 1. Hardware Tiers ⚙️

### Tier 1: The Agent Environment (Home Base) 🏠
*   **Infrastructure**: GCP Compute Engine (or Local Dev).
*   **Hardware**: **NVIDIA L4 (22 GB vRAM)**.
*   **Purpose**:
    -   Code Authoring & Version Control.
    -   Logic Debugging.
    -   Mini-Audits (< 5 mins).
    -   "Toy" Training runs (Small batch size).
*   **Constraint**: If your model + batch requires > 20GB vRAM, DO NOT run here.

### Tier 2: The Training Cluster (The Muscle) 💪
*   **Infrastructure**: Google Colab Pro+ or A100 Cluster.
*   **Hardware**: **NVIDIA A100 (80 GB vRAM)**.
*   **Purpose**:
    -   Full Scale Training (> 1B parameters).
    -   Long Context Verification (> 32k tokens).
    -   Battle-Testing SOTA.

---

## 2. The "Airlock" Protocol 🚀

Moving code and data between Tier 1 (Agent) and Tier 2 (Cluster) requires strict hygiene to prevent "Version Drift" (where the code on the cluster is different from the code in the repo).

### A. Export (Agent -> Cluster)
1.  **Commit Code**: Ensure `experiments/vXX` is clean.
2.  **Zip**: Create a portable archive.
    ```bash
    zip -r bundle.zip experiments/vXX/models experiments/vXX/train.py requirements.txt
    ```
3.  **Transfer**: Upload `bundle.zip` to Drive/Bucket.

### B. Train (Cluster)
1.  **Mount**: Connect Drive.
2.  **Unzip**: `!unzip bundle.zip`.
3.  **Run**: Execute `train.py`.
4.  **Save**: Ensure logs (`.log`) and checkpoints (`.pt`) are written to a persisted path (Drive).

### C. Import (Cluster -> Agent)
**CRITICAL**: The Agent *must* see the results to know what happened.
1.  **Log Retrieval**: Download the training log.
    ```bash
    gdown --id [FILE_ID] -O experiments/vXX/logs/train_external.log
    ```
2.  **Analysis**: The Agent reads the log and updates `journal.md`.

---

## 4. Operational Protocols 🔧

### A. Data Management 💾

**The Dual-Storage Strategy**:

| Asset Type | Primary Location | Access Protocol |
|---|---|---|
| **Raw Archival** | **Google Drive** | Immutable source of truth. Service account access. |
| **Training (High Speed)** | **HF/Kaggle Datasets** | Use for Colab/Cluster runs. Faster than `gdown`. |
| **Active Training** | **Local SSD** | **Mandatory**: Pre-fetch data from cloud to local SSD (`/content/` or `/mnt/`) before `train.py` starts. Never train directly off network mounts. |

**Location Schema**:
- **Source**: `G Drive/data/raw/` (Archival)
- **Training Source**: `huggingface.co/datasets/[USER]/[PROJECT]` 
- **Local Cache**: `/data/` on the VM SSD.

**Versioning**:
Document in `journal.md`:
```markdown
## Data
- **Dataset**: PG-19 v1.0
- **Source**: https://github.com/deepmind/pg19
- **Downloaded**: 2024-12-22
- **Preprocessing**: Tokenized with GPT-2 BPE, max_length=2048
```

**Sharing Large Datasets**:
1. Upload to Google Drive
2. Get shareable link
3. Document in `README.md` or setup script:
   ```bash
   gdown --id [FILE_ID] -O data/pg19.tar.gz
   tar -xzf data/pg19.tar.gz
   ```

### B. Triple-Redundancy Checkpointing 🛡️

**Goal**: Zero data loss from internet drops, VM crashes, or disk failures.

| Tier | Location | Frequency | Purpose |
|---|---|---|---|
| **L1: Local** | `checkpoints/` | Every 10-20 mins | Instant recovery from process crash. |
| **L2: Cloud (Drive)** | `G Drive/rsm_project/` | Every 1-2 hours | Recovery for **Logs, Journals, and Small Assets**. |
| **L3: Weights (HF)** | `HF Private Repo` | Every 2-4 hours | **Weights Only**. HF's chunked API is more stable for large files. |

> [!TIP]
> **Why the split?**
> Hugging Face (via `huggingface_hub`) handles large file uploads (GBs) far better than Drive's standard API, offering automatic resume-on-failure. Use Drive for the "living" project files and HF for the heavy weights.

**The "Volatile Environment" Protocol (Colab/Preemptible)**:
1.  **Save Local first**: Training script writes to local disk.
2.  **Background Sync**: Run `scripts/checkpoint_sync.py` in a separate terminal/cell. It watches for new files and uploads them.
3.  **No-Hang Policy**: The training loop should **NEVER** wait for a cloud upload. Use background threads.

**Code implementation**:
```python
# In train.py
checkpoint_data = { ... }
path = f"checkpoints/v68_step_{global_step}.pt"
torch.save(checkpoint_data, path)
# The sync script handles the rest.
```

**Resume Strategy**:
1. Check **Local** Disk first.
2. If empty, download `vXX_latest.pt` from **Drive**.
3. If Drive fails, check **HF Hub**.

### C. Dependency Management
**When adding a new package**:
1. Install: `pip install [package]`
2. Test it works
3. Add to `requirements.txt`: `echo "package==X.Y.Z" >> requirements.txt`
4. Freeze exact versions: `pip freeze > frozen_requirements.txt`
5. Document in `journal.md`: "Added [package] for [reason]"

**DO NOT**:
- Install system packages without user approval
- Use `pip install` inside training scripts
- Install from unofficial sources

### D. Time Estimation
Before starting work, estimate:
- **Implementation**: "2-3 hours to write code"
- **Training**: "5 mins mini-train, 6 hours full"
- **Evaluation**: "10 minutes for audit"

**Communicate format**: "Total: ~3 hours implementation + 6 hours training + 10 mins eval"

Update user if estimate is off by >50%.

---

## 5. Cost & Maintenance Protocols 💸

-   **Python**: Verified on Python 3.10+.
-   **Dependencies**: Defined in `requirements.txt`.
-   **Reproducibility**:
    -   Always run `pip freeze > frozen_requirements.txt` inside the Cluster before training.
    -   Save this alongside the logs. This proves *exactly* which library versions were used.

---

## 4. Cost & Maintenance Protocols 💸

### A. Shutdown Discipline
*   **Rule**: Turn off the VM when not in active use.
*   **Reason**: L4/A100 instances are expensive hourly. Burning credits on idle time is wasteful.

### B. Disk Hygiene (The "Crash Loop" Danger)
*   **Rule**: Aggressively delete old checkpoints (`.pt`) and huge logs.
*   **Danger**: If the disk fills up (100% Usage), the VM OS will crash and lock you out.
*   **The Fix**: You will have to spin up a second "Recovery VM", mount the old disk, and manually delete files to unbrick it. **Do not let this happen.**

### C. SSH Connectivity
*   **Scenario**: You restarted the VM (see Rule A).
*   **Symptom**: VS Code Remote / SSH fails to connect.
*   **Fix**: The **External IP** is ephemeral. It changed.
    2.  Copy the new External IP.
    3.  Update your local `~/.ssh/config` HostName.

**Critical**: Don't start a 2-day A100 job without testing on L4 first (mini-train).

---

## 6. Colab A100 Automation (No Interactive Auth) 🤖

**Problem**: Using Colab kernel from VS Code requires interactive `drive.mount()`, breaking agent workflows.

**Solution**: Use GCP service account for programmatic Drive access.

### A. One-Time Setup (User Must Do This)

**Step 1: Create Service Account**
```bash
# Set your GCP project ID
export PROJECT_ID="your-gcp-project-id"

# Create service account
gcloud iam service-accounts create colab-automation \
  --description="For automated Colab Drive access" \
  --display-name="Colab Automation"

# Download credentials
gcloud iam service-accounts keys create ~/colab-service-account.json \
  --iam-account=colab-automation@${PROJECT_ID}.iam.gserviceaccount.com
```

**Step 2: Grant Drive Access**
1. Go to Google Drive web UI
2. Create a folder: `colab_workspace/`
3. Right-click → Share
4. Add: `colab-automation@YOUR_PROJECT_ID.iam.gserviceaccount.com`
5. Grant "Editor" access

**Step 3: Store Key Securely**
```bash
# On your L4 VM, create .secrets directory
mkdir -p ~/.secrets
mv ~/colab-service-account.json ~/.secrets/
chmod 600 ~/.secrets/colab-service-account.json

# Add to .gitignore (if not already)
echo ".secrets/" >> .gitignore
```

### B. Agent Workflow (Automated)

**Upload Code to Drive** (Run on L4):
```python
# scripts/upload_to_drive.py
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load credentials
SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = os.path.expanduser('~/.secrets/colab-service-account.json')

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=creds)

# Upload experiment zip
def upload_file(local_path, drive_folder_id):
    file_metadata = {
        'name': os.path.basename(local_path),
        'parents': [drive_folder_id]  # Your colab_workspace folder ID
    }
    media = MediaFileUpload(local_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'Uploaded: {file.get("id")}')
    return file.get('id')

# Usage
upload_file('v68_lane_unification.zip', 'YOUR_FOLDER_ID')
```

**In Colab Notebook** (No Interactive Auth!):
```python
# Cell 1: Authenticate with service account
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io

# Upload your service account key to Colab (one-time)
# Option 1: Upload via Colab UI (Files panel)
# Option 2: Store in Colab Secrets (if using Colab Pro+)

SERVICE_ACCOUNT_FILE = '/content/colab-service-account.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

# Cell 2: Download experiment from Drive
from googleapiclient.http import MediaIoBaseDownload

def download_file(file_id, destination):
    request = drive_service.files().get_media(fileId=file_id)
    with open(destination, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    print(f'Downloaded to: {destination}')

download_file('YOUR_FILE_ID', '/content/v68.zip')

# Cell 3: Extract and train
!unzip /content/v68.zip -d /content/
!pip install -r /content/experiments/v68_lane_unification/requirements.txt
!python /content/experiments/v68_lane_unification/train.py

# Cell 4: Upload results back
!zip -r results.zip /content/experiments/v68_lane_unification/checkpoints/
upload_file('/content/results.zip', 'YOUR_FOLDER_ID')
```

### C. Alternative: Colab Secrets (Pro+ Only)

If you have Colab Pro+, use Secrets instead of uploading the key file:

```python
from google.colab import userdata
import json

# Store service account JSON in Colab Secrets (key: COLAB_SERVICE_ACCOUNT)
service_account_info = json.loads(userdata.get('COLAB_SERVICE_ACCOUNT'))
creds = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES)
```

### D. Security Notes
- **Never commit** `colab-service-account.json` to git
- Store in `~/.secrets/` on L4 VM
- In Colab: Upload to `/content/` (ephemeral, deleted after session) or use Secrets
- Restrict service account to only the `colab_workspace/` folder

---

## 7. Interactive Development (Jupyter) 📓

You can run Jupyter Notebooks directly on the Remote VM (Tier 1) using your local VS Code interface.

### Setup
1.  **Connect**: SSH into the VM via VS Code Remote.
2.  **Open**: Create/Open a `.ipynb` file.
3.  **Select Kernel**: Click "Select Kernel" (top right) -> "Select Another Kernel..." -> "Python Environments..."
4.  **Target**: Choose the VM's environment (e.g. `/usr/bin/python3` or `./venv/bin/python`).

### Benefit
-   You typically do **NOT** need to forward ports manually. VS Code handles the tunneling.
-   Access the L4 GPU interactively for quick tensor checks.

### Troubleshooting: "Kernel Not Found"
If VS Code cannot see the remote kernels:
1.  **The Fix**: Open Command Palette (`Ctrl+Shift+P`).
2.  **Run**: `Jupyter: Run in Dedicated Extension Host`.
3.  **Why**: Sometimes the Jupyter extension loads locally instead of remotely. This forces a reboot of the extension on the VM.

### Sanity Check
To prove you are on the VM (and not your laptop):
```python
import socket
print(socket.gethostname()) # Should match the VM name, not "MacBook"
```
