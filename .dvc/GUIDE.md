## 📦 DVC Overview

We use DVC because Git is not designed to handle large files (like raw images or trained model weights). DVC stores the actual data in **Google Drive** and keeps small `.dvc` metadata files in our Git repository.

### Key Files in this Project:

- **`.dvc/config`**: Contains the public remote URL for Google Drive.
- **`.dvc/config.local`**: Stores your private Google OAuth credentials (Client ID and Secret). **Never commit this file to Git**.
- **`data.dvc` / `models.dvc**`: Metadata files that tell DVC which version of the data/models to pull from the cloud.

---

## 🚀 Getting Started

### 1. Setup Your Credentials

Before you can pull or push data, you must link your local environment to the Google Cloud Project (**MLOPS**).

1. Obtain the **Client ID** and **Client Secret** from the Google Cloud Console.
2. Add them to your local (private) DVC configuration:

```bash
dvc remote modify --local storage gdrive_client_id YOUR_ID_HERE
dvc remote modify --local storage gdrive_client_secret YOUR_SECRET_HERE

```

### 2. Initial Authentication

Run a status check to trigger the Google login flow:

```bash
dvc status -c

```

- **Action Required**: A browser window will open. If you see "Google hasn't verified this app," click **Advanced** -> **Go to MLOPS (unsafe)** to grant permission.

---

## 🛠 Common Workflow

### Pulling Data (Getting started on a new machine)

To download all datasets and models currently tracked in the project:

```bash
dvc pull

```

### Adding New Data

If you add new files to the `data/raw/` or `models/` directories, tell DVC to track them:

```bash
dvc add data/raw/
git add data/raw.dvc .gitignore
git commit -m "Add new raw pneumonia images"

```

### Pushing Changes

After committing your `.dvc` files to Git, upload the actual data to Google Drive:

```bash
dvc push

```

---

## 🔍 Troubleshooting

| Error                          | Solution                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| **403: access_denied**         | Ensure your email is added to the **Test Users** (Público) list in the Google Cloud Console.      |
| **invalid_client**             | Your Client ID and Secret do not match. Re-copy them from the "Credenciales" tab in Google Cloud. |
| **Google Drive API disabled**  | Follow the link in the error message and click **ENABLE** for the Google Drive API.               |
| **Remote 'storage' not found** | Run `dvc remote list` to verify the remote name is correct.                                       |

---

## 📁 Project Data Map

- **`data/raw/`**: Original, immutable images.
- **`data/processed/`**: Cleaned and transformed data.
- **`models/`**: Serialized model weights (`.pt`, `.h5`, etc.).
