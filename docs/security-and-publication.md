# Security And Publication

Before publishing:

```bash
avw scan-publication .
```

The scan fails on:

- private key blocks;
- service account JSON markers;
- API token-looking assignments;
- Google Cloud Storage URIs;
- hardcoded GCP-style project IDs;
- absolute `/home/...` paths;
- generated image or video files;
- LoRA checkpoints and other model artifact files.

## Cloud Authentication

Use OAuth or application default credentials locally:

```bash
gcloud auth application-default login
gcloud config set project "$GOOGLE_CLOUD_PROJECT"
```

Do not commit:

- service account JSON files;
- `.env`;
- bucket names tied to private work;
- generated outputs;
- LoRA checkpoints;
- source datasets.

## Google Cloud Account Setup

For a fresh Google Cloud account, use the account owner to:

1. Create or select a project.
2. Enable Vertex AI, Cloud Storage, Artifact Registry, Cloud Build and Secret Manager as needed.
3. Create a bucket for experiments.
4. Authenticate locally with OAuth.
5. Keep tokens and ADC files outside the repo.
