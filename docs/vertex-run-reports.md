# Vertex Run Reports

`avw vertex-run-report` writes a compact JSON report from Vertex CustomJob
metadata. It keeps the fields useful for run review while omitting project IDs,
bucket names, service accounts, and generated media.

## From A Saved Job Payload

Export a CustomJob payload with `gcloud`, then summarize it:

```bash
gcloud ai custom-jobs describe "$VERTEX_CUSTOM_JOB_NAME" \
  --region "$VERTEX_REGION" \
  --format json > runs/example/reports/vertex-job.json

avw vertex-run-report \
  --job-json runs/example/reports/vertex-job.json \
  --out runs/example/reports/vertex-run-report.json
```

## From A Live Job Name

The command can also call `gcloud` directly:

```bash
avw vertex-run-report \
  --job-name "$VERTEX_CUSTOM_JOB_NAME" \
  --region "$VERTEX_REGION" \
  --out runs/example/reports/vertex-run-report.json
```

## Include Output Metadata

Attach backend metadata created by `avw export-backend-metadata`:

```bash
avw export-backend-metadata \
  --input runs/example/output/reports/example_manifest.json \
  --out runs/example/reports/backend-metadata.json

avw vertex-run-report \
  --job-json runs/example/reports/vertex-job.json \
  --output-metadata runs/example/reports/backend-metadata.json \
  --out runs/example/reports/vertex-run-report.json
```

## Include Logs

If you have exported Vertex logs as a JSON array or JSONL file, pass them with
`--logs-json`. The report stores severity counts, first/last timestamps, last
message, and last error after redacting private resource references.

```bash
avw vertex-run-report \
  --job-json runs/example/reports/vertex-job.json \
  --logs-json runs/example/reports/vertex-logs.json \
  --out runs/example/reports/vertex-run-report.json
```

## Report Contents

The report includes:

- CustomJob ID and location without project ID;
- display name and state;
- create/start/end/update timestamps;
- duration when start and end times are available;
- worker pool machine type, accelerator type/count, replica count, and disk;
- optional backend output summary;
- optional log summary.

Generated videos, source images, object URIs, bucket names, service account
identities, and full Vertex resource names are not included.
