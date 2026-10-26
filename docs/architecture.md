# Solution architecture

```mermaid
flowchart TD
    A[Wisconsin Diagnostic dataset] --> B[Raw CSV snapshot]
    B --> C[Quality checks and data dictionary]
    C --> D[Stratified train test split]
    D --> E[Preprocessing pipelines]
    E --> F[Cross validated model comparison]
    F --> G[Selected fitted model]
    G --> H[Held out evaluation]
    H --> I[Metrics figures and model card]
```

The split occurs before any learned preprocessing. Scaling is contained inside each relevant sklearn pipeline, so every cross-validation fold learns its transformation from training observations only. The held-out test set is used once for the final evaluation.
