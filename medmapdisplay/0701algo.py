graph TD
    A[Input CT Images] --> B[Normalization]
    B --> C[Extract 3D Patches]
    C --> D[Convert to Tensor]
    D --> E[3D Encoder]
    E --> F[Latent Space]
    F --> G[3D Decoder]
    G --> H[Reconstructed Images]
    H --> I[Calculate Loss]
    I --> J[Training Process]

    J --> K[Save Models]

    subgraph Training Loop
        E --> I
        F --> G
        I --> E
    end

    K --> L[New CT Images]
    L --> M[Load Models]
    M --> N[Normalization]
    N --> O[Extract 3D Patches]
    O --> P[Convert to Tensor]
    P --> Q[3D Encoder (Transfer Learning)]
    Q --> R[Latent Space (New Data)]
    R --> S[3D Decoder (Transfer Learning)]
    S --> T[Reconstructed Images (New Data)]
    T --> U[Calculate Anomalies]

    subgraph Transfer Learning
        M --> Q
        Q --> R
        R --> S
        S --> T
    end