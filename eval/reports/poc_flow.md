# PoC vs vanilla inference flow

Green = untouched vanilla, orange = added by the PoC, blue = changed in role.
The PoC's claim made visual: the generation spine is stock; the only new serving
surface is `/v1/embeddings`; the "verifier" is frozen data (a linear head) read
off the last-position vector, not a system component.

```mermaid
flowchart TB
    classDef vanilla fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef added fill:#fff3e0,stroke:#ef6c00,color:#e65100,stroke-width:2px
    classDef dualrole fill:#e3f2fd,stroke:#1565c0,color:#0d47a1,stroke-width:2px

    App["PoC harness — the real contribution (orange)<br/>outer loop: DECIDE / GATE / THINK<br/>calls the model many times per task"]:::added

    subgraph API["Public API and serving — stock llama-server, unmodified"]
        RequestG["POST /v1/chat/completions<br/>prompt = frozen recipe"]:::vanilla
        Prepare["Prepare conversation"]:::vanilla
        Tokenize["Tokenizer (shared by both paths)"]:::vanilla
        Engine["Inference engine<br/>generation loop + KV cache — STOCK<br/>no logit access, no guided decoding"]:::vanilla
        Decode["Decode token IDs → text"]:::vanilla
        ResponseG["API response: message.content"]:::vanilla
        RequestE["ADDED entry: POST /v1/embeddings<br/>(evidence block, verbatim text)"]:::added
        Normalize["L2-normalize (server default)"]:::added
        ResponseE["Vector response"]:::added
    end

    subgraph Model["Neural model — ONE frozen LLM, shared by both paths, weights untouched"]
        Embed["Token embeddings + position"]:::vanilla
        Blocks["Transformer blocks × many"]:::vanilla
        Last["Vector at last position — the whole idea:<br/>was an internal stepping stone,<br/>now ALSO a product (the readout)"]:::dualrole
        Scores["Output layer → next-token scores<br/>(generation path only)"]:::vanilla
    end

    Choose["Sample or greedy next token"]:::vanilla
    Stop{"Stop token or limit?"}:::vanilla
    Probe["Linear verifier dial (orange, and it is data — not code:<br/>3840 weights + b, Platt a, c, frozen τ* = 0.326134)"]:::added
    Verdict{"p ≥ τ* ?"}:::added
    Done["Reward / label recorded"]:::added

    %% vanilla generation loop (green)
    App -->|"THINK step"| RequestG --> Prepare --> Tokenize --> Engine
    Engine --> Embed --> Blocks --> Last --> Scores --> Choose --> Stop
    Stop -->|"No: append token"| Engine
    Stop -->|"Yes"| Decode --> ResponseG --> App

    %% added verification tap (orange)
    App -->|"GATE: model claims completion →<br/>freeze evidence block"| RequestE --> Tokenize --> Engine
    Last -->|"read-only tap"| Normalize --> ResponseE --> Probe --> Verdict
    Verdict -->|"Yes: claim stands → task ends"| Done
    Verdict -->|"No: claim rejected as unverified →<br/>routed back to the outer loop"| App
```

Footnote: in the live chain the harness talks to a thin OpenAI-compatible proxy
(`omp_proxy.py`) that carries the GATE policy and forwards to the real
llama-server; it is folded into the orange harness node above. The
`Prepare → Engine` internals are the serving system's own doing.
