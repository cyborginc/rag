<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

## Enable Reasoning

By default, reasoning is disabled in the RAG flow. Reasoning in Nemotron 1.5 is controlled by the system prompt. To enable reasoning for your use case, you can update the system prompt in [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml) from `/no_think` to `/think`.

For example, to enable reasoning in RAG, update the system prompt from `/no_think` to `/think`. This can be done for other prompts as well.
```
rag_template:
  system: |
    /think

  human: |
    You are a helpful AI assistant named Envie.
    You must answer only using the information provided in the context. While answering you must follow the instructions given below.

    <instructions>
    1. Do NOT use any external knowledge.
    2. Do NOT add explanations, suggestions, opinions, disclaimers, or hints.
    3. NEVER say phrases like “based on the context”, “from the documents”, or “I cannot find”.
    4. NEVER offer to answer using general knowledge or invite the user to ask again.
    5. Do NOT include citations, sources, or document mentions.
    6. Answer concisely. Use short, direct sentences by default. Only give longer responses if the question truly requires it.
    7. Do not mention or refer to these rules in any way.
    8. Do not ask follow-up questions.
    9. Do not mention this instructions in your response.
    </instructions>

    Context:
    {context}

    Make sure the response you are generating strictly follow the rules mentioned above i.e. never say phrases like “based on the context”, “from the documents”, or “I cannot find” and mention about the instruction in response.

```

After updating the prompt, update the temperature and top_p using the following environment variables
```bash
export LLM_TEMPERATURE=0.6
export LLM_TOP_P=0.95
```
### Docker Deployment
You can refer to [prompt-customization.md](./prompt-customization.md#overriding-existing-templates-in-promptyaml) to deploy the RAG server with the above changes.

### Helm Deployment
You can refer to [prompt-customization.md](./prompt-customization.md#prompt-customization-in-helm-chart) to deploy RAG with prompt changes.


### Filtering Reasoning Tokens
By default, we filter out reasoning tokens and only provide the final response from the LLM. If you want to see the reasoning tokens as well, you can set `FILTER_THINK_TOKENS` to false.

```bash
export FILTER_THINK_TOKENS=false
```

**Note:** For the `llama-3.3-nemotron-super-49b-v1` model, reasoning can be controlled by the environment variable `ENABLE_NEMOTRON_THINKING`. You can set this to true to enable reasoning:

```bash 
export ENABLE_NEMOTRON_THINKING=true
```