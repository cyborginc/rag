<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Multimodal Input Guide for NVIDIA RAG

## Overview

NVIDIA RAG now supports **multimodal input**, allowing users to include both text and images directly in their queries. This feature enables the system to process user-uploaded images alongside retrieved visual content from the knowledge base, providing comprehensive visual understanding capabilities.

### Key Capabilities

- **Direct Image Input**: Users can upload images directly with their text queries
- **Dual Visual Processing**: Handles both user-provided images and knowledge base images simultaneously  
- **OpenAI-Compatible API**: Full compatibility with OpenAI's multimodal message format

---

## How Multimodal Input Works

The multimodal input feature operates through a sophisticated pipeline that seamlessly integrates visual and textual understanding:

1. **Input Processing**: When you submit a query containing both text and images, the system first parses and validates the multimodal content according to OpenAI's message format.

2. **Retrieval**: We use VLM embedding model to fetch relevant document/images. To create query we merge text and base64 image to create a retriever query.

3. **Vision Language Model (VLM) Processing**: We send retrieved document with our query and image to VLM to generate final response
---

## Advantages of Multimodal Input

### Enhanced Accuracy
- **Visual Context**: Eliminates ambiguity by allowing users to show exactly what they're referring to
- **Comprehensive Analysis**: Combines visual and textual information for more complete understanding
- **Reduced Misinterpretation**: Images provide clear context that text alone might not convey effectively

### Improved User Experience
- **Natural Interaction**: Users can communicate the way they naturally think - combining words and visuals
- **Faster Query Resolution**: No need to describe complex visual elements in text

### Technical Benefits
- **OpenAI Compatibility**: Easy integration with existing applications using OpenAI's API format
- **Scalable Processing**: Handles multiple images simultaneously with configurable limits
- **Flexible Input**: Supports various image formats and encoding methods

---


## Message Format

### OpenAI-Compatible Schema

NVIDIA RAG follows the OpenAI multimodal message format:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What do you see in this image?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
        "detail": "high"
      }
    }
  ]
}
```

### Content Types

#### Text Content
```json
{
  "type": "text",
  "text": "Your text query here"
}
```

#### Image Content
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,<base64-data>",
    "detail": "auto|low|high"
  }
}
```

### Supported Image Formats

- **Data URIs**: `data:image/jpeg;base64,<base64-data>`
- **Base64 Strings**: Direct base64-encoded image data
- **Image Formats**: JPEG, PNG, GIF, WebP, BMP
- **Size Limit**: Up to 20MB per image

---

## Usage Examples

### Server Mode (API)

#### Basic Multimodal Query

```bash
curl -X POST "http://localhost:8081/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Analyze this chart and explain the trends you see."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
              "detail": "high"
            }
          }
        ]
      }
    ],
    "use_knowledge_base": true,
    "enable_vlm_inference": true,
    "temperature": 0.2
  }'
```

---

## Configuration

### Required Environment Variables

```bash
# Enable VLM inference (required for multimodal processing)
export ENABLE_VLM_INFERENCE="true"

# VLM model configuration
export APP_VLM_MODELNAME="nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"

# Image processing limits
export APP_VLM_MAX_TOTAL_IMAGES="4"      # Total images (query + context)
export APP_VLM_MAX_QUERY_IMAGES="1"      # User-provided images
export APP_VLM_MAX_CONTEXT_IMAGES="1"    # Knowledge base images
```

### Optional Configuration

```bash
# VLM response reasoning (quality gate)
export ENABLE_VLM_RESPONSE_REASONING="true"

# Use VLM response as final answer (bypass LLM)
export APP_VLM_RESPONSE_AS_FINAL_ANSWER="false"
```

### API Parameters

When making requests, you can override default settings:

```json
{
  "enable_vlm_inference": true,
  "vlm_model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
  "vlm_endpoint": "http://vlm-ms:8000/v1"
}
```

---
