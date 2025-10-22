<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Get your API Keys

## NGC API Key

You need to generate an API key to access NVIDIA NIM Microservices from the [NVIDIA RAG Blueprint](readme.md). 
You need an API key to access models hosted in the NVIDIA API Catalog, and to download models on-premises. 
For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#ngc-api-keys).

To generate an API key, use the following procedure.

1. Go to https://org.ngc.nvidia.com/setup/api-keys.
2. Click **Generate Personal Key**.
3. Enter a **Key Name**.
4. For **Expiration**, choose **Never Expire**.
5. For **Services Included**, select **NGC Catalog** and **Public API Endpoints**.
6. Click **Generate Personal Key**.
7. Copy your key and save it somewhere safe and private.
8. (Important) Export your key as an environment variable by using the following code.

    ```bash
    export NGC_API_KEY="<your-ngc-api-key>"
    ```

## CyborgDB API Key

You need to generate an API key to authenticate requests and determine your service capabilities with CyborgDB. For more information, refer to [CyborgDB API Keys](https://docs.cyborg.co/versions/v0.12.x/intro/get-api-key)

To generate an API key, use the following procedure.

1. Log in or sign up for CyborgDB at https://cyborgdb.co/
2. In your dashboard, click the **Add** button next to "API Keys"

After you generate your key, export your key as an environment variable by using the following code.

```bash
export CYBORGDB_API_KEY="<your-cyborgdb-api-key>"
```

## API Key Expiration

If your API key expires, do one of the following:

- Create a new key by using the previous procedure, and then delete the expired key. 
- Use the **Action** menu to **Rotate** your key. (NGC API Key only)

You must update the new key information in your environment variables and code.

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
