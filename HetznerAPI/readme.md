# Hetzner LLM Inference

At time of writing, Hetzner has introduced an inference API, in the experimental phase. Currently, it offers free use of multiple open source models, though this is subject to change.

Included in this repo is some examples of how these would be used to complete querys, using the openai SDK.

UPDATE - Hetzner has announced that the experiment will end on Nov 10th 2026. They will be phasing out the Qwen models, and no new tokens are being intrduced. I'll be keeping this around as reference, but the links will no longer function

If you are using these links, you will need to switch from

```
https://inference.hetzner.com/api/v1
```
to 
```
https://inference.hetzner.com/v1
```