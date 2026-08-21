# Image Generation

`ImageProvider` (RFC §25) is the surface for an agent that draws. It is
deliberately **separate from the AI provider**, the way STT and TTS already
are: the model holding a conversation is rarely one that draws, so an agent
conversing through Anthropic draws with a Gemini, OpenAI, or xAI key. Nothing
about `AIResponse` changes — images never ride the conversational response.

```python
from roomkit.providers.openai import OpenAIImageConfig, OpenAIImageProvider

images = OpenAIImageProvider(
    OpenAIImageConfig(api_key="sk-...", model="gpt-image-2", quality="high")
)
results = await images.generate("an origami fox", size="1024x1024", n=1)

results[0].data          # "data:image/png;base64,..." — always a data URI
results[0].mime_type     # "image/png"
results[0].decoded()     # raw bytes
results[0].to_image_part()  # AIImagePart — AI input, or the next edit's reference
await images.close()
```

`generate()` returns **exactly `n` results or raises** — never fewer without
an error. A size the model cannot produce raises rather than silently becoming
another geometry.

## Providers

| Provider | Import | Extra | Notes |
|---|---|---|---|
| `OpenAIImageProvider` | `roomkit.providers.openai` | `roomkit[openai]` | `/v1/images`; `gpt-image-2` takes near-arbitrary sizes, older models a fixed menu — the vendor judges |
| `GeminiImageProvider` | `roomkit.providers.gemini` | `roomkit[gemini]` | Interactions API; size translated to aspect ratio + tier (`512`/`1K`/`2K`/`4K`) |
| `XAIImageProvider` | `roomkit.providers.xai` | `roomkit[xai]` | Grok Imagine; size translated to ratio + `1k`/`2k`; edits go as JSON to `/v1/images/edits` |
| `OpenRouterImageProvider` | `roomkit.providers.openrouter` | `roomkit[openrouter]` | OpenRouter's Image API — 40+ aggregated models (Seedream, FLUX, Recraft, ...); billed cost surfaces as `usage["cost"]` |
| `AzureImageProvider` | `roomkit.providers.azure` | `roomkit[azure]` | Azure OpenAI deployment; `model` is the deployment name |
| `MockImageProvider` | `roomkit.providers.image` | none | Draws a real 1×1 PNG; records `calls`; whole path testable offline |

Configs are per-vendor (`OpenAIImageConfig`, `GeminiImageConfig`,
`XAIImageConfig`, `OpenRouterImageConfig`, `AzureImageConfig`) and carry the
vendor knobs — OpenAI's `quality`/`background`/`output_format`, Gemini's
`image_size`/`output_mime_type`, xAI's `quality`/`resolution`, Azure's
`azure_endpoint`/`api_version`. The `generate()` call is identical everywhere.

```python
from roomkit.providers.xai import XAIImageConfig, XAIImageProvider

images = XAIImageProvider(XAIImageConfig(api_key="xai-..."))  # grok-imagine-image-2.0

from roomkit.providers.openrouter import OpenRouterImageConfig, OpenRouterImageProvider

images = OpenRouterImageProvider(
    OpenRouterImageConfig(api_key="sk-or-...", model="bytedance-seed/seedream-5-0-pro")
)

from roomkit.providers.azure import AzureImageConfig, AzureImageProvider

images = AzureImageProvider(
    AzureImageConfig(
        api_key="...",
        azure_endpoint="https://myresource.openai.azure.com",
        model="my-gpt-image-deployment",
    )
)
```

## Editing

Editing is `generate()` with `reference_images`, not a second method:

```python
[original] = await images.generate("an origami fox")
[edited] = await images.generate(
    "make the paper blue",
    reference_images=[original.to_image_part()],
)
```

Each provider absorbs its vendor's split (OpenAI: multipart `images.edit`;
Gemini: same-call content; xAI: JSON `/images/edits`; OpenRouter:
`input_references` on the same request). A provider that cannot edit reports
`supports_editing == False` and **raises** on references rather than quietly
generating from the prompt alone. `XAIImageProvider` answers per model:
`grok-imagine-image-2.0` and `-quality` edit, the base `grok-imagine-image`
does not.

## The data-URI invariant

`ImageResult.data` is always `data:<mime>;base64,<payload>` — never bare
base64, never a remote URL. `MediaContent.url` accepts data URIs, so a
generated image enters a room without conversion:

```python
from roomkit.models.event import MediaContent

await kit.send_event(
    room_id=room_id,
    channel_id="bot",
    content=MediaContent(url=result.data, mime_type=result.mime_type),
    addressed_to=[],  # the picture answers the turn; unaddressed, it would
                      # re-enter as a fresh prompt and the agent would draw again
)
```

The usual wiring is a tool: the `AIChannel` calls a `generate_image` tool, the
handler draws through the `ImageProvider` and posts the `MediaContent`, and
returns *text* to the model ("Drew 1 image"). See
`examples/image_generation.py` for the complete runnable version.

## Usage and cost

`ImageResult.usage` carries up to four **disjoint** counters — `input_tokens`,
`input_image_tokens`, `output_tokens`, `output_image_tokens` — each token
counted exactly once; counters a vendor does not report are absent, not zero.

- OpenAI and Gemini meter per token: their catalog entries carry
  `ModelPricing` with `image_input_per_million` / `image_output_per_million`,
  and `entry.pricing.cost_for(result.usage)` prices a generation in USD.
- xAI and most of OpenRouter's lineup bill a flat amount per image: their
  catalog entries deliberately carry **no** `pricing` (a per-image charge
  restated per token would be wrong, not missing). OpenRouter reports what it
  billed on every call, surfaced as `result.usage["cost"]` (USD).

## Catalogs

`available_models()` (classmethod, offline) lists each provider's image
models, tagged `capabilities=["image_gen", "edit"]`. The image catalog is
**disjoint** from `AIProvider.available_models()` — no id draws *and*
converses. `catalog_entry()` returns the active model's `ModelInfo` or `None`.
Azure returns an empty catalog (deployments are user-named); OpenRouter's is a
curated slice — the live set is OpenRouter's public
`GET /api/v1/images/models`.

## Testing

```python
from roomkit.providers.image import MockImageProvider

images = MockImageProvider()                      # or MockImageProvider(images=[png_bytes])
[result] = await images.generate("anything")
assert result.decoded().startswith(b"\x89PNG")
assert images.calls == [("anything", None, 1, [])]

MockImageProvider(supports_editing=False)         # raises on reference_images
```
