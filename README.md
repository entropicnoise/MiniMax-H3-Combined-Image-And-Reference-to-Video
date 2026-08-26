# MiniMax H3 Combined Image And Reference to Video

A ComfyUI custom node that combines MiniMax H3 target-frame image conditioning and Ref2VA reference conditioning in a single node.

It supports:

- `first_frame`
- `middle_frame`
- `last_frame`
- reference images
- reference videos
- reference-video audio
- standalone reference audio
- MiniMax H3 reference tags such as `<Picture 1>`, `<Video 1>`, `<Audio 1>`, and `<Subject 1>`
- T2V, I2V, Ref2V, T2I, I2I, REF2I MiniMax H3 workflows

The node is designed for workflows where target-timeline image guides and full-reference identity / motion / audio guidance need to coexist in the same MiniMax H3 conditioning payload.

> **Version:** 1.6  
> **Node:** `MiniMax H3 Combined Image And Reference to Video`

---
## Screenshot

<img width="376" height="363" alt="image" src="https://github.com/user-attachments/assets/70492cc2-a130-485a-bbb0-eac9dcd695a2" />

## Features

### Combined image + reference conditioning

The node combines the two main MiniMax H3 conditioning styles:

- **Target-frame image guides** — first, middle, and last images are placed at specific positions on the generated video's timeline.
- **Ref2VA references** — images, videos, and audio act as reference material without being assigned to a fixed output frame.
- Text to Video (when no reference or images are supplied)

with length set to 1 frame and  1-frame VAE for image workflows, it can be used in "to image" workflows (MiniMax H3: T2I, REF2I, I2I).

Both systems can be used independently or together.

### First / middle / last image guides

| Input | Position | Resize behavior |
|---|---:|---|
| `first_frame` | frame `0` | stretched to the target canvas, matching the stock H3 first-frame behavior |
| `middle_frame` | `frame_count // 2` | aspect-preserving cover crop |
| `last_frame` | `frame_count - 1` | aspect-preserving cover crop |

For a 124-frame generation:

```text
first_frame   -> frame 0
middle_frame  -> frame 62
last_frame    -> frame 123
```

At 24 fps, frame 62 is approximately 2.58 seconds into the generated clip.

`middle_frame` is optional. When it is not connected, the node keeps the normal first/last path and does not activate the legacy interior-keyframe compatibility path.

### Ref2VA media

Supported reference inputs:

- up to 10 reference images
- up to 4 reference videos
- matching audio for each reference video
- up to 4 standalone reference audios

Reference media is passed to MiniMax H3 using the same general reference-conditioning mechanisms used by ComfyUI's H3 reference node.

---

## Installation

1. Download or clone the node folder.
2. Place it inside:

```text
ComfyUI/custom_nodes/
```

The resulting structure should look like:

```text
ComfyUI/
└── custom_nodes/
    └── MiniMaxH3CombinedImageAndReference/
        ├── __init__.py
        ├── minimax_h3_combined_image_and_reference.py
        └── README.md
```

3. Restart ComfyUI.
4. Find the node under:

```text
model / conditioning / minimax
```

Node display name:

```text
MiniMax H3 Combined Image And Reference to Video
```

---

## Requirements

This node relies on MiniMax H3 support already present in ComfyUI, including the built-in H3 helpers and model implementation.

It imports functionality from:

```python
comfy_extras.nodes_minimax_h3
```

and therefore requires a ComfyUI build containing MiniMax H3 support.

No separate Python package requirements are added by this node.

---

## Inputs

### Core model inputs

| Input | Description |
|---|---|
| `clip` | MiniMax H3-compatible CLIP / text encoder input |
| `vae` | Video/image VAE used for target images and reference visual latents |
| `audio_vae` | Audio VAE used for reference audio |
| `prompt` | MiniMax H3 multimodal prompt |
| `width` | Output width |
| `height` | Output height |
| `length` | Output frame count |

`length` follows the MiniMax H3 frame grid used by ComfyUI. The node defaults to 124 frames.

---

## Target-frame image inputs

### `first_frame`

Optional opening image guide.

It is conditioned at:

```text
frame 0
```

The first frame follows the stock H3 first-frame geometry behavior and is resized directly to the requested output canvas.

### `middle_frame`

Optional interior image guide.

It is conditioned at:

```python
frame_count // 2
```

This makes it an actual target-timeline guide rather than a general reference image.

The image uses an aspect-preserving cover crop to the generated frame.

> `middle_frame` is an extended workflow. Current H3 layout implementations can place keyframes at arbitrary target-frame indices. For older ComfyUI H3 cores that only accepted endpoint indices, this node contains a compatibility path that is activated only when an interior guide is present.

### `last_frame`

Optional ending image guide.

It is conditioned at:

```python
frame_count - 1
```

The image uses an aspect-preserving cover crop.

---

## Reference image resize modes

`reference_resize_mode` controls how `ref_images` are resized before reference encoding.

All modes:

- preserve the reference image aspect ratio
- avoid intentional upscaling
- round dimensions to the H3 canvas multiple

### `resize to frame short edge`

Downscales the reference until its **short edge** fits the generated frame's short edge.

Conceptually:

```python
scale = min(
    1.0,
    min(frame_width, frame_height) / min(reference_width, reference_height)
)
```

Useful when reference size should be bounded by the target frame's smaller dimension.

### `resize to start frame long edge`

Downscales the reference until its **long edge** fits the generated frame's long edge.

Conceptually:

```python
scale = min(
    1.0,
    max(frame_width, frame_height) / max(reference_width, reference_height)
)
```

This can be a stricter fit when the reference and output use different aspect ratios.

### `resize to start frame total_pixels`

This is the original node's **`match`** behavior.

```python
scale = min(
    1.0,
    sqrt(
        (frame_width * frame_height)
        / (reference_width * reference_height)
    )
)
```

Meaning:

> Preserve the reference aspect ratio, never intentionally upscale it, and keep its total reference-token / latent area roughly around the scale of one generated frame.

This is the default mode.

### `resize MiniMax H3 2k resolution`

This is the original node's **`max`** behavior.

```python
scale = min(
    1.0,
    REF_IMAGE_SHORT_EDGE / min(reference_width, reference_height)
)
```

In the ComfyUI H3 implementation this node was based on, `REF_IMAGE_SHORT_EDGE` is 2048 px.

Meaning:

> Keep the reference as large as possible, only downscaling when its short edge exceeds the MiniMax H3 reference-image short-edge limit.

This can produce significantly larger reference latents and may use more VRAM and sampling time.

---

## Prompting MiniMax H3

MiniMax H3 prompts can refer directly to supplied reference media.

Common tags include:

```text
<Picture 1>
<Picture 2>
<Video 1>
<Audio 1>
<Subject 1>
<Subject 2>
```

Keep numbering consistent with the order in which references are supplied to the node.

For example:

```text
<Subject 1> is the person shown in <Picture 1>.
<Subject 2> is the person shown in <Picture 2>.
```

or:

```text
Use the appearance of <Picture 1> while following the motion demonstrated in <Video 1>.
```

---

## Recommended prompt structures

### 1. T2VA / I2VA / FL2VA / L2VA

Recommended section order:

```text
integrated_multimodal_description
→ overall_soundscape
→ non_diegetic_music
```

#### `integrated_multimodal_description`

Describe the complete visual generation:

- subjects
- appearance
- action
- environment
- camera
- framing
- timing
- image references
- video references
- important continuity instructions

Example structure:

```text
integrated_multimodal_description:
<Subject 1> walks through a rainy neon-lit street at night.
The camera tracks backward at chest height while the subject approaches.
Maintain the facial identity and clothing from <Picture 1>.
The movement is natural and continuous from the opening frame to the ending frame.
```

#### `overall_soundscape`

Describe diegetic audio:

- dialogue
- footsteps
- ambience
- environmental sound
- object sounds
- room tone
- weather
- crowds
- vehicles

Example:

```text
overall_soundscape:
Soft rainfall, distant traffic, wet footsteps, quiet city ambience.
```

#### `non_diegetic_music`

Describe music or score that does not originate from the visible scene.

Example:

```text
non_diegetic_music:
Slow atmospheric electronic score with restrained bass and soft synth pads.
```

---

### 2. Full-reference Ref2VA

Recommended section order:

```text
subject_definitions
→ summary
→ retention_analysis
→ detailed_description
→ overall_soundscape
→ non_diegetic_music
```

#### `subject_definitions`

Define referenced subjects and connect them to reference media.

Example:

```text
subject_definitions:
<Subject 1> is the woman shown in <Picture 1>.
<Subject 2> is the man shown in <Picture 2>.
```

#### `summary`

Give the concise overall intent of the generated shot.

Example:

```text
summary:
<Subject 1> and <Subject 2> meet in a quiet railway station and briefly speak before boarding a train.
```

#### `retention_analysis`

State which properties from the supplied references should remain recognizable.

Typical retention targets:

- identity
- face
- hairstyle
- clothing
- body proportions
- voice
- motion style
- object appearance
- environmental traits
- visual style

Example:

```text
retention_analysis:
Preserve <Subject 1>'s facial identity, hairstyle, coat, and overall proportions from <Picture 1>.
Preserve <Subject 2>'s identity and voice characteristics from the supplied references.
Use <Video 1> primarily as motion guidance rather than as a fixed scene layout.
```

#### `detailed_description`

Describe the complete generated scene in detail:

- action sequence
- camera movement
- blocking
- expressions
- interaction
- timing
- lighting
- environment
- reference usage
- continuity

#### `overall_soundscape`

Describe dialogue and scene audio.

#### `non_diegetic_music`

Describe background score or music.

---

## Reference ordering

The node's autogrow inputs are numbered.

Examples:

```text
ref_image_1
ref_image_2
ref_image_3
```

correspond to prompt references such as:

```text
<Picture 1>
<Picture 2>
<Picture 3>
```

Likewise, reference videos are presented in numbered order.

When using several references, keep the prompt explicit about what each reference contributes.

---

## Reference video audio

A reference video's soundtrack can be supplied using the matching numbered audio input.

For example:

```text
ref_video_1
ref_video_audio_1
```

The node associates the audio with the same-numbered reference video.

Standalone reference audio can also be supplied separately through the `ref_audios` inputs.

---

## Outputs

The node returns:

| Output | Description |
|---|---|
| `positive` | MiniMax H3 conditioning containing prompt, keyframe, and/or reference information |
| `LATENT` | Empty audio/video latent prepared for sampling |

Connect these to the normal MiniMax H3 sampling workflow.

---

## Compatibility behavior

The node contains narrowly scoped compatibility handling for older MiniMax H3 implementations in ComfyUI.

### Keyframes + Ref2VA references

Older H3 cores could build keyframe visual latents and then overwrite them when Ref2VA references were also present.

The node repairs the combined payload so the latent order remains:

```text
target keyframes
→ reference visual latents
```

This compatibility behavior is scoped to conditioning generated by this node.

### Target timeline alignment

Older H3 `PackedLayout` versions could place first/last keyframes relative to the wrong temporal origin when Ref2VA references were also present.

For the normal first/last path, the node preserves the existing target-origin alignment repair when references are used.

### Interior `middle_frame`

Some older H3 layout implementations accepted only first/last endpoint indices.

When `middle_frame` is present, the node conditionally uses a compatibility path that:

1. lets the legacy layout allocate the guide rows
2. identifies the generated target video's temporal origin
3. places the guide rows at their requested frame coordinates

When `middle_frame` is not supplied, this interior-keyframe path is not activated.

On newer ComfyUI H3 implementations with native arbitrary keyframe positioning, no interior layout patch is required.

---

## Example configurations

### Text + references only

```text
first_frame:   empty
middle_frame:  empty
last_frame:    empty

ref_image_1:   character reference
ref_video_1:   motion reference
```

Use the full-reference Ref2VA prompt structure.

### First frame + references

```text
first_frame:   opening composition
middle_frame:  empty
last_frame:    empty

ref_image_1:   identity reference
```

The opening composition is fixed to target frame 0 while the reference image provides non-positional identity guidance.

### First + last frame

```text
first_frame:   opening shot
middle_frame:  empty
last_frame:    ending shot
```

Use this for normal endpoint-guided H3 generation.

### First + middle + last

```text
first_frame:   opening shot
middle_frame:  intermediate composition
last_frame:    ending shot
```

This provides three target-timeline visual anchors.

### First + middle + last + Ref2VA

```text
first_frame:   opening composition
middle_frame:  midpoint composition
last_frame:    ending composition

ref_image_1:   subject identity
ref_video_1:   desired motion
ref_audio_1:   voice / audio reference
```

This is the most strongly conditioned configuration and may require more VRAM and careful prompting.

---

## Notes

- Target-frame images and Ref2VA reference images serve different purposes.
- `first_frame`, `middle_frame`, and `last_frame` are placed on the generated target timeline.
- `ref_images` are non-positional references used for identity / appearance / style guidance.
- `ref_videos` can supply motion and multimodal reference information.
- Larger reference image modes create larger reference latents and can increase resource usage.
- `resize to start frame total_pixels` is the renamed original `match` behavior.
- `resize MiniMax H3 2k resolution` is the renamed original `max` behavior.
- The node can be used with either keyframes or references alone; neither group requires the other.

---

## Troubleshooting

### Different results after changing reference resize mode

Check that:

```text
reference_resize_mode = resize to start frame total_pixels
```

if you want the behavior previously called `match`.

The other resize modes intentionally produce different reference dimensions and can substantially change the generated result.

### Error involving `PackedLayout`

Make sure you are using the latest version of this node.

Version 1.6 includes the missing interior-keyframe marker required by the legacy middle-frame compatibility path.

### High VRAM use with reference images

Try:

```text
resize to start frame total_pixels
```

instead of:

```text
resize MiniMax H3 2k resolution
```

The 2K reference mode can create much larger visual reference latents.

### Middle frame has too much influence

`middle_frame` is a target-timeline keyframe rather than a loose reference.

If you want appearance or identity guidance without fixing an image near the center of the target timeline, use `ref_images` instead.

---

## Version history

### v1.6

- fixed missing `_H3_INTERIOR_KEYFRAME_KEY` marker
- retained conditional legacy support for `middle_frame`

### v1.5

- made legacy interior-keyframe handling conditional on `middle_frame`
- preserved the earlier first/last + Ref2VA target-origin compatibility repair

### v1.4

- added `middle_frame`
- placed the middle guide at `frame_count // 2`
- added legacy arbitrary-keyframe compatibility handling

### v1.3

- renamed the node to `MiniMax H3 Combined Image And Reference to Video`

### v1.2

- expanded the prompt tooltip
- documented reference tags and recommended MiniMax H3 prompt section structures

### v1.1

- renamed `ref_image_size` to `reference_resize_mode`
- added four explicit reference-image resize modes
- renamed original `match` and `max` behaviors to descriptive labels

### v1.0

- split the node into a minimal `__init__.py` and implementation module

---

## Node mapping

```python
from .minimax_h3_combined_image_and_reference import MiniMaxH3CombinedImageAndReferenceToVideo

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CombinedImageAndReferenceToVideo": MiniMaxH3CombinedImageAndReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CombinedImageAndReferenceToVideo":
        "MiniMax H3 Combined Image And Reference to Video",
}
```

---

## Disclaimer

This is a custom ComfyUI node built around ComfyUI's MiniMax H3 implementation.

The `middle_frame` workflow extends the usual first/last endpoint interface by using H3 target keyframe positioning at an interior frame index. Behavior can depend on the MiniMax H3 implementation present in the installed ComfyUI version.
