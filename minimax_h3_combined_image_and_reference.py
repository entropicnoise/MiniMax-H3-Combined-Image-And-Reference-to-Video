import math
import logging
import inspect
from functools import wraps


import nodes
import node_helpers
from comfy_api.latest import io

# Re-use the exact helpers/constants the built-in H3 nodes use.
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent,
    _resize,
    adapt_canvas,
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    FPS,
)

# ComfyUI compatibility:
# - older H3 builds expose this as MiniMaxH3ReferenceToVideo._encode_ref_audio
# - newer builds expose it directly as a module-level helper
try:
    from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
except ImportError:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    _encode_ref_audio = MiniMaxH3ReferenceToVideo._encode_ref_audio


_H3_PAYLOAD_PATCH_MARKER = "_minimax_h3_keyframe_ref_merge_patch_v2"
_H3_LAYOUT_PATCH_MARKER = "_minimax_h3_combined_keyframe_layout_patch_v2"
_H3_TARGET_ANCHOR_KEY = "_minimax_h3_combined_target_anchor"
_H3_ACTUAL_FRAME_INDEX_KEY = "_minimax_h3_combined_actual_frame_index"
_H3_INTERIOR_KEYFRAME_KEY = "_minimax_h3_combined_interior_keyframe"
_LOG = logging.getLogger("ComfyUI.MiniMaxH3CombinedImageAndReference")


def _is_our_combined_keyframes(keyframes):
    return bool(keyframes) and any(
        isinstance(kf, dict) and kf.get(_H3_TARGET_ANCHOR_KEY, False)
        for kf in keyframes
    )


def _ensure_h3_keyframe_ref_merge():
    """Let this pack's keyframes and Ref2VA refs coexist on older ComfyUI.

    Older MiniMaxH3.extra_conds builds keyframe visual latents first, then the
    refs branch overwrites that list. PackedLayout still contains both sets of
    fixed rows, so sampling either crashes or assigns the wrong latent to the
    wrong rows. Rebuild the list in layout order: keyframes first, refs second.

    This wrapper is deliberately scoped to keyframes produced by this pack.
    """
    try:
        import comfy.model_base as model_base
    except Exception as e:
        _LOG.warning("MiniMax H3 payload patch: could not import comfy.model_base: %s", e)
        return False

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("MiniMax H3 payload patch: MiniMaxH3.extra_conds not found.")
        return False

    current = getattr(cls, "extra_conds")
    if getattr(current, _H3_PAYLOAD_PATCH_MARKER, False):
        return True

    @wraps(current)
    def _patched_extra_conds(self, **kwargs):
        out = current(self, **kwargs)

        keyframes = kwargs.get("minimax_keyframes", None)
        refs = kwargs.get("minimax_refs", None)
        if not refs or not _is_our_combined_keyframes(keyframes):
            return out

        cond = out.get("minimax_payload", None) if isinstance(out, dict) else None
        payload = getattr(cond, "cond", None) if cond is not None else None
        if not isinstance(payload, dict):
            _LOG.warning(
                "MiniMax H3 payload patch: could not access minimax_payload; "
                "combined keyframes+refs may still fail."
            )
            return out

        payload["cond_video_latents"] = (
            [kf["latent"] for kf in keyframes if isinstance(kf, dict) and "latent" in kf]
            + [ref["latent"] for ref in refs if isinstance(ref, dict) and "latent" in ref]
        )
        payload["cond_audio_latents"] = [
            ref["audio_latent"]
            for ref in refs
            if isinstance(ref, dict) and ref.get("audio_latent") is not None
        ]

        frame_count = kwargs.get("minimax_frame_count", None)
        if frame_count is not None:
            payload["frame_count"] = frame_count

        return out

    setattr(_patched_extra_conds, _H3_PAYLOAD_PATCH_MARKER, True)
    cls.extra_conds = _patched_extra_conds
    _LOG.info("MiniMax H3 payload patch: enabled keyframe+reference latent merge.")
    return True


def _ensure_h3_keyframe_ref_layout():
    """Keep stock first/last behavior and add interior-guide support on legacy H3 cores.

    Current ComfyUI accepts arbitrary ``resolved_frame_index`` values directly,
    so this patch is not installed there.

    On older H3 cores, the wrapper has two deliberately separate paths:

    1. No interior guide (normal first/last use): call stock ``PackedLayout``
       unchanged. Only when Ref2VA refs are also present do we apply the same
       target-origin offset repair used before middle-frame support was added.

    2. Interior guide present: temporarily present interior guides as frame 0 so
       legacy ``PackedLayout`` can allocate their rows, then place all of this
       node's guide rows at their exact requested target-frame coordinates.

    This keeps ``middle_frame`` compatibility conditional per invocation rather
    than changing the ordinary first/last path.
    """
    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:
        _LOG.warning("MiniMax H3 layout patch: could not import minimax model: %s", e)
        return False

    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        _LOG.warning("MiniMax H3 layout patch: PackedLayout not found.")
        return False

    current = getattr(cls, "__init__", None)
    if current is None:
        return False
    if getattr(current, _H3_LAYOUT_PATCH_MARKER, False):
        return True

    try:
        params = inspect.signature(current).parameters
    except (TypeError, ValueError):
        params = {}

    # Current ComfyUI natively uses:
    # cond_t = target_origin + FRAME_RESCALE * resolved_frame_index.
    if "frame_count" not in params:
        _LOG.info(
            "MiniMax H3 layout patch: native arbitrary target keyframes detected; "
            "no patch needed."
        )
        return True

    frame_rescale = float(getattr(mm, "FRAME_RESCALE", 5.0 / 3.0))

    @wraps(current)
    def _patched_layout_init(
        self, text_len, latent_t, latent_h, latent_w, audio_t,
        keyframes=None, refs=None, frame_count=None,
    ):
        ours = _is_our_combined_keyframes(keyframes)
        has_interior = bool(
            ours and keyframes and any(
                isinstance(kf, dict) and kf.get(_H3_INTERIOR_KEYFRAME_KEY, False)
                for kf in keyframes
            )
        )

        # Ordinary first/last-only invocation: pass exactly the original data to
        # stock PackedLayout. This is the normal path whenever middle_frame is empty.
        if not has_interior:
            current(
                self, text_len, latent_t, latent_h, latent_w, audio_t,
                keyframes=keyframes, refs=refs, frame_count=frame_count,
            )

            # Preserve the pre-v1.4 legacy compatibility fix only for the case
            # that needs it: this node's first/last guides combined with Ref2VA.
            if not refs or not ours or not keyframes:
                return

            segments = getattr(self, "segments", None)
            position_ids = getattr(self, "position_ids", None)
            if not segments or position_ids is None:
                raise RuntimeError(
                    "MiniMax H3 combined conditioning: PackedLayout no longer exposes "
                    "segments/position_ids; cannot safely align first/last guides to the target."
                )

            target = None
            cond_segments = []
            for seg in segments:
                if not isinstance(seg, (tuple, list)) or len(seg) != 3:
                    continue
                a, b, kind = seg
                if kind == "cond":
                    cond_segments.append((int(a), int(b)))
                elif kind == "video":
                    target = (int(a), int(b))

            if target is None:
                raise RuntimeError(
                    "MiniMax H3 combined conditioning: target video segment was not found "
                    "in PackedLayout; refusing to guess keyframe positions."
                )
            if len(cond_segments) != len(keyframes):
                raise RuntimeError(
                    "MiniMax H3 combined conditioning: keyframe/cond segment count mismatch "
                    f"({len(keyframes)} keyframes vs {len(cond_segments)} cond segments)."
                )

            target_a, target_b = target
            if target_b <= target_a:
                raise RuntimeError("MiniMax H3 combined conditioning: target video segment is empty.")

            target_origin = float(position_ids[target_a, 0])
            offset = target_origin - float(text_len)
            if abs(offset) < 1e-12:
                return

            for a, b in cond_segments:
                position_ids[a:b, 0] = position_ids[a:b, 0] + offset

            _LOG.debug(
                "MiniMax H3 layout patch: shifted %d first/last condition span(s) by %.6f "
                "for legacy keyframe+reference coexistence.",
                len(cond_segments), offset,
            )
            return

        # middle_frame is present. Old PackedLayout rejects interior indices
        # before allocating condition rows, so temporarily map only interior
        # guides to frame 0. First/last entries are otherwise passed unchanged.
        core_keyframes = []
        for kf in keyframes:
            if not isinstance(kf, dict):
                core_keyframes.append(kf)
                continue
            item = dict(kf)
            actual = int(item.get(_H3_ACTUAL_FRAME_INDEX_KEY, item["resolved_frame_index"]))
            item[_H3_ACTUAL_FRAME_INDEX_KEY] = actual
            if item.get(_H3_INTERIOR_KEYFRAME_KEY, False):
                item["resolved_frame_index"] = 0
            core_keyframes.append(item)

        current(
            self, text_len, latent_t, latent_h, latent_w, audio_t,
            keyframes=core_keyframes, refs=refs, frame_count=frame_count,
        )

        segments = getattr(self, "segments", None)
        position_ids = getattr(self, "position_ids", None)
        if not segments or position_ids is None:
            raise RuntimeError(
                "MiniMax H3 combined conditioning: PackedLayout no longer exposes "
                "segments/position_ids; cannot place middle_frame safely."
            )

        target = None
        cond_segments = []
        for seg in segments:
            if not isinstance(seg, (tuple, list)) or len(seg) != 3:
                continue
            a, b, kind = seg
            if kind == "cond":
                cond_segments.append((int(a), int(b)))
            elif kind == "video":
                target = (int(a), int(b))

        if target is None:
            raise RuntimeError(
                "MiniMax H3 combined conditioning: target video segment was not found "
                "in PackedLayout; refusing to guess middle_frame position."
            )
        if len(cond_segments) != len(keyframes):
            raise RuntimeError(
                "MiniMax H3 combined conditioning: keyframe/cond segment count mismatch "
                f"({len(keyframes)} keyframes vs {len(cond_segments)} cond segments)."
            )

        target_a, target_b = target
        if target_b <= target_a:
            raise RuntimeError("MiniMax H3 combined conditioning: target video segment is empty.")

        target_origin = float(position_ids[target_a, 0])
        total_frames = int(frame_count) if frame_count is not None else None

        for (a, b), kf in zip(cond_segments, keyframes):
            actual = int(kf.get(_H3_ACTUAL_FRAME_INDEX_KEY, kf["resolved_frame_index"]))
            if actual < 0 or (total_frames is not None and actual >= total_frames):
                raise RuntimeError(
                    f"MiniMax H3 combined conditioning: keyframe index {actual} is outside "
                    f"the target frame range 0..{total_frames - 1 if total_frames else '?'}"
                )
            position_ids[a:b, 0] = target_origin + frame_rescale * actual

        _LOG.debug(
            "MiniMax H3 layout patch: middle_frame active; placed %d guide span(s) "
            "on exact target-frame coordinates (origin %.6f).",
            len(cond_segments), target_origin,
        )

    setattr(_patched_layout_init, _H3_LAYOUT_PATCH_MARKER, True)
    cls.__init__ = _patched_layout_init
    _LOG.info(
        "MiniMax H3 layout patch: legacy support enabled; interior-guide path is conditional."
    )
    return True


class MiniMaxH3CombinedImageAndReferenceToVideo(io.ComfyNode):
    """
    Union of MiniMaxH3ImageToVideo (fl2va keyframes) and
    MiniMaxH3ReferenceToVideo (ref2va references) inputs, in one node.

    - first_frame / last_frame behave like MiniMaxH3ImageToVideo endpoints.
    - middle_frame is an additional target-timeline keyframe placed around the
      halfway point of the generated clip (frame_count // 2). All three are
      added to conditioning as `minimax_keyframes`.
    - ref_images / ref_videos / ref_video_audios / ref_audios behave exactly
      like MiniMaxH3ReferenceToVideo (identity/motion/voice steering, no fixed
      frame position, added to the conditioning as `minimax_refs`).
    - Both mechanisms can be used together or independently; either group of
      inputs may be left empty.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3CombinedImageAndReferenceToVideo",
            display_name="MiniMax H3 Combined Image And Reference to Video",
            description=(
                "Combined fl2va + ref2va conditioning for MiniMax H3. "
                "Supports first/middle/last target keyframes AND <Picture i> / <Video k> / "
                "<Audio j> references in the same conditioning payload."
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "MiniMax H3 prompt guide. Use reference tags to point the prompt at supplied "
                        "media, for example <Picture 1>, <Picture 2>, <Video 1>, <Audio 1>, and "
                        "structured subject labels such as <Subject 1>, <Subject 2>. Keep numbering "
                        "consistent with the order of the corresponding references.\n\n"
                        "Recommended prompt sections depend on the generation mode:\n\n"
                        "1. T2VA / I2VA / FL2VA / L2VA\n"
                        "integrated_multimodal_description → overall_soundscape → non_diegetic_music\n\n"
                        "Use integrated_multimodal_description for the main visual action, subjects, "
                        "camera, timing, environment, and any image/video references. Use "
                        "overall_soundscape for diegetic/environmental sound and dialogue. Use "
                        "non_diegetic_music for score or background music.\n\n"
                        "2. Full-reference Ref2VA\n"
                        "subject_definitions → summary → retention_analysis → detailed_description → "
                        "overall_soundscape → non_diegetic_music\n\n"
                        "Use subject_definitions to establish referenced subjects and their identities "
                        "with tags such as <Subject 1>. Use summary for the overall shot intent. Use "
                        "retention_analysis to state which reference traits, identity, appearance, "
                        "motion, voice, or other characteristics should be retained. Use "
                        "detailed_description for the full scene, action, camera, timing, and reference "
                        "usage. Finish with overall_soundscape and non_diegetic_music for audio."
                    ),
                ),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362, longer is untested)",
                ),

                # --- target-timeline image keyframes ---
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Optional exact opening keyframe, conditioned at target frame 0.",
                ),
                io.Image.Input(
                    "middle_frame",
                    optional=True,
                    tooltip=(
                        "Optional middle target keyframe. It is conditioned at approximately half "
                        "the generated clip: exact 0-based index frame_count // 2. For example, "
                        "124 output frames place it at frame 62 (~2.58 s at 24 fps). The image is "
                        "aspect-preserving cover-cropped to the output canvas. Current ComfyUI H3 "
                        "supports arbitrary target keyframe indices natively; this node also includes "
                        "a compatibility layout repair for older H3 cores that only accepted endpoints."
                    ),
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Optional exact ending keyframe, conditioned at target frame frame_count - 1.",
                ),

                # --- from MiniMaxH3ReferenceToVideo (ref2va) ---
                io.Combo.Input(
                    "reference_resize_mode",
                    options=[
                        "resize to frame short edge",
                        "resize to start frame long edge",
                        "resize to start frame total_pixels",
                        "resize MiniMax H3 2k resolution",
                    ],
                    default="resize to start frame total_pixels",
                    tooltip=(
                        "How reference images are resized before MiniMax H3 reference encoding. "
                        "Aspect ratio is always preserved and references are never upscaled.\n\n"
                        "• resize to frame short edge — downscale until the reference short edge "
                        "fits the generated frame short edge. Useful for keeping reference detail "
                        "roughly bounded by the smaller frame dimension.\n"
                        "• resize to start frame long edge — downscale until the reference long edge "
                        "fits the generated/start-frame long edge. Useful for a stricter fit when the "
                        "reference and target have different aspect ratios.\n"
                        "• resize to start frame total_pixels — downscale by total pixel area so the "
                        "reference uses about the same image-token/latent area as one generated frame. "
                        "This is the former 'match' mode.\n"
                        "• resize MiniMax H3 2k resolution — keep the reference as large as possible, "
                        "only downscaling when its short edge exceeds MiniMax H3's 2048 px reference "
                        "limit. This is the former 'max' mode and can be significantly slower."
                    ),
                ),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip=(
                                "Reference image used for MiniMax H3 identity/style guidance. "
                                "Its aspect ratio is preserved. The selected reference_resize_mode "
                                "controls any downscaling; this node never enlarges the reference."
                            ),
                        ),
                        names=[f"ref_image_{i}" for i in range(1, 11)], # 1 through 10
                        min=0, 
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_video", tooltip="Reference video frames at 24 fps (2-15s)"
                        ),
                        names=[f"ref_video_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the same-numbered reference video",
                        ),
                        names=[f"ref_video_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        names=[f"ref_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls, clip, vae, audio_vae, prompt, width, height, length,
        first_frame=None, middle_frame=None, last_frame=None,
        reference_resize_mode="resize to start frame total_pixels",
        ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        # ---------- target-timeline image keyframes ----------
        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas, matching stock H3 first-frame behavior
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img, _H3_TARGET_ANCHOR_KEY: True})
        if middle_frame is not None:
            # interior guide: aspect-preserving cover-crop, placed around half the output timeline
            middle_index = frame_count // 2
            img = _resize(middle_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({
                "resolved_frame_index": middle_index,
                "image": img,
                _H3_TARGET_ANCHOR_KEY: True,
                _H3_ACTUAL_FRAME_INDEX_KEY: middle_index,
                _H3_INTERIOR_KEYFRAME_KEY: True,
            })
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img, _H3_TARGET_ANCHOR_KEY: True})

        # ---------- ref2va: images / videos / audio references ----------
        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if reference_resize_mode == "resize to frame short edge":
                scale = min(1.0, min(width, height) / min(w, h))
            elif reference_resize_mode == "resize to start frame long edge":
                scale = min(1.0, max(width, height) / max(w, h))
            elif reference_resize_mode == "resize to start frame total_pixels":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            elif reference_resize_mode == "resize MiniMax H3 2k resolution":
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            else:
                raise ValueError(f"Unknown reference_resize_mode: {reference_resize_mode}")
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({
                "type": "video", "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent,
            })

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        # ---------- tokenize + encode (both keyframe images and refs at once) ----------
        tokens = clip.tokenize(prompt, images=images, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        payload = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            payload["minimax_keyframes"] = keyframes
            payload["minimax_frame_count"] = frame_count
        if ref_blocks:
            payload["minimax_refs"] = ref_blocks
        if payload:
            cond = node_helpers.conditioning_set_values(cond, payload)

        return io.NodeOutput(cond, latent)

_ensure_h3_keyframe_ref_merge()
_ensure_h3_keyframe_ref_layout()
