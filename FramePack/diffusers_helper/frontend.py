import gradio as gr

from diffusers_helper.gradio.progress_bar import make_progress_bar_css


def build_frontend(
    *,
    quick_prompts,
    enable_fbcache: bool,
    enable_sim_cache: bool,
    enable_kv_cache: bool,
    cache_mode: str,
    tensorrt_available: bool,
    tensorrt_transformer_available: bool,
    process_fn,
    end_fn,
    relationship_modes=("off", "hidden_state", "residual", "modulation"),
):
    """
    Create the Gradio Blocks layout for the FramePack demo UI.

    Parameters mirror the previously inlined configuration in demo_gradio.py so the
    UI can be reused by other entry points.
    """
    css = make_progress_bar_css()
    block = gr.Blocks(css=css, analytics_enabled=False).queue()

    with block:
        gr.Markdown('# FramePack')
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=420):
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                prompt = gr.Textbox(label="Prompt", value='')
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used

                with gr.Row(variant='compact'):
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)

                with gr.Row(variant='compact'):
                    start_button = gr.Button(value="Start Generation", variant='primary')
                    end_button = gr.Button(value="End Generation", interactive=False)

                with gr.Accordion("Quality & Cache", open=False):
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    use_fb_cache = gr.Checkbox(
                        label='Use First Block Cache',
                        value=enable_fbcache,
                        info="Caches the transformer's first block to reuse prompts faster. Disable to save VRAM or avoid stale results.",
                    )
                    use_sim_cache = gr.Checkbox(
                        label='Use Similarity Cache',
                        value=enable_sim_cache,
                        info='Reuses similar frames via FAISS-backed cache. Disable if it causes artifacts or to save VRAM.',
                    )
                    use_kv_cache = gr.Checkbox(
                        label='Use KV Cache',
                        value=enable_kv_cache,
                        info='Keeps transformer KV states for reuse. Disable for lower memory usage.',
                    )
                    quality_mode = gr.Checkbox(label='Quality Mode (Better Hands)', value=False, info='Uses larger VAE chunks (2-4 frames) for better quality, especially for hands. Requires more VRAM.')
                    cache_mode_selector = gr.Radio(
                        label='Cache Mode',
                        choices=['hash', 'semantic', 'off'],
                        value=cache_mode,
                        info='hash = deterministic exact reuse, semantic = FAISS-backed approximate hits, off = disable caching.',
                    )
                    slow_prompt_hint = gr.Checkbox(
                        label='Add "move slowly" hint',
                        value=True,
                        info='Appends a slow-motion phrasing to your prompt to encourage smoother motion.',
                    )

                with gr.Accordion("Experimental", open=False):
                    relationship_trainer_mode = gr.Radio(
                        label="Relationship Trainer Mode",
                        choices=list(relationship_modes),
                        value=relationship_modes[0],
                        info="Choose which DiT surrogate to train: off=disabled, hidden_state=block I/O residual, residual=Δh predictor, modulation=γ/β predictor.",
                    )
                    rt_learning_rate = gr.Slider(
                        label="Trainer Learning Rate",
                        minimum=1e-6,
                        maximum=1e-3,
                        value=1e-4,
                        step=1e-6,
                        info="Learning rate for whichever relationship trainer mode is active.",
                    )

                with gr.Accordion("Sampler Controls", open=False):
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=8.0, step=0.01, info='Changing this value is not recommended.')
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                with gr.Accordion("Performance & Output", open=False):
                    gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=2, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed. For 16GB VRAM, use 2-5GB for best performance.")
                    mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                    tensorrt_decode_checkbox = gr.Checkbox(
                        label="TensorRT VAE Acceleration (beta)",
                        value=tensorrt_available,
                        visible=tensorrt_available,
                        info="Requires torch-tensorrt + CUDA. Speeds up both VAE encode/decode and falls back automatically if unsupported.",
                    )
                    tensorrt_transformer_checkbox = gr.Checkbox(
                        label="TensorRT Transformer Acceleration (experimental)",
                        value=tensorrt_transformer_available,
                        visible=tensorrt_transformer_available,
                        info="Requires torch-tensorrt + CUDA. Compiles transformer on first use (takes ~5-10min), then provides significant speedup. Experimental feature.",
                    )

            with gr.Column(scale=1, min_width=360):
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                result_video = gr.Video(label="Finished Frames (30 FPS)", autoplay=True, show_share_button=False, height=512, loop=True)
                gr.Markdown('''**Important Notes:**
- Videos are rendered at 30 FPS (standard video framerate)
- The ending actions are generated before starting actions (inverted sampling)
- For better hand quality: Enable "Quality Mode" and disable "TeaCache"
- If video appears too fast, reduce "Total Video Length" to generate more frames per second of content
''')
                with gr.Accordion("Status", open=True):
                    progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                    cache_timeline_md = gr.Markdown('No cache hits recorded yet.')

        gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

        ips = [
            input_image,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_preservation,
            use_teacache,
            use_fb_cache,
            use_sim_cache,
            use_kv_cache,
            slow_prompt_hint,
            cache_mode_selector,
            mp4_crf,
            quality_mode,
            tensorrt_decode_checkbox,
            tensorrt_transformer_checkbox,
            relationship_trainer_mode,
            rt_learning_rate,
        ]
        start_button.click(
            fn=process_fn,
            inputs=ips,
            outputs=[result_video, preview_image, progress_desc, progress_bar, cache_timeline_md, start_button, end_button],
        )
        end_button.click(fn=end_fn)

    return block
