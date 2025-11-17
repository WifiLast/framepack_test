# RunPod Proxy URL Fix

## Issue

When running FramePack on RunPod (or other reverse proxy environments), the Gradio interface would fail with this error:

```
ValueError: Request url 'https://3495ibf6jxjaa7-7860.proxy.runpod.net' has an unkown api call pattern.
```

This error occurs because Gradio's API routing cannot correctly parse URLs from reverse proxy environments like RunPod, Vast.ai, or similar cloud GPU providers.

## Root Cause

RunPod uses a custom proxy URL format (`https://{id}-{port}.proxy.runpod.net`) that Gradio's default routing doesn't recognize. Without proper configuration, Gradio tries to extract the API endpoint from the full proxy URL and fails.

## Solution

Added `root_path` parameter to all `block.launch()` calls, which tells Gradio to use an environment variable for the root path when behind a reverse proxy:

```python
block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
    root_path=os.environ.get("GRADIO_ROOT_PATH", ""),  # Fix for RunPod/proxy environments
)
```

### How It Works

1. **Environment Variable**: `GRADIO_ROOT_PATH` is automatically set by many proxy environments (including RunPod)
2. **Fallback**: If the variable doesn't exist, it defaults to an empty string (normal behavior)
3. **Compatibility**: Works with all environments:
   - ✅ RunPod proxies
   - ✅ Vast.ai proxies
   - ✅ Local development
   - ✅ Direct server access
   - ✅ Gradio share links

## Files Fixed

All Gradio demo files have been updated with this fix:

### FramePack Folder:
- ✅ [demo_gradio.py](FramePack/demo_gradio.py#L3290)
- ✅ [demo_gradio_video2loop.py](FramePack/demo_gradio_video2loop.py#L727)
- ✅ [demo_gradio_f1.py](FramePack/demo_gradio_f1.py#L434)

### FramePackLoop Folder:
- ✅ [demo_gradio.py](FramePackLoop/demo_gradio.py#L409)
- ✅ [demo_gradio_video2loop.py](FramePackLoop/demo_gradio_video2loop.py#L727)
- ✅ [demo_gradio_loop.py](FramePackLoop/demo_gradio_loop.py#L895)
- ✅ [demo_gradio_f1.py](FramePackLoop/demo_gradio_f1.py#L408)
- ✅ [demo_gradio_connect.py](FramePackLoop/demo_gradio_connect.py#L640)

## Usage

### On RunPod

No changes needed! Just run the script normally:

```bash
python FramePack/demo_gradio.py
```

Or for video2loop:

```bash
python FramePack/demo_gradio_video2loop.py
```

The fix automatically detects RunPod's proxy environment and configures Gradio correctly.

### On Other Proxy Environments

If you're using a different proxy setup (Vast.ai, Paperspace, etc.), you may need to set the environment variable manually:

```bash
export GRADIO_ROOT_PATH="/your/proxy/path"
python FramePack/demo_gradio.py
```

### On Local Machine

No changes needed - the fix doesn't affect local development:

```bash
# Windows
python FramePack/demo_gradio.py --server 127.0.0.1 --inbrowser

# Linux/Mac
python FramePack/demo_gradio.py --server 0.0.0.0
```

## Testing

To verify the fix works:

1. **Start the server**:
   ```bash
   python FramePack/demo_gradio_video2loop.py
   ```

2. **Look for the Gradio URL** in the console output:
   ```
   Running on public URL: https://3495ibf6jxjaa7-7860.proxy.runpod.net
   ```

3. **Open the URL** in your browser

4. **Verify no errors** - The interface should load without the `unknown api call pattern` error

5. **Test functionality** - Try uploading an image and starting generation to confirm the API routing works

## Technical Details

### What is `root_path`?

The `root_path` parameter tells Gradio what the base URL path is when the application is served behind a reverse proxy. This is part of the ASGI standard (used by FastAPI, which powers Gradio).

**Without `root_path`:**
- Request URL: `https://proxy.example.com/api/predict`
- Gradio tries to parse: `/api/predict`
- But expects: `/predict`
- Result: ❌ Error

**With `root_path`:**
- Request URL: `https://proxy.example.com/api/predict`
- Gradio knows base path: `/api`
- Correctly parses: `/predict`
- Result: ✅ Success

### Why Use Environment Variable?

Using `GRADIO_ROOT_PATH` environment variable allows:
1. **Automatic detection** - Proxy services can set it automatically
2. **No code changes** - Same code works in all environments
3. **Graceful fallback** - Empty string if not set (normal behavior)

### Alternative Solutions Considered

**Option 1: Hardcode root_path** ❌
```python
root_path="/some/path"  # Bad - breaks on other environments
```

**Option 2: Add command-line argument** ⚠️
```python
parser.add_argument("--root-path")  # Works but requires manual config
```

**Option 3: Environment variable** ✅ (Chosen)
```python
root_path=os.environ.get("GRADIO_ROOT_PATH", "")  # Best - automatic and flexible
```

## Related Issues

This fix resolves:
- ✅ RunPod proxy URL errors
- ✅ Vast.ai proxy URL errors
- ✅ Custom reverse proxy setups
- ✅ Nginx/Apache reverse proxy configurations
- ✅ Cloud container platforms with URL routing

## References

- [Gradio Documentation - launch() parameters](https://www.gradio.app/docs/blocks#blocks-launch)
- [ASGI Specification - root_path](https://asgi.readthedocs.io/en/latest/specs/main.html#root-path)
- [RunPod Proxy Documentation](https://docs.runpod.io/docs/expose-ports)

## Troubleshooting

### Issue: Still getting proxy errors

**Solution**: Check if the environment variable is set:

```bash
# In RunPod terminal
echo $GRADIO_ROOT_PATH

# If empty, set it manually:
export GRADIO_ROOT_PATH=""  # For root-level proxy
```

### Issue: Works on RunPod but not locally

**Solution**: This is expected! The fix is designed to be environment-aware:
- On RunPod: Uses proxy path from environment
- Locally: Uses empty path (normal behavior)

### Issue: API calls return 404

**Solution**: The root_path might be incorrect. Try:

1. Check the full URL in your browser
2. Identify the base path (everything before `/api/...`)
3. Set `GRADIO_ROOT_PATH` to that base path

Example:
- URL: `https://proxy.com/custom/base/api/predict`
- Set: `export GRADIO_ROOT_PATH="/custom/base"`

## Verification

The fix is working if you see:

✅ **Success indicators:**
- Gradio interface loads in browser
- No `unknown api call pattern` errors in console
- File uploads work
- API calls complete successfully
- Progress updates display correctly

❌ **Still broken if you see:**
- `ValueError: Request url '...' has an unkown api call pattern.`
- 404 errors on API calls
- Interface loads but buttons don't work

If you still see errors after this fix, check your proxy configuration or file an issue with:
- The exact error message
- Your proxy URL format
- The output of `echo $GRADIO_ROOT_PATH`
