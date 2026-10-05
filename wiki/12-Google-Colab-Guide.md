# Google Colab Guide

Google Colab is the primary V1 runtime. Open [Quick Start](../notebooks/Quick_Start.ipynb) in Colab and select **Runtime > Change runtime type > GPU**. GPU models and availability vary. The notebook is the supported path from a fresh session through reviewed ZIP download.

## Run The Notebook

1. Run the clone cell. It uses the confirmed `projectvault000/opensource-clipping` repository and reuses the checkout when rerun. It does not delete `/content` files.
2. Run the install cell. It installs `requirements.txt` and checks FFmpeg, FFprobe, Deno, and CLI help.
3. Add `GOOGLE_API_KEY` in Colab Secrets, allow notebook access, and run the Secrets cell. Add `PEXELS_API_KEY` only if enabling B-roll. The notebook never prints key values.
4. Optionally mount Drive. Rendering stays on fast `/content` storage; the finished approved ZIP can be copied to Drive.
5. Set a real `VIDEO_URL`. Start with one clip. A GPU is recommended for Whisper `float16`; if none is available, use `WHISPER_DEVICE = 'cpu'` with `WHISPER_COMPUTE_TYPE = 'float32'`.
6. Run the pipeline and inspect the current job's manifest and video. Set `TOTAL_CLIPS = 5` for the production attempt after the beginner test.
7. Preview each result and record `APPROVED`, `REJECTED`, or `NEEDS_CHANGES`. For a clip needing changes, edit its rank in the job's `metadata_preview.json` if needed, then run the notebook's retry cell with a clip ID such as `clip_01`. The retry rerenders only that clip and returns it to pending review; approve the new generation explicitly.
8. Export the approved set and download the ZIP. The notebook refuses an empty approved package.

## Interruption And Storage

`/content` is temporary. If only the Python process fails while the Colab runtime remains alive, rerunning the same job can reuse its local checkpoint, source, and caches. A full runtime reset can delete those files. Optional Drive mounting in this notebook copies the finished ZIP only; it does not provide persistent checkpoint recovery. Do not assume the runtime lasts for any fixed duration.

Monitor the free disk reading in the options cell before a long job. Video sources, render intermediates, package copies, and ZIPs can coexist. Disk, RAM, and GPU memory use for five clips have not yet been measured in a hosted Colab run.

## Approved Download

After the notebook's review and export cells have run, the ZIP is at the current job path:

```python
from pathlib import Path
from google.colab import files

approved_zip = Path('outputs') / CURRENT_JOB_ID / 'approved_clips.zip'
if not approved_zip.is_file():
    raise RuntimeError('Approve at least one passing clip and run the export cell first.')
files.download(str(approved_zip))
```

`notebooks/Lib_OpenSource_Clipping.ipynb` is a legacy template. It is not the V1 release notebook.
