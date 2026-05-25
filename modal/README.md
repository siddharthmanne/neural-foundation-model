# Modal scaffold — neural-fm

The minimal Modal setup for our team workspace. The scaffold is just enough infrastructure to let any of us spin up a Modal job when they need one; it doesn't try to anticipate what those jobs will be. Each of us writes our own Modal files as our work actually needs them.

## What's in this folder

- `modal_app.py` — shared App, Image, Volume, Secret. Everything else imports from here.
- `requirements.txt` — Python packages installed into the Modal container.
- `MODAL_GUIDE.md` — primer for teammates on what Modal is and how to write a Modal file when they need one.
- `demo_train.py` + `modal_demo.py` — a runnable example of the "two-file pattern" for training scripts. Copy this pair as a template when you need to run a real training job on Modal. See `MODAL_GUIDE.md` for the explanation.

## First-time setup (per teammate)

1. Install the Modal CLI on your laptop:
   ```
   pip install modal
   modal setup
   ```
   `modal setup` opens a browser to link your CLI to your Modal account.

2. Make sure you're in the `neural-fm` workspace, not your personal one. In the Modal dashboard, switch via the workspace dropdown in the top-left. To set it as your CLI default:
   ```
   modal config set-workspace neural-fm
   ```
   (You can also pass `--workspace neural-fm` to any individual command.)

3. Test it works:
   ```
   modal run modal_app.py::hello
   ```
   Should print "Hello from Modal!" and list what's currently in `/data`.

## Try the demo

```
modal run modal_demo.py
```

This runs `demo_train.py` (a plain Python script) on a Modal container for a few seconds and writes a result file to the shared Volume at `/data/demo_output/`. It's there so you can see the wrapper pattern actually work end-to-end before you adapt it for your real work.

## When to add more files

You'll need a new Modal file when you have something to run on a Modal GPU — typically a training script or an evaluation that's too big for FarmShare. See `MODAL_GUIDE.md` for a template and explanation.

## Common gotchas

- **"Secret not found":** the `huggingface` secret hasn't been created in the team workspace yet, or you're running in your personal workspace instead of `neural-fm`.
- **Volume changes don't persist:** make sure your function calls `data_volume.commit()` before returning.
- **Job runs forever:** check the dashboard at modal.com and kill stuck containers manually.
