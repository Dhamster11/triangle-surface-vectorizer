# Triangle Surface Vectorizer (CSC)

A small experimental tool for reconstructing raster images as a **triangle-based continuous color surface** and rendering them at arbitrary resolution.

This is **not** a classical vectorizer and does **not** use Bézier curves. Instead, the image is approximated using a dense triangle mesh with per-vertex color.

The result behaves like a **resolution-independent image approximation**, which is especially useful for **upscaling without blur**.

## ✨ What it does

- Reconstructs raster images as a triangle mesh (RGBA surface)
- Allows high-resolution rendering without typical interpolation blur
- Works as a deterministic alternative to AI upscaling
- Handles a wide range of inputs, including:
  - portraits
  - anime artwork
  - logos and flat graphics
  - scanned images

## 🎯 Why this exists

This tool was built to solve a practical problem:

> Small or low-quality images often become unusable when enlarged  
> (blur, mush, artifacts).

Standard interpolation in tools like Photoshop smooths everything. AI upscalers may produce inconsistent or incorrect results.

This approach instead:

> reconstructs a continuous approximation of the image  
> and renders it at the required resolution.

### Result

- no blur
- no hallucinated details
- stable, predictable output

## 🧠 How it works (simplified)

1. The input image is treated as a sampled color field
2. The algorithm builds a triangle mesh approximation
3. Colors are interpolated across triangles (RGBA)
4. The final image is rendered at the target resolution

**Key idea:** pixels are no longer the fundamental unit — triangles are.

## ⚙️ Usage

```bash
pipeline_battle_test.exe <input_image> <output_dir> <scale> <triangle_limit> <param>
```

### Example

```bash
pipeline_battle_test.exe input.png out 4 0 0
```

### Parameters

- `input_image` — path to the source image
- `output_dir` — directory where results are saved
- `scale` — upscale factor (for example: `2`, `4`, `8`)
- `triangle_limit` — limit on triangle count (`0` = automatic)
- `param` — internal parameter (currently not required for typical usage)

## 📦 Output

The tool currently renders results to **PNG only**.

Internally, the image is represented as a triangle mesh, but there is currently no standardized external format for storing this representation directly.

## ⚠️ Limitations

The algorithm is general-purpose, but some image regions are harder than others.

### Known weak spots

- dense foliage
- some grass textures
- cloudy skies
- highly stochastic / high-frequency regions

In such cases:

- the image usually remains usable
- but local detail may be simplified or unevenly approximated

This is mainly due to the current **RMSE-based optimization**, which does not fully capture perceptual texture importance.

## ✅ Where it works best

- portraits / people
- old or scanned photos
- anime-style artwork
- logos and clean graphics
- images that need to be printed at larger sizes than their original resolution allows

## 💡 Typical use case

> A client provides a small image and wants a larger print  
> (canvas, poster, mockup, etc.)

Instead of getting:

- blurry interpolation
- or an unpredictable AI result

You get:

- a stable geometric approximation
- consistent visual structure
- print-friendly output

## 🧪 Project status

This is an experimental tool built for real-world use, not a polished product.

It is already very effective in many scenarios, but:

- some edge cases are not fully solved
- there is no GUI
- there is no standard file format yet for the triangle surface itself

## 📜 License

MIT License

## 👤 Author

Built as a personal tool for image reconstruction and print workflows.
