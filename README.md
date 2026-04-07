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
pipeline_battle_test.exe <input_image> <output_dir> [scale] [triangle_limit] [target_error] [--no-stall|--stall]
```

### Basic example

```bash
pipeline_battle_test.exe input.png out 4 0 0
```

### High-quality example

```bash
pipeline_battle_test.exe input.png out 4 500000 0 --no-stall
```

### Parameters

- `input_image` — path to the source image
- `output_dir` — directory where results are saved
- `scale` — upscale factor (for example: `2`, `4`, `8`)
- `triangle_limit` — triangle budget (`0` = automatic budget selection)
- `target_error` — target error (`0` = automatic)
- `--no-stall` — disables early stopping by progress stall
- `--stall` — explicitly enables progress-stall stopping

## ⚡ Quality vs Triangle Budget

The quality of the result is **directly controlled by the triangle budget**.

- Lower triangle counts → faster results, but rougher approximation
- Higher triangle counts → significantly better detail, especially in complex regions

### Automatic mode

If you run the tool like this:

```bash
pipeline_battle_test.exe input.png out 4 0 0
```

then:

- `triangle_limit = 0` means **automatic triangle budget**
- `target_error = 0` means **automatic target error**
- progress-stall stopping stays **enabled by default**

This is convenient for normal usage and quick tests.

### Manual high-quality mode

If the result looks too rough, simplified, or under-detailed, try:

1. replacing `triangle_limit = 0` with a much larger manual value
2. disabling progress-stall stopping with `--no-stall`

For example:

```bash
pipeline_battle_test.exe input.png out 4 500000 0 --no-stall
```

This tells the tool to:

- use a much larger triangle budget
- keep refining even when progress becomes slow

That often improves difficult regions, but it also makes the run much slower.

## ⏱️ Progress-stall stopping

By default, the tool may stop refinement early when additional improvement becomes too small.

This is useful for normal runs because it prevents wasting time on tiny gains.

However, for difficult images, disabling this behavior can improve quality:

```bash
pipeline_battle_test.exe input.png out 4 500000 0 --no-stall
```

Use this mode when you want to push quality harder, especially on:

- foliage
- grass
- clouds
- water
- dense textures
- complex natural photos

### Important trade-off

Disabling progress-stall stopping may increase runtime significantly.

At high triangle budgets, processing may take **several minutes or more**.

## 📈 Practical recommendation

If the output looks rough:

```text
1. keep scale as needed
2. replace triangle_limit = 0 with a larger manual value
3. add --no-stall for maximum refinement
```

A good first manual value is:

```text
500000
```

Then increase further if needed.

### Typical behavior

- simple images → good quality with relatively small budgets
- portraits / anime / clean graphics → usually converge much faster
- heavily textured photos → may require **hundreds of thousands of triangles**

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
- better quality often requires a much larger triangle budget
- disabling progress-stall stopping may help on difficult scenes

This is mainly due to the current optimization strategy, which still spends budget inefficiently in some highly detailed regions.

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

## 🏗️ Build

### Requirements

- CMake 3.20+
- C++ compiler with C++20 support  
  (MSVC / Visual Studio 2022 recommended on Windows)

---

### Build using CMake (recommended)

```bash
cmake -B build
cmake --build build --config Release
```

After build, the executable will be located in:

```text
build/Release/
```

---

### Build using Visual Studio

1. Open the project folder in Visual Studio  
2. CMake will configure automatically using `CMakePresets.json`  
3. Select configuration: **Release**  
4. Build the project

---

### Notes

- `CMakePresets.json` is included, so no manual configuration is required in Visual Studio
- On Windows, `--config Release` is required because Visual Studio uses multi-config builds
- The project has no external dependencies (third-party headers are included)

---

### Troubleshooting

If something fails to build, try removing the `build/` directory and re-running the commands:

```bash
rm -rf build
cmake -B build
cmake --build build --config Release
```
