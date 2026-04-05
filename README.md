Triangle Surface Vectorizer (CSC)

A small experimental tool for reconstructing raster images as a triangle-based continuous color surface and rendering them at arbitrary resolution.

This is not a classical vectorizer (no Bézier curves).
Instead, the image is approximated using a dense triangle mesh with per-vertex color.

The result behaves like a resolution-independent image approximation, which is especially useful for upscaling without blur.

✨ What it does

Reconstructs raster images as a triangle mesh (RGBA surface)
Allows high-resolution rendering without typical interpolation blur
Works as a deterministic alternative to AI upscaling
Handles a wide range of inputs:
portraits
anime artwork
logos and flat graphics
scanned images

🎯 Why this exists

This tool was built to solve a practical problem:

Small or low-quality images often become unusable when enlarged (blur, mush, artifacts).

Standard interpolation (Photoshop, etc.) smooths everything.
AI upscalers may produce inconsistent or incorrect results.

This approach instead:

reconstructs a continuous approximation of the image and renders it at the required resolution.

Result:

no blur
no hallucinated details
stable, predictable output

🧠 How it works (simplified)

Input image is treated as a sampled color field
The algorithm builds a triangle mesh approximation
Colors are interpolated across triangles (RGBA)
Final image is rendered at target resolution

Key idea:

Pixels are no longer the fundamental unit — triangles are.

⚙️ Usage

pipeline_battle_test.exe <input_image> <output_dir> <scale> <triangle_limit> <param>

Example:

pipeline_battle_test.exe input.png out 4 0 0
Parameters
input_image — path to source image
output_dir — where results are saved
scale — upscale factor (e.g. 2, 4, 8, etc.)
triangle_limit — limit on triangle count (0 = automatic)
param — internal parameter (currently not required for typical usage)

📦 Output

The tool currently renders results directly to raster images (e.g. PNG/TIFF).

Internally, the image is represented as a triangle mesh, but:

there is currently no standardized external format for storing this representation.

⚠️ Limitations

The algorithm is general-purpose, but some image types are harder than others.

Known weak spots:

dense foliage
grass textures
cloudy skies
highly stochastic / high-frequency regions

In such cases:

the image remains usable
but local detail may be simplified or unevenly approximated

This is mainly due to the current RMSE-based optimization, which does not fully capture perceptual texture importance.

✅ Where it works best

portraits / people
old or scanned photos
anime-style artwork
logos and clean graphics
images that need to be printed at larger sizes than their original resolution allows

💡 Typical use case

Client provides a small image → needs large print (canvas, poster, etc.)

Instead of:

blurry upscale
or unpredictable AI result

You get:

stable geometric approximation
consistent visual structure
print-friendly output

🧪 Project status

This is an experimental tool built for real-world use, not a polished product.

It is already very effective in many scenarios, but:

some edge cases are not fully solved
no GUI
no standard file format yet


📜 License

MIT License

👤 Author

Built as a personal tool for image reconstruction and print workflows.
