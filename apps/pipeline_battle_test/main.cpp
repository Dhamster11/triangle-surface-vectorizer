#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "svec/edge/edge_map.h"
#include "svec/error/error_estimator.h"
#include "svec/error/hierarchical_surface_error.h"
#include "svec/image/image_io.h"
#include "svec/image/oklab_convert.h"
#include "svec/image/scanline_integral_stats.h"
#include "svec/math/scalar.h"
#include "svec/meshopt/mesh_optimization.h"
#include "svec/pyramid/error_pyramid.h"
#include "svec/refine/pyramid_refinement.h"
#include "svec/render/reference_renderer.h"
#include "svec/render/surface_plane_renderer.h"
#include "svec/surface/mesh_builders.h"
#include "svec/surface/mesh_topology.h"
#include "svec/surface/triangle_plane.h"

namespace fs = std::filesystem;
using namespace svec;

namespace {

    constexpr const char* kRefineEngineName = "pyramid_hierarchical";

    struct AppConfig {
        std::string input_path;
        std::string output_dir;
        u32 upscale_factor = 4;
        u32 max_triangles = 0;  // 0 = auto
        f64 target_error = 0.0; // 0 = auto
        bool stop_on_progress_stall = true; // default: enabled
    };

    struct AutoProfile {
        u32 grid_x = 16;
        u32 grid_y = 16;
        u32 bootstrap_splits = 128;
        u32 batch_size = 16;
        u32 max_splits = 12000;
        u32 max_triangles = 12000;
        u32 optimize_passes = 2;
        f64 target_error = 0.022;

        f64 min_edge_length = 0.65;
        f64 min_triangle_area = 0.20;
        f64 min_triangle_bbox_extent = 0.60;
    };

    void PrintUsage(const char* exe) {
        std::cout
            << "Usage:\n  " << exe << " <input_image> <output_dir> [upscale_factor] [max_triangles_or_0_auto] [target_error_or_0_auto] [--no-stall|--stall]\n\n"
            << "Examples:\n"
            << "  " << exe << " input.jpg out 4 0 0\n"
            << "  " << exe << " input.jpg out 4 500000 0 --no-stall\n";
    }

    bool ParseU32(const char* text, u32& out_value) {
        try {
            out_value = static_cast<u32>(std::stoul(text));
            return true;
        }
        catch (...) {
            return false;
        }
    }

    bool ParseF64(const char* text, f64& out_value) {
        try {
            out_value = std::stod(text);
            return true;
        }
        catch (...) {
            return false;
        }
    }

    bool ParseArgs(int argc, char** argv, AppConfig& cfg) {
        if (argc < 3) return false;

        int positional_index = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--no-stall") {
                cfg.stop_on_progress_stall = false;
                continue;
            }
            if (arg == "--stall") {
                cfg.stop_on_progress_stall = true;
                continue;
            }

            switch (positional_index) {
            case 0:
                cfg.input_path = arg;
                break;
            case 1:
                cfg.output_dir = arg;
                break;
            case 2:
                if (!ParseU32(argv[i], cfg.upscale_factor)) return false;
                break;
            case 3:
                if (!ParseU32(argv[i], cfg.max_triangles)) return false;
                break;
            case 4:
                if (!ParseF64(argv[i], cfg.target_error)) return false;
                break;
            default:
                return false;
            }

            ++positional_index;
        }

        if (positional_index < 2) return false;
        if (cfg.upscale_factor == 0) cfg.upscale_factor = 1;
        return true;
    }

    bool SaveOKLabImage(const fs::path& path, const ImageOKLaba& image) {
        std::string error;
        if (!SaveRGBA8ToFile(path.string(), ConvertToRGBA8(image), {}, &error)) {
            std::cerr << "Save failed: " << path << " :: " << error << "\n";
            return false;
        }
        return true;
    }

    ImageSize MakeUpscaledImageSize(const ImageSize& base, u32 factor) {
        if (!base.IsValid() || factor <= 1) return base;
        return {
            static_cast<i32>((base.width - 1) * static_cast<i32>(factor) + 1),
            static_cast<i32>((base.height - 1) * static_cast<i32>(factor) + 1)
        };
    }

    AutoProfile BuildAutoProfile(const ImageOKLaba& reference, const EdgeMap& edge_map, const AppConfig& cfg) {
        AutoProfile out{};
        const f64 pixel_count = static_cast<f64>(reference.Size().PixelCount());
        const f64 mp = pixel_count / 1'000'000.0;

        f64 edge_sum = 0.0;
        f64 strong = 0.0;
        for (f64 v : edge_map.Pixels()) {
            edge_sum += v;
            if (v >= 0.20) strong += 1.0;
        }
        const f64 mean_edge = edge_map.Pixels().empty() ? 0.0 : edge_sum / static_cast<f64>(edge_map.Pixels().size());
        const f64 strong_fraction = edge_map.Pixels().empty() ? 0.0 : strong / static_cast<f64>(edge_map.Pixels().size());

        const f64 mean_edge_n = Saturate(mean_edge * 4.0);
        const f64 strong_fraction_n = Saturate(strong_fraction * 6.0);

        const f64 complexity = Saturate(
            0.55 * mean_edge_n
            + 0.45 * strong_fraction_n
        );

        const f64 aspect = static_cast<f64>(reference.Width()) / static_cast<f64>(Max(reference.Height(), 1));
        const f64 aspect_sqrt = std::sqrt(Max(aspect, 1e-6));
        const f64 base_grid = 10.0 + 10.0 * std::sqrt(Max(mp, 0.01)) + 18.0 * complexity;
        out.grid_x = static_cast<u32>(Clamp<i64>(static_cast<i64>(std::llround(base_grid * aspect_sqrt)), 8, 80));
        out.grid_y = static_cast<u32>(Clamp<i64>(static_cast<i64>(std::llround(base_grid / aspect_sqrt)), 8, 80));

        out.bootstrap_splits = static_cast<u32>(Clamp<i64>(static_cast<i64>(2 * out.grid_x + 2 * out.grid_y), 64, 256));
        out.batch_size = complexity >= 0.35 ? 64u : 48u;
        out.max_triangles = cfg.max_triangles > 0 ? cfg.max_triangles
            : static_cast<u32>(Clamp<i64>(
                static_cast<i64>(std::llround(
                    3500.0
                    + pixel_count / 25.0
                    + 2558000.0 * complexity
                )),
                6000,
                300000));
        out.max_splits = Max<u32>(out.max_triangles * 2u, out.bootstrap_splits + 1000u);
        out.optimize_passes = complexity >= 0.35 ? 2u : 3u;
        out.target_error = cfg.target_error > 0.0 ? cfg.target_error : Clamp(0.022 - 0.007 * complexity, 0.012, 0.022);
        out.min_edge_length = Clamp(0.85 - 0.25 * complexity, 0.50, 0.85);
        out.min_triangle_area = Clamp(0.35 - 0.15 * complexity, 0.12, 0.35);
        out.min_triangle_bbox_extent = Clamp(0.95 - 0.25 * complexity, 0.55, 0.95);
        return out;
    }

    void WriteStats(
        const fs::path& path,
        const AppConfig& cfg,
        const AutoProfile& profile,
        const PyramidRefinementOptions& refine_opts,
        const ImageOKLaba& reference,
        const PyramidRefinementReport& refine_report,
        const MeshOptimizationReport& opt_report,
        const Mesh& mesh,
        const SurfacePlaneRenderResult& preview,
        const SurfacePlaneRenderResult& wire,
        const EdgeMap& edge_map) {

        std::ofstream os(path);
        os << "input_path=" << cfg.input_path << "\n";
        os << "refine_engine=" << kRefineEngineName << "\n";
        os << "image_width=" << reference.Width() << "\n";
        os << "image_height=" << reference.Height() << "\n";
        os << "upscale_factor=" << cfg.upscale_factor << "\n";
        os << "auto_grid_x=" << profile.grid_x << "\n";
        os << "auto_grid_y=" << profile.grid_y << "\n";
        os << "auto_max_triangles=" << profile.max_triangles << "\n";
        os << "auto_target_error=" << profile.target_error << "\n";
        os << "bootstrap_splits=" << profile.bootstrap_splits << "\n";
        os << "batch_size=" << profile.batch_size << "\n";
        os << "stop_on_progress_stall=" << (refine_opts.safety.stop_on_progress_stall ? 1 : 0) << "\n";
        os << "splits_requested=" << refine_report.splits_requested << "\n";
        os << "stop_reason=" << ToString(refine_report.stop_reason) << "\n";
        os << "splits_performed=" << refine_report.splits_performed << "\n";
        os << "batches_performed=" << refine_report.batches_performed << "\n";
        os << "blocked_split_attempts=" << refine_report.blocked_split_attempts << "\n";
        os << "stale_heap_pops=" << refine_report.stale_heap_pops << "\n";
        os << "heap_pushes_total=" << refine_report.telemetry.heap_pushes_total << "\n";
        os << "heap_pops_total=" << refine_report.telemetry.heap_pops_total << "\n";
        os << "heap_valid_pops=" << refine_report.telemetry.heap_valid_pops << "\n";
        os << "heap_rebuild_count=" << refine_report.telemetry.heap_rebuild_count << "\n";
        os << "heap_entries_discarded_by_rebuild=" << refine_report.telemetry.heap_entries_discarded_by_rebuild << "\n";
        os << "heap_max_size=" << refine_report.telemetry.heap_max_size << "\n";
        os << "edge_bias_enabled=" << (refine_opts.edge_bias.enabled ? 1 : 0) << "\n";
        os << "edge_bias_mean_weight=" << refine_opts.edge_bias.mean_weight << "\n";
        os << "edge_bias_peak_weight=" << refine_opts.edge_bias.peak_weight << "\n";
        os << "edge_bias_power=" << refine_opts.edge_bias.power << "\n";
        os << "edge_bias_strong_edge_threshold=" << refine_opts.edge_bias.strong_edge_threshold << "\n";
        os << "edge_bias_strong_edge_bonus=" << refine_opts.edge_bias.strong_edge_bonus << "\n";
        os << "edge_bias_max_multiplier=" << refine_opts.edge_bias.max_multiplier << "\n";
        os << "time_build_pyramid_ms=" << refine_report.telemetry.time_build_pyramid_ms << "\n";
        os << "time_initial_topology_ms=" << refine_report.telemetry.time_initial_topology_ms << "\n";
        os << "time_recolor_vertices_ms=" << refine_report.telemetry.time_recolor_vertices_ms << "\n";
        os << "time_initial_cache_ms=" << refine_report.telemetry.time_initial_cache_ms << "\n";
        os << "time_split_geometry_ms=" << refine_report.telemetry.time_split_geometry_ms << "\n";
        os << "time_topology_rebuild_ms=" << refine_report.telemetry.time_topology_rebuild_ms << "\n";
        os << "time_refresh_ms=" << refine_report.telemetry.time_refresh_ms << "\n";
        os << "time_plane_fit_ms=" << refine_report.telemetry.time_plane_fit_ms << "\n";
        os << "time_hier_error_ms=" << refine_report.telemetry.time_hier_error_ms << "\n";
        os << "time_edge_bias_ms=" << refine_report.telemetry.time_edge_bias_ms << "\n";
        os << "time_final_planes_ms=" << refine_report.telemetry.time_final_planes_ms << "\n";
        os << "time_final_error_ms=" << refine_report.telemetry.time_final_error_ms << "\n";
        os << "refresh_calls_total=" << refine_report.telemetry.refresh_calls_total << "\n";
        os << "refresh_unique_triangles=" << refine_report.telemetry.refresh_unique_triangles << "\n";
        os << "refresh_neighbor_calls=" << refine_report.telemetry.refresh_neighbor_calls << "\n";
        os << "topology_rebuild_count=" << refine_report.telemetry.topology_rebuild_count << "\n";
        os << "initial_cache_triangle_count=" << refine_report.telemetry.initial_cache_triangle_count << "\n";
        os << "edge_proxy_evaluations=" << refine_report.telemetry.edge_proxy_evaluations << "\n";
        os << "split_rejected_seed_too_small=" << refine_report.telemetry.split_rejected_seed_too_small << "\n";
        os << "split_rejected_bbox_too_small=" << refine_report.telemetry.split_rejected_bbox_too_small << "\n";
        os << "split_rejected_edge_too_short=" << refine_report.telemetry.split_rejected_edge_too_short << "\n";
        os << "split_rejected_split_point_unsafe=" << refine_report.telemetry.split_rejected_split_point_unsafe << "\n";
        os << "split_rejected_neighbor_child_invalid=" << refine_report.telemetry.split_rejected_neighbor_child_invalid << "\n";
        os << "split_rejected_split_execution_failed=" << refine_report.telemetry.split_rejected_split_execution_failed << "\n";
        os << "triangles_final=" << mesh.triangles.size() << "\n";
        os << "vertices_final=" << mesh.vertices.size() << "\n";
        os << "initial_mean_error=" << refine_report.initial_error.mean_composite_error << "\n";
        os << "after_refine_mean_error=" << refine_report.final_error.mean_composite_error << "\n";
        os << "after_opt_weighted_rmse=" << opt_report.final_error.weighted_rmse << "\n";
        os << "final_min_triangle_angle_deg=" << ComputeMeshMinTriangleAngleDegrees(mesh) << "\n";
        os << "final_mean_min_angle_deg=" << ComputeMeshMeanTriangleMinAngleDegrees(mesh) << "\n";
        os << "edge_map_max=" << edge_map.MaxValue() << "\n";
        os << "topology_non_manifold=false\n";
        os << "preview_pixels_shaded=" << preview.stats.pixels_shaded << "\n";
        os << "wire_pixels_shaded=" << wire.stats.wire_pixels_shaded << "\n";
        os << "flips_applied=" << opt_report.TotalFlipsApplied() << "\n";
        os << "vertex_moves_applied=" << opt_report.TotalVertexMovesApplied() << "\n";
    }

    void WriteRefineStageStats(
        const fs::path& path,
        const AppConfig& cfg,
        const AutoProfile& profile,
        const PyramidRefinementOptions& refine_opts,
        const ImageOKLaba& reference,
        const PyramidRefinementReport& refine_report,
        i64 refine_ms,
        const Mesh& mesh) {

        std::ofstream os(path);
        os << "input_path=" << cfg.input_path << "\n";
        os << "refine_engine=" << kRefineEngineName << "\n";
        os << "image_width=" << reference.Width() << "\n";
        os << "image_height=" << reference.Height() << "\n";
        os << "auto_grid_x=" << profile.grid_x << "\n";
        os << "auto_grid_y=" << profile.grid_y << "\n";
        os << "bootstrap_splits=" << profile.bootstrap_splits << "\n";
        os << "batch_size=" << profile.batch_size << "\n";
        os << "target_mean_error=" << refine_opts.target_mean_error << "\n";
        os << "stop_on_progress_stall=" << (refine_opts.safety.stop_on_progress_stall ? 1 : 0) << "\n";
        os << "splits_requested=" << refine_report.splits_requested << "\n";
        os << "splits_performed=" << refine_report.splits_performed << "\n";
        os << "stop_reason=" << ToString(refine_report.stop_reason) << "\n";
        os << "refine_ms=" << refine_ms << "\n";
        os << "triangles_after_refine=" << mesh.triangles.size() << "\n";
        os << "vertices_after_refine=" << mesh.vertices.size() << "\n";
        os << "initial_mean_error=" << refine_report.initial_error.mean_composite_error << "\n";
        os << "final_mean_error=" << refine_report.final_error.mean_composite_error << "\n";
        os << "heap_pushes_total=" << refine_report.telemetry.heap_pushes_total << "\n";
        os << "heap_pops_total=" << refine_report.telemetry.heap_pops_total << "\n";
        os << "heap_valid_pops=" << refine_report.telemetry.heap_valid_pops << "\n";
        os << "heap_rebuild_count=" << refine_report.telemetry.heap_rebuild_count << "\n";
        os << "stale_heap_pops=" << refine_report.stale_heap_pops << "\n";
        os << "refresh_calls_total=" << refine_report.telemetry.refresh_calls_total << "\n";
        os << "refresh_unique_triangles=" << refine_report.telemetry.refresh_unique_triangles << "\n";
        os << "time_split_geometry_ms=" << refine_report.telemetry.time_split_geometry_ms << "\n";
        os << "time_refresh_ms=" << refine_report.telemetry.time_refresh_ms << "\n";
        os << "time_topology_rebuild_ms=" << refine_report.telemetry.time_topology_rebuild_ms << "\n";
        os << "time_hier_error_ms=" << refine_report.telemetry.time_hier_error_ms << "\n";
        os << "time_plane_fit_ms=" << refine_report.telemetry.time_plane_fit_ms << "\n";
    }

    void PrintRefineStageSummary(
        const PyramidRefinementReport& refine_report,
        i64 refine_ms,
        const Mesh& mesh) {

        std::cout << "Refine complete.\n";
        std::cout << "  stop reason=" << ToString(refine_report.stop_reason) << "\n";
        std::cout << "  refine ms=" << refine_ms << "\n";
        std::cout << "  triangles after refine=" << mesh.triangles.size() << "\n";
        std::cout << "  vertices after refine=" << mesh.vertices.size() << "\n";
        std::cout << "  initial mean error=" << refine_report.initial_error.mean_composite_error << "\n";
        std::cout << "  final mean error=" << refine_report.final_error.mean_composite_error << "\n";
        std::cout << "  refresh calls/unique=" << refine_report.telemetry.refresh_calls_total
            << "/" << refine_report.telemetry.refresh_unique_triangles << "\n";
        std::cout << "  stale heap pops=" << refine_report.stale_heap_pops << "\n";
    }

    void WriteOptimizeStageStats(
        const fs::path& path,
        i64 optimize_ms,
        const MeshOptimizationReport& opt_report,
        const Mesh& mesh) {

        std::ofstream os(path);
        os << "optimize_ms=" << optimize_ms << "\n";
        os << "triangles_after_opt=" << mesh.triangles.size() << "\n";
        os << "vertices_after_opt=" << mesh.vertices.size() << "\n";
        os << "initial_weighted_rmse=" << opt_report.initial_error.weighted_rmse << "\n";
        os << "final_weighted_rmse=" << opt_report.final_error.weighted_rmse << "\n";
        os << "initial_min_triangle_angle_deg=" << opt_report.initial_min_triangle_angle_deg << "\n";
        os << "final_min_triangle_angle_deg=" << opt_report.final_min_triangle_angle_deg << "\n";
        os << "initial_mean_min_angle_deg=" << opt_report.initial_mean_min_angle_deg << "\n";
        os << "final_mean_min_angle_deg=" << opt_report.final_mean_min_angle_deg << "\n";
        os << "flips_applied=" << opt_report.TotalFlipsApplied() << "\n";
        os << "vertex_moves_applied=" << opt_report.TotalVertexMovesApplied() << "\n";
        os << "iterations_performed=" << opt_report.IterationsPerformed() << "\n";
    }

    void PrintOptimizeStageSummary(i64 optimize_ms, const MeshOptimizationReport& opt_report) {
        std::cout << "Optimize complete.\n";
        std::cout << "  optimize ms=" << optimize_ms << "\n";
        std::cout << "  weighted_rmse before opt=" << opt_report.initial_error.weighted_rmse << "\n";
        std::cout << "  weighted_rmse after opt=" << opt_report.final_error.weighted_rmse << "\n";
        std::cout << "  flips applied=" << opt_report.TotalFlipsApplied() << "\n";
        std::cout << "  vertex moves applied=" << opt_report.TotalVertexMovesApplied() << "\n";
    }

} // namespace

int main(int argc, char** argv) {
    AppConfig cfg{};
    if (!ParseArgs(argc, argv, cfg)) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        fs::create_directories(cfg.output_dir);
        std::cout << "Loading: " << cfg.input_path << "\n";
        const ImageLoadResult loaded = LoadRGBA8FromFile(cfg.input_path);
        if (!loaded.Ok()) {
            std::cerr << "Image load failed: " << loaded.error << "\n";
            return 2;
        }

        const ImageRGBA8 original_rgba = loaded.image;
        const ImageOKLaba reference = ConvertToOKLab(original_rgba);
        const EdgeMap edge_map = ComputeEdgeMapSobel(reference);
        const AutoProfile profile = BuildAutoProfile(reference, edge_map, cfg);

        Mesh mesh = CreateImageGridMesh(reference.Size(), profile.grid_x, profile.grid_y, {});
        for (auto& v : mesh.vertices) {
            v.color = SampleImageOKLabaBilinear(reference, v.position);
        }
        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
        if (!topo.ok) {
            std::cerr << "Topology build failed: " << topo.error << "\n";
            return 3;
        }

        PyramidRefinementOptions refine_opts{};
        refine_opts.bootstrap_splits = profile.bootstrap_splits;
        refine_opts.batch_size = profile.batch_size;
        refine_opts.max_splits = profile.max_splits;
        refine_opts.max_triangles = profile.max_triangles;
        refine_opts.target_mean_error = profile.target_error;
        refine_opts.min_error_to_split = Max(0.0008, profile.target_error * 0.06);
        refine_opts.split.min_edge_length = profile.min_edge_length;
        refine_opts.split.min_triangle_area = profile.min_triangle_area;
        refine_opts.split.min_triangle_bbox_extent = profile.min_triangle_bbox_extent;
        refine_opts.split.min_midpoint_separation = 1e-6;
        refine_opts.error.max_levels_used = 3;
        refine_opts.error.per_level_samples = 9;
        refine_opts.error.color_weight = 1.0;
        refine_opts.error.gradient_weight = 0.25;
        refine_opts.error.detail_weight = 0.30;
        refine_opts.error.structure_weight = 0.90;
        refine_opts.error.peak_weight = 0.75;
        refine_opts.plane_fit.interior_barycentric_samples = 1;

        refine_opts.safety.stop_on_progress_stall = cfg.stop_on_progress_stall;

        refine_opts.edge_bias.enabled = true;
        refine_opts.edge_bias.mean_weight = 0.60;
        refine_opts.edge_bias.peak_weight = 2.20;
        refine_opts.edge_bias.power = 1.15;
        refine_opts.edge_bias.strong_edge_threshold = 0.40;
        refine_opts.edge_bias.strong_edge_bonus = 0.90;
        refine_opts.edge_bias.max_multiplier = 4.00;

        std::cout << "  progress stall stop="
            << (refine_opts.safety.stop_on_progress_stall ? "enabled" : "disabled")
            << "\n";

        const auto refine_t0 = std::chrono::high_resolution_clock::now();
        PyramidRefinementReport refine_report = PyramidRefineMesh(mesh, reference, edge_map, refine_opts);
        const auto refine_t1 = std::chrono::high_resolution_clock::now();
        const auto refine_ms = std::chrono::duration_cast<std::chrono::milliseconds>(refine_t1 - refine_t0).count();
        WriteRefineStageStats(fs::path(cfg.output_dir) / "refine_stage_stats.txt", cfg, profile, refine_opts, reference, refine_report, refine_ms, mesh);
        PrintRefineStageSummary(refine_report, refine_ms, mesh);

        MeshOptimizationOptions opt_opts{};
        opt_opts.outer_iterations = profile.optimize_passes;
        opt_opts.flip.preserve_strong_edges = true;
        opt_opts.flip.strong_edge_threshold = 0.45;
        opt_opts.smooth.iterations = 1;
        opt_opts.smooth.lambda = 0.18;
        opt_opts.smooth.max_move_distance = 0.80;
        const auto optimize_t0 = std::chrono::high_resolution_clock::now();
        MeshOptimizationReport opt_report = OptimizeMesh(mesh, reference, opt_opts, &edge_map);
        const auto optimize_t1 = std::chrono::high_resolution_clock::now();
        const auto optimize_ms = std::chrono::duration_cast<std::chrono::milliseconds>(optimize_t1 - optimize_t0).count();
        WriteOptimizeStageStats(fs::path(cfg.output_dir) / "optimize_stage_stats.txt", optimize_ms, opt_report, mesh);
        PrintOptimizeStageSummary(optimize_ms, opt_report);

        const ScanlineIntegralStats plane_stats(reference);
        const auto plane_fit_t0 = std::chrono::high_resolution_clock::now();
        std::vector<TrianglePlane> planes = FitAllTrianglePlanes(mesh, reference, &plane_stats, refine_opts.plane_fit);
        const auto plane_fit_t1 = std::chrono::high_resolution_clock::now();
        SurfacePlaneRenderOptions preview_opts{};
        preview_opts.mode = SurfacePlaneRenderMode::PlaneShaded;
        SurfacePlaneRenderResult preview = RenderMeshPlaneSurface(mesh, planes, reference.Size(), preview_opts);

        SurfacePlaneRenderOptions wire_opts = preview_opts;
        wire_opts.overlay_wireframe = true;
        SurfacePlaneRenderResult wire = RenderMeshPlaneSurface(mesh, planes, reference.Size(), wire_opts);

        ReferenceRenderOptions tri_id_opts{};
        tri_id_opts.mode = ReferenceRenderMode::TriangleIdFlat;
        ReferenceRenderResult triangle_id = RenderMeshReference(mesh, reference.Size(), tri_id_opts);

        SurfacePlaneRenderOptions fit_err_opts{};
        fit_err_opts.mode = SurfacePlaneRenderMode::PlaneFitErrorTint;
        SurfacePlaneRenderResult fit_error_tint = RenderMeshPlaneSurface(mesh, planes, reference.Size(), fit_err_opts);

        const ImageOKLaba edge_preview = RenderEdgeMapPreview(edge_map, true);
        const ErrorEstimatorOptions exact_error_opts{};
        const auto exact_error_t0 = std::chrono::high_resolution_clock::now();
        const MeshErrorReport mesh_error = ComputeMeshError(mesh, reference, exact_error_opts);
        const auto exact_error_t1 = std::chrono::high_resolution_clock::now();
        const ImageOKLaba error_heatmap = RenderTriangleErrorHeatmap(
            mesh,
            reference.Size(),
            mesh_error.per_triangle,
            ErrorHeatmapMode::RedTint,
            0.0,
            exact_error_opts.skip_degenerate_triangles);

        const ImageSize upscaled_size = MakeUpscaledImageSize(reference.Size(), cfg.upscale_factor);
        SurfacePlaneRenderOptions preview_upscaled_opts = preview_opts;
        preview_upscaled_opts.transform.scale_x = static_cast<f64>(cfg.upscale_factor);
        preview_upscaled_opts.transform.scale_y = static_cast<f64>(cfg.upscale_factor);

        SurfacePlaneRenderOptions wire_upscaled_opts = wire_opts;
        wire_upscaled_opts.transform.scale_x = static_cast<f64>(cfg.upscale_factor);
        wire_upscaled_opts.transform.scale_y = static_cast<f64>(cfg.upscale_factor);

        SurfacePlaneRenderResult preview_upscaled = RenderMeshPlaneSurface(mesh, planes, upscaled_size, preview_upscaled_opts);
        SurfacePlaneRenderResult wire_upscaled = RenderMeshPlaneSurface(mesh, planes, upscaled_size, wire_upscaled_opts);

        const auto plane_fit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(plane_fit_t1 - plane_fit_t0).count();
        const auto exact_error_ms = std::chrono::duration_cast<std::chrono::milliseconds>(exact_error_t1 - exact_error_t0).count();

        SaveOKLabImage(fs::path(cfg.output_dir) / "preview.png", preview.image);
        SaveOKLabImage(fs::path(cfg.output_dir) / "preview_upscaled.png", preview_upscaled.image);
        WriteStats(fs::path(cfg.output_dir) / "mesh_stats.txt", cfg, profile, refine_opts, reference, refine_report, opt_report, mesh, preview, wire, edge_map);

        std::cout << "Render/export complete.\n";
        std::cout << "  refine engine=" << kRefineEngineName << "\n";
        std::cout << "  triangles=" << mesh.triangles.size() << "\n";
        std::cout << "  vertices=" << mesh.vertices.size() << "\n";
        std::cout << "  plane fit ms=" << plane_fit_ms << "\n";
        std::cout << "  exact error ms=" << exact_error_ms << "\n";
        std::cout << "  final weighted_rmse(after opt)=" << opt_report.final_error.weighted_rmse << "\n";
        std::cout << "  edge bias peak weight=" << refine_opts.edge_bias.peak_weight << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 10;
    }

    return 0;
}
