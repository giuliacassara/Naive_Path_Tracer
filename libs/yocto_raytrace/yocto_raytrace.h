//
// # Yocto/RayTrace: Tiny raytracer tracer
//
//
// Yocto/RayTrace is a simple ray tracing library with support for microfacet
// materials, area and environment lights, and advacned sampling.
//
//
// ## Physically-based Path Tracing
//
// Yocto/Trace includes a tiny, but fully featured, path tracer with support for
// textured mesh area lights, GGX materials, environment mapping. The algorithm
// makes heavy use of MIS for fast convergence.
// The interface supports progressive parallel execution both synchronously,
// for CLI applications, and asynchronously for interactive viewing.
//
// Materials are represented as sums of an emission term, a diffuse term and
// a specular microfacet term (GGX or Phong), and a transmission term for
// this sheet glass.
// Lights are defined as any shape with a material emission term. Additionally
// one can also add environment maps. But even if you can, you might want to
// add a large triangle mesh with inward normals instead. The latter is more
// general (you can even more an arbitrary shape sun). For now only the first
// environment is used.
//
// 1. prepare the ray-tracing acceleration structure with `build_bvh()`
// 2. prepare lights for rendering with `init_trace_lights()`
// 3. create the random number generators with `init_trace_state()`
// 4. render blocks of samples with `trace_samples()`
// 5. you can also start an asynchronous renderer with `trace_asynch_start()`
//
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2020 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

// TODO: lines
// TODO: opacity
// TODO: plastics
// TODO: progress

#ifndef _YOCTO_RAYTRACE_H_
#define _YOCTO_RAYTRACE_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include <yocto/yocto_image.h>
#include <yocto/yocto_math.h>
#include <yocto/yocto_common.h>

#include <atomic>
#include <future>
#include <memory>

// -----------------------------------------------------------------------------
// ALIASES
// -----------------------------------------------------------------------------
namespace yocto::raytrace {

// Namespace aliases
namespace rtr = yocto::raytrace;
namespace img = yocto::image;
namespace common = yocto::common;

using common::range;


// Math defitions
using math::bbox3f;
using math::byte;
using math::frame3f;
using math::identity3x4f;
using math::ray3f;
using math::rng_state;
using math::vec2f;
using math::vec2i;
using math::vec3b;
using math::vec3f;
using math::vec3i;
using math::vec4f;
using math::vec4i;
using math::zero2f;
using math::zero3f;
using math::rand2f;
using math::transform_point;
using math::transform_direction;
using math::make_rng;
using math::camera_ray;
using math::normalize;
using math::interpolate_triangle;
using math::interpolate_quad;
using math::inverse;
using math::pi;

}  // namespace yocto::raytrace

// -----------------------------------------------------------------------------
// HIGH LEVEL API
// -----------------------------------------------------------------------------
namespace yocto::raytrace {

// Trace scene
struct scene;
struct camera;
struct environment;
struct shape;
struct texture;
struct material;
struct object;

// Add scene elements
rtr::camera*      add_camera(rtr::scene* scene);
rtr::object*      add_object(rtr::scene* scene);
rtr::texture*     add_texture(rtr::scene* scene);
rtr::material*    add_material(rtr::scene* scene);
rtr::shape*       add_shape(rtr::scene* scene);
rtr::environment* add_environment(rtr::scene* scene);

// camera properties
void set_frame(rtr::camera* camera, const frame3f& frame);
void set_lens(rtr::camera* camera, float lens, float aspect, float film);
void set_focus(rtr::camera* camera, float aperture, float focus);

// object properties
void set_frame(rtr::object* object, const frame3f& frame);
void set_material(rtr::object* object, rtr::material* material);
void set_shape(rtr::object* object, rtr::shape* shape);

// texture properties
void set_texture(rtr::texture* texture, const img::image<vec3b>& img);
void set_texture(rtr::texture* texture, const img::image<vec3f>& img);
void set_texture(rtr::texture* texture, const img::image<byte>& img);
void set_texture(rtr::texture* texture, const img::image<float>& img);

// material properties
void set_emission(rtr::material* material, const vec3f& emission,
    rtr::texture* emission_tex = nullptr);
void set_color(rtr::material* material, const vec3f& color,
    rtr::texture* color_tex = nullptr);
void set_specular(rtr::material* material, float specular = 1,
    rtr::texture* specular_tex = nullptr);
void set_ior(rtr::material* material, float ior);
void set_metallic(rtr::material* material, float metallic,
    rtr::texture* metallic_tex = nullptr);
void set_transmission(rtr::material* material, float transmission, bool thin,
    float trdepth, rtr::texture* transmission_tex = nullptr);
void set_roughness(rtr::material* material, float roughness,
    rtr::texture* roughness_tex = nullptr);
void set_opacity(rtr::material* material, float opacity,
    rtr::texture* opacity_tex = nullptr);
void set_thin(rtr::material* material, bool thin);
void set_scattering(rtr::material* material, const vec3f& scattering,
    float scanisotropy, rtr::texture* scattering_tex = nullptr);

// shape properties
void set_points(rtr::shape* shape, const std::vector<int>& points);
void set_lines(rtr::shape* shape, const std::vector<vec2i>& lines);
void set_triangles(rtr::shape* shape, const std::vector<vec3i>& triangles);
void set_positions(rtr::shape* shape, const std::vector<vec3f>& positions);
void set_normals(rtr::shape* shape, const std::vector<vec3f>& normals);
void set_texcoords(rtr::shape* shape, const std::vector<vec2f>& texcoords);
void set_radius(rtr::shape* shape, const std::vector<float>& radius);

// environment properties
void set_frame(rtr::environment* environment, const frame3f& frame);
void set_emission(rtr::environment* environment, const vec3f& emission,
    rtr::texture* emission_tex = nullptr);

// Type of tracing algorithm
enum struct shader_type {
  // clang-format on
  raytrace,  // path tracing
  eyelight,  // eyelight rendering
  normal,    // normals
  texcoord,  // texcoords
  color,     // colors
             // clang-format off
};

// Default trace seed
const auto default_seed = 961748941ull;

// Options for trace functions
struct trace_params {
  int             resolution = 720;
  shader_type     shader     = shader_type::raytrace;
  int             samples    = 512;
  int             bounces    = 4;
  float           clamp      = 100000;
  uint64_t        seed       = default_seed;
  bool            noparallel = false;
  int             pratio     = 8;
};

const auto shader_names = std::vector<std::string>{
    "raytrace", "eyelight", "normal", "texcoord", "color"};

// Progress report callback
using progress_callback =
    std::function<void(const std::string& message, int current, int total)>;

// Build the bvh acceleration structure.
void init_bvh(rtr::scene* scene, const trace_params& params,
    progress_callback progress_cb = {});

// Initialize the rendering state
struct state;
void init_state(rtr::state* state, const rtr::scene* scene,
    const rtr::camera* camera, const trace_params& params);

// Progressively computes an image.
void trace_samples(rtr::state* state, 
    const rtr::scene* scene, const rtr::camera* camera, 
    const trace_params& params);

// Progressively computes an image. Stop if requested.
void trace_samples(rtr::state* state, 
    const rtr::scene* scene, const rtr::camera* camera, 
    const trace_params& params, std::atomic<bool>* stop);

}  // namespace yocto::raytrace

// -----------------------------------------------------------------------------
// SCENE AND RENDERING DATA
// -----------------------------------------------------------------------------
namespace yocto::raytrace {

// BVH tree node containing its bounds, indices to the BVH arrays of either
// primitives or internal nodes, the node element type,
// and the split axis. Leaf and internal nodes are identical, except that
// indices refer to primitives for leaf nodes or other nodes for internal nodes.
struct bvh_node {
  bbox3f bbox;
  int    start;
  short  num;
  bool   internal;
  byte   axis;
};

// BVH tree stored as a node array with the tree structure is encoded using
// array indices. BVH nodes indices refer to either the node array,
// for internal nodes, or the primitive arrays, for leaf nodes.
// Application data is not stored explicitly.
struct bvh_tree {
  std::vector<bvh_node> nodes      = {};
  std::vector<int>      primitives = {};
};

// Camera based on a simple lens model. The camera is placed using a frame.
// Camera projection is described in photorgaphics terms. In particular,
// we specify fil size (35mm by default), the lens' focal length, the focus
// distance and the lens aperture. All values are in meters.
// Here are some common aspect ratios used in video and still photography.
// 3:2    on 35 mm:  0.036 x 0.024
// 16:9   on 35 mm:  0.036 x 0.02025 or 0.04267 x 0.024
// 2.35:1 on 35 mm:  0.036 x 0.01532 or 0.05640 x 0.024
// 2.39:1 on 35 mm:  0.036 x 0.01506 or 0.05736 x 0.024
// 2.4:1  on 35 mm:  0.036 x 0.015   or 0.05760 x 0.024 (approx. 2.39 : 1)
// To compute good apertures, one can use the F-stop number from phostography
// and set the aperture to focal_leangth/f_stop.
struct camera {
  frame3f frame        = identity3x4f;
  float   lens         = 0.050;
  vec2f   film         = {0.036, 0.024};
  float   focus        = 10000;
  float   aperture     = 0;
};

// Texture containing either an LDR or HDR image. HdR images are encoded
// in linear color space, while LDRs are encoded as sRGB.
struct texture {
  img::image<vec3f> colorf  = {};
  img::image<vec3b> colorb  = {};
  img::image<float> scalarf = {};
  img::image<byte>  scalarb = {};
};

// Material for surfaces, lines and triangles.
// For surfaces, uses a microfacet model with thin sheet transmission.
// The model is based on OBJ, but contains glTF compatibility.
// For the documentation on the values, please see the OBJ format.
struct material {
  // material
  vec3f emission     = {0, 0, 0};
  vec3f color        = {0, 0, 0};
  float specular     = 0;
  float roughness    = 0;
  float metallic     = 0;
  float ior          = 1.5;
  vec3f spectint     = {1, 1, 1};
  float transmission = 0;
  vec3f scattering   = {0, 0, 0};
  float scanisotropy = 0;
  float trdepth      = 0.01;
  float opacity      = 1;
  bool  thin         = false;

  // textures
  rtr::texture* emission_tex     = nullptr;
  rtr::texture* color_tex        = nullptr;
  rtr::texture* specular_tex     = nullptr;
  rtr::texture* metallic_tex     = nullptr;
  rtr::texture* roughness_tex    = nullptr;
  rtr::texture* transmission_tex = nullptr;
  rtr::texture* spectint_tex     = nullptr;
  rtr::texture* scattering_tex   = nullptr;
  rtr::texture* opacity_tex      = nullptr;
};

// Shape data represented as an indexed meshes of elements.
// May contain either points, lines, triangles and quads.
// Additionally, we support faceavarying primitives where
// each verftex data has its own topology.
struct shape {
  // primitives
  std::vector<int>   points    = {};
  std::vector<vec2i> lines     = {};
  std::vector<vec3i> triangles = {};

  // vertex data
  std::vector<vec3f> positions = {};
  std::vector<vec3f> normals   = {};
  std::vector<vec2f> texcoords = {};
  std::vector<float> radius    = {};

  // computed properties
  bvh_tree* bvh = nullptr;

  // cleanup
  ~shape();
};

// Object.
struct object {
  frame3f         frame    = identity3x4f;
  rtr::shape*    shape    = nullptr;
  rtr::material* material = nullptr;
};

// Environment map.
struct environment {
  frame3f            frame        = identity3x4f;
  vec3f              emission     = {0, 0, 0};
  rtr::texture*     emission_tex = nullptr;
};

// Scene comprised an array of objects whose memory is owened by the scene.
// All members are optional,Scene objects (camera, instances, environments)
// have transforms defined internally. A scene can optionally contain a
// node hierarchy where each node might point to a camera, instance or
// environment. In that case, the element transforms are computed from
// the hierarchy. Animation is also optional, with keyframe data that
// updates node transformations only if defined.
struct scene {
  std::vector<rtr::camera*>      cameras      = {};
  std::vector<rtr::object*>      objects      = {};
  std::vector<rtr::shape*>       shapes       = {};
  std::vector<rtr::material*>    materials    = {};
  std::vector<rtr::texture*>     textures     = {};
  std::vector<rtr::environment*> environments = {};

  // computed properties
  bvh_tree*                  bvh    = nullptr;

  // cleanup
  ~scene();
};

// State of a pixel during tracing
struct pixel {
  vec4f         accumulated = {0, 0, 0, 0};
  int           samples  = 0;
  rng_state rng      = {};
};

// Rendering state
struct state {
  img::image<vec4f> render = {};
  img::image<pixel> pixels = {};
};

}  // namespace yocto::raytrace

// -----------------------------------------------------------------------------
// INTERSECTION
// -----------------------------------------------------------------------------
namespace yocto::raytrace {

// Results of intersect functions that include hit flag, the instance id,
// the shape element id, the shape element uv and intersection distance.
// Results values are set only if hit is true.
struct intersection3f {
  int   object   = -1;
  int   element  = -1;
  vec2f uv       = {0, 0};
  float distance = 0;
  bool  hit      = false;
};

// Intersect ray with a bvh returning either the first or any intersection
// depending on `find_any`. Returns the ray distance , the instance id,
// the shape element index and the element barycentric coordinates.
intersection3f intersect_scene_bvh(const rtr::scene* scene, const ray3f& ray,
    bool find_any = false, bool non_rigid_frames = true);
intersection3f intersect_instance_bvh(const rtr::object* object,
    const ray3f& ray, bool find_any = false, bool non_rigid_frames = true);

}  // namespace yocto::raytrace

#endif
