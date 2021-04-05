[[block]]
struct SimParams {
  decaySpeed: f32;
};

[[group(0), binding(0)]] var<uniform> params : SimParams;
[[group(0), binding(1)]] var trailSrc : [[access(read)]] texture_storage_2d<r32float>;
[[group(0), binding(2)]] var trailDst : [[access(write)]] texture_storage_2d<r32float>;


[[stage(compute), workgroup_size(16, 16, 1)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    const coords : vec2<i32> = vec2<i32>(global_invocation_id.xy);

    if (coords.x >= 3200 || coords.y >= 1800) {
        return;
    }

    var color: vec4<f32> = textureLoad(trailSrc, coords);
    color.r = color.r * params.decaySpeed;

    textureStore(trailDst, coords, color);
}
