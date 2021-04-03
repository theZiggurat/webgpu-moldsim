[[group(0), binding(0)]] var trailSrc : [[access(read)]] texture_storage_2d<r32float>;
[[group(0), binding(1)]] var trailDst : [[access(write)]] texture_storage_2d<r32float>;

[[stage(compute), workgroup_size(16, 16, 1)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    const coords : vec2<i32> = vec2<i32>(global_invocation_id.xy);
    //var bounds: vec2<i32> = textureDimensions(trailSrc);

    if (coords.x >= 1600 || coords.y >= 900) {
        return;
    }

    var color: vec4<f32> = textureLoad(trailSrc, coords);
    //color.r = color.r * 0.92;

    // var col: f32 = f32(coords.y) / 900.0;
    // var color: vec4<f32> = vec4<f32>(col, 0.0, 0.0, 1.0);

    textureStore(trailDst, coords, color);
}
