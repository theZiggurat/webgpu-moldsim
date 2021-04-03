[[group(0), binding(0)]] var trailSrc : [[access(read)]] texture_storage_2d<r32float>;
[[group(0), binding(1)]] var trailDst : [[access(write)]] texture_storage_2d<r32float>;

//const kernel: array<f32, 9> = array<f32, 9>(0.7, 0.8, 0.7, 0.8, 1.0, 0.8, 0.7, 0.8, 0.7);

[[stage(compute), workgroup_size(16, 16, 1)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    const coords : vec2<i32> = vec2<i32>(global_invocation_id.xy);
    //var bounds: vec2<i32> = textureDimensions(trailSrc);

    if (coords.x >= 1600 || coords.y >= 900) {
        return;
    }

    var avg: f32 = 0.0;
    for(var i : i32 = -1; i < 2; i = i + 1) {
        for(var j : i32 = -1; j < 2; j = j + 1) {
            var new_coord: vec2<i32> = coords + vec2<i32>(i, j);
            avg = avg + (textureLoad(trailSrc, new_coord).r);
        }
    }
    avg = avg / 9.0;

    //var color: vec4<f32> = vec4<f32>(avg, 0.0, 0.0, 1.0);
    var color: vec4<f32> = textureLoad(trailSrc, coords);
    textureStore(trailDst, coords, color);
}
