struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn main(
    [[location(0)]] position: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    return out;
}

[[group(0), binding(0)]]
var r_color: texture_2d<f32>;
[[group(0), binding(1)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var uv: vec2<f32> = vec2<f32>(in.position.x / 1600.0 / 2.0, in.position.y / 900.0 / 2.0);
    var color: f32 = textureSample(r_color, r_sampler, uv).r;
    //var new_color: f32 = clamp(color, 0.2, 0.8);
    return vec4<f32>(color, 0.0, 0.0, 1.0);
}
