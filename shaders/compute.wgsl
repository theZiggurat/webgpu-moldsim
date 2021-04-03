// This should match `NUM_PARTICLES` on the Rust side.
const NUM_PARTICLES: u32 = 1024u;
const PI: f32 = 3.14159265358979323846264338327950288f32;

struct Particle {
  pos : vec2<f32>;
  vel : vec2<f32>;
};

[[block]]
struct SimParams {
  deltaT : f32;
  rule1Distance : f32;
  rule2Distance : f32;
  rule3Distance : f32;
  rule1Scale : f32;
  rule2Scale : f32;
  rule3Scale : f32;
};

[[block]]
struct Particles {
  particles : [[stride(16)]] array<Particle>;
};

[[group(0), binding(0)]] var<uniform> params : SimParams;
[[group(0), binding(1)]] var<storage> particlesSrc : [[access(read)]] Particles;
[[group(0), binding(2)]] var<storage> particlesDst : [[access(read_write)]] Particles;
[[group(0), binding(3)]] var trailSrc : [[access(read)]] texture_storage_2d<r32float>;
[[group(0), binding(4)]] var trailDst : [[access(write)]] texture_storage_2d<r32float>;


fn rotate(vec: vec2<f32>, ang: f32) -> vec2<f32> {
  var s: f32 = sin(ang);
  var c: f32 = cos(ang);
  var mat: mat2x2<f32> = mat2x2<f32>(vec2<f32>(c, s), vec2<f32>(-s, c));
  return mat * vec;
}


[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
  const index : u32 = global_invocation_id.x;
  if (index >= NUM_PARTICLES) {
    return;
  }

  var vPos : vec2<f32> = particlesSrc.particles[index].pos;
  var vVel : vec2<f32> = particlesSrc.particles[index].vel;

  var trail_power: f32 = 32.0;
  var speed: f32 = 0.001;
  var sensor_angle: f32 = PI / 6.0;
  var sensor_distance: f32 = 0.005;

  vVel = normalize(vVel);

  var sens_left: vec2<f32> = rotate(vVel, sensor_angle);
  var sens_right: vec2<f32> = rotate(vVel, -sensor_angle);


  // var cMass : vec2<f32> = vec2<f32>(0.0, 0.0);
  // var cVel : vec2<f32> = vec2<f32>(0.0, 0.0);
  // var colVel : vec2<f32> = vec2<f32>(0.0, 0.0);
  // var cMassCount : i32 = 0;
  // var cVelCount : i32 = 0;

  // var pos : vec2<f32>;
  // var vel : vec2<f32>;
  // var i : u32 = 0u;
  // loop {
  //   if (i >= NUM_PARTICLES) {
  //     break;
  //   }
  //   if (i == index) {
  //     continue;
  //   }

  //   pos = particlesSrc.particles[i].pos;
  //   vel = particlesSrc.particles[i].vel;

  //   if (distance(pos, vPos) < params.rule1Distance) {
  //     cMass = cMass + pos;
  //     cMassCount = cMassCount + 1;
  //   }
  //   if (distance(pos, vPos) < params.rule2Distance) {
  //     colVel = colVel - (pos - vPos);
  //   }
  //   if (distance(pos, vPos) < params.rule3Distance) {
  //     cVel = cVel + vel;
  //     cVelCount = cVelCount + 1;
  //   }

  //   continuing {
  //     i = i + 1u;
  //   }
  // }
  // if (cMassCount > 0) {
  //   cMass = cMass * (1.0 / f32(cMassCount)) - vPos;
  // }
  // if (cVelCount > 0) {
  //   cVel = cVel * (1.0 / f32(cVelCount));
  // }

  // vVel = vVel + (cMass * params.rule1Scale) +
  //     (colVel * params.rule2Scale) +
  //     (cVel * params.rule3Scale);

  // // clamp velocity for a more pleasing simulation
  // vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);

  // // kinematic update
  // vPos = vPos + (vVel * params.deltaT);

  // Wrap around boundary
  

  // Write back
  var vPos_new: vec2<f32> = vPos + (speed * vVel);

  if (vPos_new.x <= -1.0) {
    vPos_new.x = 1.0;
  }
  if (vPos_new.x >= 1.0) {
    vPos_new.x = -1.0;
  }
  if (vPos_new.y <= -1.0) {
    vPos_new.y = 1.0;
  }
  if (vPos_new.y >= 1.0) {
    vPos_new.y = -1.0;
  }

 

  particlesDst.particles[index].pos = vPos_new;
  particlesDst.particles[index].vel = vVel;

  var vPos_norm: vec2<f32> = vec2<f32>((vPos_new.x + 1.0) / 2., (vPos_new.y + 1.0) / 2.);

  var particle_pixel_index: vec2<i32> = vec2<i32>(i32(vPos_norm.x * 1600.0), i32(vPos_norm.y * 900.0));
  var trail: vec4<f32> = textureLoad(trailSrc, particle_pixel_index);
  trail.r = trail.r + (1.0 / 144.0 * trail_power);

  textureStore(trailDst, particle_pixel_index, trail);
}

// fn rotate(vec: vec2<f32>, amt: f32) -> vec2<f32> {
//   return vec2<f32>(0.0);
// }