use serde::{Deserialize, Serialize};
use crate::uniform::Uniform;

#[derive(Serialize, Deserialize, Debug)]
pub struct ParamManager {
    pub current: usize,
    pub params: Vec<Params>,
    pub global: GlobalParams,
}

impl ParamManager {
    pub fn current(&self) -> &Params {
        self.params.get(self.current).unwrap()
    }

    pub fn current_mut(&mut self) -> &mut Params {
        self.params.get_mut(self.current).unwrap()
    }

    pub fn new(&mut self) {
        self.params.push(self.current().clone());
        self.current = self.params.len() - 1;
        self.current_mut().name = format!("Custom {}", self.current);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Params {
    pub name: String,
    pub particle: ParticleParams,
    pub decay: DecayParams,
    pub diffuse: DiffuseParams,
    pub render: RenderParams,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct GlobalParams {
    pub post_enabled: bool,
    pub max_particles: u32,
}

#[repr(C, packed)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ParticleParams {
    pub trail_power: f32,
    pub speed: f32,
    pub sensor_angle: f32,
    pub sensor_distance: f32,
    pub turn_speed: f32,
    pub num_particles: u32,
}

#[repr(C, packed)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct DecayParams {
    pub decay_rate: f32,

}
#[repr(C, packed)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct DiffuseParams {
    pub diffuse_amount: f32,
}

#[repr(C, packed)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct RenderParams {
    pub color_1: [f32; 3],
    pub color_2: [f32; 3],
    pub color_pow: f32,
    pub cutoff: f32,
}

impl Uniform for ParticleParams {}
impl Uniform for DecayParams {}
impl Uniform for DiffuseParams {}
impl Uniform for RenderParams {}

impl ParamManager {
    pub fn from_json(path: &str) -> ParamManager {
        use std::io::prelude::*;
        use std::fs::File;

        let mut file = File::open(path).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        
        serde_json::from_slice(&buf).unwrap()
    }

    pub fn save(&self, path: &str) {
        match std::fs::write(path, serde_json::to_vec_pretty(&self).unwrap().as_slice()) {
            Err(e) => log::error!("Error saving params to file: {}", e),
            _ => ()
        }
    }
}
