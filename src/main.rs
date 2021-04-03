use rand::Rng;
use wgpu::{include_spirv, util::DeviceExt};

#[path = "./framework.rs"]
mod framework;

const NUM_PARTICLES: u32 = 1024;
const PARTICLES_PER_GROUP: u32 = 64;

const SCREEN_SIZE: (u32, u32) = (1600, 900);

struct SimBuffers {
    particle_buffers: Vec<wgpu::Buffer>,
    trail_textures: Vec<wgpu::Texture>,
    vertices_buffer: wgpu::Buffer, 
}

struct SimBindGroups {
    particle_bind_groups: Vec<wgpu::BindGroup>,
    trail_decay_bind_groups: Vec<wgpu::BindGroup>,
    trail_diffuse_bind_groups: Vec<wgpu::BindGroup>,
    render_bind_groups: Vec<wgpu::BindGroup>,
}

struct SimPipelines {
    particle_bind_group_layout: wgpu::BindGroupLayout,
    diffuse_bind_group_layout: wgpu::BindGroupLayout,
    decay_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,
    particle_compute_pipeline: wgpu::ComputePipeline,
    trail_decay_compute_pipeline: wgpu::ComputePipeline,
    trail_diffuse_compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
}

struct MoldSim {
    buffers: SimBuffers,
    bind_groups: SimBindGroups,
    pipelines: SimPipelines,
    particle_work_group_count: u32,
    screen_work_group_count: (u32, u32),
    frame_num: usize,
}


impl framework::Framework for MoldSim {

    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {

        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Gl => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
            }
            _ => {} //TODO
        }

        let (compute_shader, decay_shader, diffuse_shader, draw_shader) = create_shaders(&device);

        let sim_param_data: Vec<f32> = [
            1./144., // deltaT
            0.1,     // rule1Distance
            0.025,   // rule2Distance
            0.025,   // rule3Distance
            0.02,    // rule1Scale
            0.05,    // rule2Scale
            0.005,   // rule3Scale
        ]
        .to_vec();

        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            contents: bytemuck::cast_slice(&sim_param_data),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });


        let texture_size = wgpu::Extent3d {
            width: SCREEN_SIZE.0,
            height: SCREEN_SIZE.1,
            depth: 1,
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let pipelines = {

            let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (sim_param_data.len() * std::mem::size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                ],
                label: None,
            });

            let decay_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                ],
                label: None,
            });

            let diffuse_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                ],
                label: None,
            });

            let render_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: false,
                        },
                        count: None,
                    },
                ],
                label: None,
            });

            let particle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle"),
                bind_group_layouts: &[&particle_bind_group_layout],
                push_constant_ranges: &[],
            });

            let decay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("decay"),
                bind_group_layouts: &[&decay_bind_group_layout],
                push_constant_ranges: &[],
            });

            let diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("diffuse"),
                bind_group_layouts: &[&diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });

            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render"),
                    bind_group_layouts: &[&render_bind_group_layout],
                    push_constant_ranges: &[],
            });

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: "main",
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 2 * 4,
                            step_mode: wgpu::InputStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float2],
                        },
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_shader,
                    entry_point: "main",
                    targets: &[sc_desc.format.into()],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
            });
    
            let particle_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Particle compute pipeline"),
                layout: Some(&particle_pipeline_layout),
                module: &compute_shader,
                entry_point: "main",
            });
    
            let trail_decay_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Decay compute pipeline"),
                layout: Some(&decay_pipeline_layout),
                module: &decay_shader,
                entry_point: "main",
            });
    
            let trail_diffuse_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Diffuse compute pipeline"),
                layout: Some(&diffuse_pipeline_layout),
                module: &diffuse_shader,
                entry_point: "main",
            });

            SimPipelines {
                particle_bind_group_layout,
                diffuse_bind_group_layout,
                decay_bind_group_layout,
                render_bind_group_layout,
                particle_compute_pipeline,
                trail_decay_compute_pipeline,
                trail_diffuse_compute_pipeline,
                render_pipeline
            }

        };
        

        let buffers = {
            let vertex_buffer_data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
            let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::bytes_of(&vertex_buffer_data),
                usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
            });
            let mut particle_buffers = Vec::<wgpu::Buffer>::new();
            let mut trail_textures = Vec::<wgpu::Texture>::new();
    
            let mut rng = rand::thread_rng();
            let mut initial_particle_data = vec![0.0f32; (4 * NUM_PARTICLES) as usize];
            for particle_instance_chunk in initial_particle_data.chunks_mut(4) {
                particle_instance_chunk[0] = rng.gen::<f32>() * 2.0 - 1.0;
                particle_instance_chunk[1] = rng.gen::<f32>() * 2.0 - 1.0;
                particle_instance_chunk[2] = rng.gen::<f32>() * 2.0 - 1.0;
                particle_instance_chunk[3] = rng.gen::<f32>() * 2.0 - 1.0;
            }

            for i in 0..2 {
                particle_buffers.push(
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("Particle Buffer {}", i)),
                        contents: bytemuck::cast_slice(&initial_particle_data),
                        usage: wgpu::BufferUsage::STORAGE
                            | wgpu::BufferUsage::COPY_DST
                            | wgpu::BufferUsage::COPY_SRC,
                    }),
                );
    
                trail_textures.push(device.create_texture(
                    &wgpu::TextureDescriptor {
                        label: Some(&format!("Trail Texture {}", i)),
                        size: texture_size,
                        mip_level_count: 1, 
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
                    }
                ));
            }

            SimBuffers {
                vertices_buffer,
                particle_buffers,
                trail_textures
            }
        };

        let bind_groups = {

            let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
            let mut trail_decay_bind_groups = Vec::<wgpu::BindGroup>::new();
            let mut trail_diffuse_bind_groups = Vec::<wgpu::BindGroup>::new();
            let mut render_bind_groups = Vec::<wgpu::BindGroup>::new();

            let desc = Default::default();

            for i in 0..2 {
                particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &pipelines.particle_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: sim_param_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.particle_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffers.particle_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[i].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[(i + 1) % 2].create_view(&desc)), // bind to opposite buffer
                        },
                    ],
                    label: None,
                }));
    
                trail_decay_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &pipelines.decay_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[(i + 1) % 2].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[i].create_view(&desc)), // bind to opposite buffer
                        },
                    ],
                    label: None,
                }));
    
                trail_diffuse_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &pipelines.diffuse_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[i].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[(i + 1) % 2].create_view(&desc)), // bind to opposite buffer
                        },
                    ],
                    label: None,
                }));
    
                render_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &pipelines.render_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[(i + 1) % 2].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                    label: None,
                }));
            }

            SimBindGroups {
                particle_bind_groups,
                trail_decay_bind_groups,
                trail_diffuse_bind_groups,
                render_bind_groups,
            }

        };

        // calculates number of work groups from PARTICLES_PER_GROUP constant
        let particle_work_group_count =
            ((NUM_PARTICLES as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        let screen_work_group_count: (u32, u32) = 
            ((SCREEN_SIZE.0 as f32 / 16.0).ceil() as u32, (SCREEN_SIZE.1 as f32 / 16.0).ceil() as u32);

        println!("{:?}", screen_work_group_count);

        MoldSim {
            buffers,
            bind_groups,
            pipelines,
            particle_work_group_count,
            screen_work_group_count,
            frame_num: 0
        }
    }

    /// update is called for any WindowEvent not handled by the framework
    fn update(&mut self, _event: winit::event::WindowEvent) {}

    /// resize is called on WindowEvent::Resized events
    fn resize(
        &mut self,
        _sc_desc: &wgpu::SwapChainDescriptor,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {



    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {

        let color_attachments = [wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &frame.view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            },
        }];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
        };


        // get command encoder
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute buffer
        command_encoder.push_debug_group("compute particle movement");
        {
            // compute pass
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipelines.particle_compute_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.particle_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch(self.particle_work_group_count, 1, 1);
        }
        command_encoder.pop_debug_group();

        command_encoder.push_debug_group("compute trail decay");
        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipelines.trail_decay_compute_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.trail_decay_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch(self.screen_work_group_count.0, self.screen_work_group_count.1, 1);
        }
        command_encoder.pop_debug_group();

        command_encoder.push_debug_group("compute trail diffuse");
        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipelines.trail_diffuse_compute_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.trail_diffuse_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch(self.screen_work_group_count.0, self.screen_work_group_count.1, 1);
        }
        command_encoder.pop_debug_group();

        // render texture to screen
        command_encoder.push_debug_group("render boids");
        {
            // render pass
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.pipelines.render_pipeline);
            // render dst particles
            // the three instance-local vertices
            rpass.set_vertex_buffer(0, self.buffers.vertices_buffer.slice(..));
            rpass.set_bind_group(0, &self.bind_groups.render_bind_groups[self.frame_num % 2], &[]);
            rpass.draw(0..6, 0..1);
        }
        command_encoder.pop_debug_group();

        // update frame count
        self.frame_num += 1;

        // done
        queue.submit(Some(command_encoder.finish()));

        // to update uniform buffers
        // self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }
}

#[cfg(debug_assertions)]
fn create_shaders(device: &wgpu::Device) -> (wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule) {

    use std::io::prelude::*;
    use std::fs::File;

    let mut file = File::open("./resources/spirv/compute.spv").unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let compute_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("compute shader"),
        source: wgpu::util::make_spirv(&buf[..]),
        flags: wgpu::ShaderFlags::VALIDATION
    });

    let mut file = File::open("./resources/spirv/decay.spv").unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let decay_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("decay shader"),
        source: wgpu::util::make_spirv(&buf[..]),
        flags: wgpu::ShaderFlags::VALIDATION
    });

    let mut file = File::open("./resources/spirv/diffuse.spv").unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let diffuse_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("diffuse shader"),
        source: wgpu::util::make_spirv(&buf[..]),
        flags: wgpu::ShaderFlags::VALIDATION
    });

    let mut file = File::open("./resources/spirv/draw.spv").unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let draw_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("draw shader"),
        source: wgpu::util::make_spirv(&buf[..]),
        flags: wgpu::ShaderFlags::VALIDATION
    });
    (compute_shader, decay_shader, diffuse_shader, draw_shader)
}

#[cfg(not(debug_assertions))]
fn create_shaders(device: &wgpu::Device) -> (wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule, wgpu::ShaderModule) {
    let compute_shader = device.create_shader_module(&include_spirv!("../resources/spirv/compute.spv"));
    let decay_shader = device.create_shader_module(&include_spirv!("../resources/spirv/decay.spv"));
    let diffuse_shader = device.create_shader_module(&include_spirv!("../resources/spirv/diffuse.spv"));
    let draw_shader = device.create_shader_module(&include_spirv!("../resources/spirv/draw.spv"));
    (compute_shader, decay_shader, diffuse_shader, draw_shader)
}

/// run example
fn main() {
    framework::run::<MoldSim>("Mold sim");
}
