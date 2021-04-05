use params::ParamManager;
use rand::Rng;
use wgpu::util::DeviceExt;
use crate::uniform::Uniform;

#[path = "./framework.rs"]
mod framework;
mod util;
mod params;
mod uniform;

const PARTICLES_PER_GROUP: u32 = 64;
const SCREEN_SIZE: (u32, u32) = (3200, 1800);

struct SimBuffers {
    particle_buffers: Vec<wgpu::Buffer>,
    trail_textures: Vec<wgpu::Texture>,
    vertices_buffer: wgpu::Buffer, 
    particle_uniform: wgpu::Buffer,
    decay_uniform: wgpu::Buffer,
    diffuse_uniform: wgpu::Buffer,
    render_uniform: wgpu::Buffer,
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
    params: ParamManager,
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

        let params = params::ParamManager::from_json("./resources/params.json");

        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Gl => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
            }
            _ => {} //TODO
        }

        let (compute_shader, decay_shader, diffuse_shader, draw_shader) = 
        (
            crate::util::create_shader(device, "./resources/spirv/compute.spv"),
            crate::util::create_shader(device, "./resources/spirv/decay.spv"),
            crate::util::create_shader(device, "./resources/spirv/diffuse.spv"),
            crate::util::create_shader(device, "./resources/spirv/draw.spv")
        );

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

            log::info!("Creating particle bind group...");
            let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: params.current().particle.memsize(),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((params.global.max_particles * 16) as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((params.global.max_particles * 16) as _),
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

            log::info!("Creating decay bind group...");
            let decay_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: params.current().decay.memsize(),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

            log::info!("Creating diffuse bind group...");
            let diffuse_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: params.current().diffuse.memsize(),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

            log::info!("Creating render bind group...");
            let render_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: params.current().render.memsize(),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

            log::info!("Creating particle pipeline layout...");
            let particle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle"),
                bind_group_layouts: &[&particle_bind_group_layout],
                push_constant_ranges: &[],
            });

            log::info!("Creating decay pipeline layout...");
            let decay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("decay"),
                bind_group_layouts: &[&decay_bind_group_layout],
                push_constant_ranges: &[],
            });

            log::info!("Creating diffuse pipeline layout...");
            let diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("diffuse"),
                bind_group_layouts: &[&diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });

            log::info!("Creating render pipeline layout...");
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render"),
                    bind_group_layouts: &[&render_bind_group_layout],
                    push_constant_ranges: &[],
            });

            log::info!("Creating particle pipeline...");
            let particle_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Particle compute pipeline"),
                layout: Some(&particle_pipeline_layout),
                module: &compute_shader,
                entry_point: "main",
            });
    
            log::info!("Creating decay pipeline...");
            let trail_decay_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Decay compute pipeline"),
                layout: Some(&decay_pipeline_layout),
                module: &decay_shader,
                entry_point: "main",
            });
    
            log::info!("Creating diffuse pipeline...");
            let trail_diffuse_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Diffuse compute pipeline"),
                layout: Some(&diffuse_pipeline_layout),
                module: &diffuse_shader,
                entry_point: "main",
            });

            log::info!("Creating render pipeline...");
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
            let mut initial_particle_data = vec![0.0f32; (4 * params.global.max_particles) as usize];
            for particle_instance_chunk in initial_particle_data.chunks_mut(4) {
                particle_instance_chunk[0] = rng.gen::<f32>();
                particle_instance_chunk[1] = rng.gen::<f32>();
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
                        usage: wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_SRC | wgpu::TextureUsage::COPY_DST,
                    }
                ));
            }

            let particle_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Parameter Buffer"),
                contents: params.current().particle.to_bytes(),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

            let decay_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Parameter Buffer"),
                contents: params.current().decay.to_bytes(),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

            let diffuse_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Parameter Buffer"),
                contents: params.current().diffuse.to_bytes(),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

            let render_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Parameter Buffer"),
                contents: params.current().render.to_bytes(),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

            SimBuffers {
                vertices_buffer,
                particle_buffers,
                trail_textures,
                particle_uniform,
                decay_uniform,
                diffuse_uniform,
                render_uniform
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
                            resource: buffers.particle_uniform.as_entire_binding(),
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
                            resource: buffers.decay_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[(i + 1) % 2].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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
                            resource: buffers.diffuse_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[i].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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
                            resource: buffers.render_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&buffers.trail_textures[(i + 1) % 2].create_view(&desc)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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
            ((params.current().particle.num_particles as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        let screen_work_group_count: (u32, u32) = 
            ((SCREEN_SIZE.0 as f32 / 16.0).ceil() as u32, (SCREEN_SIZE.1 as f32 / 16.0).ceil() as u32);

        log::info!("Particle work group count: {:?}", (particle_work_group_count, particle_work_group_count));
        log::info!("Screen work group count: {:?}", screen_work_group_count);

        MoldSim {
            params,
            buffers,
            bind_groups,
            pipelines,
            particle_work_group_count,
            screen_work_group_count,
            frame_num: 0,
        }
    }

    /// update is called for any WindowEvent not handled by the framework
    fn update(&mut self, _event: &winit::event::WindowEvent) {}

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

        self.particle_work_group_count = ((self.params.current().particle.num_particles as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        // update uniforms
        // TODO: only update when value is changed
        queue.write_buffer(&self.buffers.particle_uniform, 0, self.params.current().particle.to_bytes());
        queue.write_buffer(&self.buffers.decay_uniform, 0, self.params.current().decay.to_bytes());
        queue.write_buffer(&self.buffers.diffuse_uniform, 0, self.params.current().diffuse.to_bytes());

        let r = &self.params.current().render;
        let vec: Vec<f32> = vec![
            r.color_1[0], r.color_1[1], r.color_1[2], r.color_2[0], r.color_2[1], r.color_2[2], r.color_pow, r.cutoff
        ];
        queue.write_buffer(&self.buffers.render_uniform, 0, bytemuck::cast_slice(vec.as_slice()));


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

        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("compute particle movement");
        {
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipelines.particle_compute_pipeline);
            cpass.set_bind_group(0, &self.bind_groups.particle_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch(self.particle_work_group_count, 1, 1);
        }
        command_encoder.pop_debug_group();

        if self.params.global.post_enabled {
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
        }

        command_encoder.push_debug_group("render to screen");
        {
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.pipelines.render_pipeline);
            rpass.set_vertex_buffer(0, self.buffers.vertices_buffer.slice(..));
            rpass.set_bind_group(0, &self.bind_groups.render_bind_groups[self.frame_num % 2], &[]);
            rpass.draw(0..6, 0..1);
        }
        command_encoder.pop_debug_group();

        self.frame_num += 1;

        queue.submit(Some(command_encoder.finish()));
    }

    fn ui(
        &mut self,
        ui: &imgui::Ui
    ) {
        
        use imgui::Condition;
        use imgui::im_str;

        //ui.show_default_style_editor();
        //ui.show_demo_window(&mut true);

        let window = imgui::Window::new(im_str!("Configs"));
        window
            .size([300.0, 600.0], Condition::FirstUseEver)
            .bg_alpha(1.0)
            .menu_bar(true)
            .build(&ui, || {

                if let Some(token) = ui.begin_menu_bar() {
                    if imgui::MenuItem::new(im_str!("New")).build(ui) {
                        self.params.new();
                    }
                    if imgui::MenuItem::new(im_str!("Save")).build(ui) {
                        self.params.save("./resources/params.json");
                    }
                    token.end(ui);
                }

                let status = if self.params.global.post_enabled {
                    im_str!("Post-processing enabled")
                } else {
                    im_str!("Post-processing disabled")
                };
                
                imgui::ComboBox::new(im_str!("Preset"))
                    .flags(imgui::ComboBoxFlags::empty())
                    //.preview_value(&imgui::ImString::new(self.params.current_name()))
                    .build_simple(ui, &mut self.params.current, &self.params.params[..], &|p: &crate::params::Params| {
                        std::borrow::Cow::from(imgui::ImString::new(&p.name))
                });

                let mut str = imgui::ImString::new(&self.params.current().name);
                imgui::InputText::new(ui, im_str!("Name"), &mut str)
                    .no_horizontal_scroll(true)
                    .build();
                self.params.current_mut().name = str.to_string();

                
                if ui.radio_button_bool(status, true) {
                    self.params.global.post_enabled = !self.params.global.post_enabled
                }

                unsafe {
                    ui.text(im_str!("Particle Compute"));
                    imgui::Slider::new(im_str!("Num Particles"))
                        .range(0u32..=self.params.global.max_particles-1)
                        .build(ui, &mut self.params.current_mut().particle.num_particles);
                    imgui::Slider::new(im_str!("Trail Power"))
                        .range(0.0..=64.0)
                        .build(ui, &mut self.params.current_mut().particle.trail_power);
                    imgui::Slider::new(im_str!("Speed"))
                        .range(0.0..=15.0)
                        .build(ui, &mut self.params.current_mut().particle.speed);
                    imgui::Slider::new(im_str!("Sensor Angle"))
                        .range(0.0..=1.5)
                        .build(ui, &mut self.params.current_mut().particle.sensor_angle);
                    imgui::Slider::new(im_str!("Sensor Distance"))
                        .range(0.0..=0.01)
                        .build(ui, &mut self.params.current_mut().particle.sensor_distance);
                    imgui::Slider::new(im_str!("Turn Speed"))
                        .range(0.0..=3.14)
                        .build(ui, &mut self.params.current_mut().particle.turn_speed);
                    ui.separator();
                    ui.text(im_str!("Decay Compute"));
                    imgui::Slider::new(im_str!("Decay Factor"))
                        .range(0.5..=1.0)
                        .build(ui, &mut self.params.current_mut().decay.decay_rate);
                    ui.separator();
                    ui.text(im_str!("Diffuse Compute"));
                    imgui::Slider::new(im_str!("Diffuse Amount"))
                        .range(0.0..=1.0)
                        .build(ui, &mut self.params.current_mut().diffuse.diffuse_amount);
                    ui.separator();
                    ui.text(im_str!("Render"));
                    imgui::Slider::new(im_str!("Color Power"))
                        .range(0.2..=1.0)
                        .build(ui, &mut self.params.current_mut().render.color_pow);
                    imgui::ColorPicker::new(im_str!("Color 1"), &mut self.params.current_mut().render.color_1)
                        .input_mode(imgui::ColorEditInputMode::Rgb)
                        .mode(imgui::ColorPickerMode::HueWheel)
                        .build(ui);
                    imgui::ColorPicker::new(im_str!("Color 2"), &mut self.params.current_mut().render.color_2)
                        .input_mode(imgui::ColorEditInputMode::Rgb)
                        .mode(imgui::ColorPickerMode::HueWheel)
                        .build(ui);

                }
                
        });
    }
}

fn main() {
    framework::run::<MoldSim>("Mold sim");
}
