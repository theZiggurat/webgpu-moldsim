pub fn create_shader(device: &wgpu::Device, path: &str) -> wgpu::ShaderModule {
    use std::io::prelude::*;
    use std::fs::File;

    let mut file = File::open(path).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let label = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str());

    device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: label,
        source: wgpu::util::make_spirv(&buf[..]),
        flags: wgpu::ShaderFlags::VALIDATION
    })
}


