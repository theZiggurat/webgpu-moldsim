pub trait Uniform: Sized {
    fn memsize(&self) -> Option<wgpu::BufferSize> {
        log::info!("size: {}", std::mem::size_of::<Self>() as u64);
        std::num::NonZeroU64::new(std::mem::size_of::<Self>() as u64)
    }

    fn to_bytes(&self) -> &[u8] {
        let p: *const Self = self;     
        let p: *const u8 = p as *const u8;  
        unsafe { 
            std::slice::from_raw_parts(p, std::mem::size_of::<Self>())
        }
    }
}