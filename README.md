# wgpu-toy

Little project to learn web-gpu.

## how to run

* `sh build.sh` to compile WGSL shaders into SPIR-V
* `cargo run` 

## notes
* How shaders are loaded into the program will depend on what compilation mode rustc is in:
	* compiling in optimized mode (`cargo run --release`) will statically load shaders through `include_bytes!`macro, resulting in faster program initialization.
	* compiling in debug mode (`cargo run`) will dynamically load compiled SPIR-V from file IO, resulting to no recompile.