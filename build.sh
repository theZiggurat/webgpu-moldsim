#!/bin/bash

cd shaders
for f in *; do echo -n "compiling: $f...\n" && ../convert.exe $f $f.spv && mv $f.spv ../resources/spirv/${f%.wgsl}.spv; done