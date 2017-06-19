# unity-cuda-interop
Demo of Unity3D CUDA texture interop issue

This project demonstrates a problem of accessing Unity3D `RenderTarget` from CUDA.

In particular, for D3D9, D3D11 cores `cudaGraphicsD3D??RegisterResource()` returns 11 (invalid parameter).
For OpenGL core `cudaGraphicsGLRegisterImage()` returns 63 (cudaErrorOperatingSystem, This error indicates that an OS call failed).

Replacing `RenderTarget` with normal `Texture` allows CUDA calls to succeed.
