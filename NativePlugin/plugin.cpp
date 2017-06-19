#include "Unity/IUnityGraphics.h"
#include "Unity/IUnityInterface.h"

#ifdef _WIN32
    #include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_d3d9_interop.h>
#include <cuda_d3d11_interop.h>

static IUnityGraphics *unityGraphics = nullptr;

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces *unityInterfaces) {
    unityGraphics = unityInterfaces->Get<IUnityGraphics>();
}

extern "C" UNITY_INTERFACE_EXPORT cudaError_t UNITY_INTERFACE_API RegisterTexture(void *texture) {
    cudaGraphicsResource_t resource;
    switch (unityGraphics->GetRenderer()) {
    case kUnityGfxRendererOpenGL:
        return cudaGraphicsGLRegisterImage(&resource, (GLuint)texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
    case kUnityGfxRendererD3D11:
        return cudaGraphicsD3D11RegisterResource(&resource, (ID3D11Resource*)texture, cudaGraphicsRegisterFlagsNone);
    case kUnityGfxRendererD3D9:
        return cudaGraphicsD3D9RegisterResource(&resource, (IDirect3DResource9*)texture, cudaGraphicsRegisterFlagsNone);
    default:
        return cudaErrorNotYetImplemented;
    }
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetErrorString(cudaError_t error, char *str) {
    strcpy(str, cudaGetErrorString(error));
}
