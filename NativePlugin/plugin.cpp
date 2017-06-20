#include "Unity/IUnityGraphics.h"
#include "Unity/IUnityInterface.h"

#ifdef _WIN32
    #include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_d3d9_interop.h>
#include <cuda_d3d11_interop.h>

#include <sstream>

using namespace std;

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

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetTextureInfo(void *texture, char *info) {
    stringstream out;

    if (unityGraphics->GetRenderer() != kUnityGfxRendererD3D11) {
        out << "Not D3D11" << endl;
    }
    else if (!texture) {
        out << "Texture is null" << endl;
    }
    else {
        ID3D11Texture2D *d3dTex = (ID3D11Texture2D *)texture;
        D3D11_RESOURCE_DIMENSION dims;
        d3dTex->GetType(&dims);
        if (dims != D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
            out << "Not a 2D texture: " << dims << endl;
        }
        else {
            D3D11_TEXTURE2D_DESC desc;
            d3dTex->GetDesc(&desc);

            out << "Width: " << desc.Width << endl;
            out << "Height: " << desc.Height << endl;
            out << "MipLevels: " << desc.MipLevels << endl;
            out << "ArraySize: " << desc.ArraySize << endl;
            out << "Format: " << desc.Format << endl;
            out << "Sample.Count: " << desc.SampleDesc.Count << endl;
            out << "Sample.Quality: " << desc.SampleDesc.Quality << endl;
            out << "Usage: " << desc.Usage << endl;
            out << "BindFlags: " << desc.BindFlags << endl;
            out << "CPUAccessFlags: " << desc.CPUAccessFlags << endl;
            out << "MiscFlags: " << desc.MiscFlags << endl;
        }
    }

    strcpy(info, out.str().c_str());
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetErrorString(cudaError_t error, char *str) {
    strcpy(str, cudaGetErrorString(error));
}
