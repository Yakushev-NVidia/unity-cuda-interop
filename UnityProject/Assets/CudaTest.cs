using System;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

public class CudaTest : MonoBehaviour {
    public Texture texture;

    [DllImport("NativePlugin", EntryPoint = "RegisterTexture")]
    private static extern int RegisterTexture(IntPtr texture);

    [DllImport("NativePlugin", EntryPoint = "GetErrorString")]
    private static extern void GetErrorString(int error, StringBuilder str);

    [DllImport("NativePlugin", EntryPoint = "GetTextureInfo")]
    private static extern void GetTextureInfo(IntPtr texture, StringBuilder str);

    void Start () {
        int error = RegisterTexture(texture.GetNativeTexturePtr());

        StringBuilder desc = new StringBuilder(1000);
        GetErrorString(error, desc);
        Debug.Log("CUDA Error: " + error + " (" + desc.ToString() + ")");

        StringBuilder info = new StringBuilder(10000);
        GetTextureInfo(texture.GetNativeTexturePtr(), info);
        Debug.Log("Texture info: " + info.ToString());
    }
}
