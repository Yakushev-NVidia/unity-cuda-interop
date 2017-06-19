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

    void Start () {
        int error = RegisterTexture(texture.GetNativeTexturePtr());

        StringBuilder desc = new StringBuilder(1000);
        GetErrorString(error, desc);

        Debug.Log("CUDA Error: " + error + " (" + desc.ToString() + ")");
	}
}
