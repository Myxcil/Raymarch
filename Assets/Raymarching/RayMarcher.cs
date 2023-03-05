using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;

//------------------------------------------------------------------------------------------------------------------------------------------------------
namespace VolumeRendering
{
    //--------------------------------------------------------------------------------------------------------------------------------------------------
    class RayMarcher
    {
        //------------------------------------------------------------------------------------------------------------------------------------------
        private RenderTexture rayData;
        private RenderTargetIdentifier idRayData;
        private RenderTexture rayDataSmall;
        private RenderTargetIdentifier idRayDataSmall;
        private RenderTexture rayEdges;
        private RenderTargetIdentifier idRayEdges;
        private RenderTexture rayMarchLayer;
        private RenderTargetIdentifier idRayMarchLayer;

        private readonly Material volumeMaterial;
        private const int VOLUME_PASS_BACK_FACES = 0;
        private const int VOLUME_PASS_FRONT_FACES = 1;
        private const int VOLUME_PASS_EDGE_FILTER = 2;

        private readonly Material rayMarchMaterial;
        private const int RAYMARCH_PASS_RAYCAST_VOLUME = 0;
        private const int RAYMARCH_PASS_COMBINE_SCENE_EDGEFILTER = 1;
        private const int RAYMARCH_PASS_COMBINE_SCENE_SIMPLE = 2;

        private const int BLUR_PASS_HORIZONTAL = 1;
        private const int BLUR_PASS_VERTICAL = 2;

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private RenderTexture rayMarchRenderTarget;
        private RenderTargetIdentifier idRayMarchRenderTarget;
        private readonly Vector3 gridScale;
        private readonly Vector4 gridDim;
        private readonly Vector4 rcpGridDim;
        private Texture2D jitterTex;

        private readonly Material matBlurShader;
        private int currBlurPasses;

        //-----------------------------------------------------------------------------------------------------------------------------------------     
        private CameraEvent cameraEvent;
        private readonly Dictionary<Camera, CommandBuffer> cameraCommands = new Dictionary<Camera, CommandBuffer>();

        //-----------------------------------------------------------------------------------------------------------------------------------------     
        public Texture Result { get { return rayMarchLayer; } }

        //-----------------------------------------------------------------------------------------------------------------------------------------     
        public RayMarcher(RenderTexture volumeTexture, Vector3 volumeSize, int raycastResolution, Shader volume, Shader raymarch, Shader blur)
        {
            CreateRenderTargets(raycastResolution);

            CreateJitterTexture(256);

            Vector3 v = new Vector3(volumeTexture.width, volumeTexture.height, volumeTexture.volumeDepth);
            float maxDim = Mathf.Max(Mathf.Max(v.x, v.y), v.z);

            gridDim = new Vector4(v.x, v.y, v.z, maxDim);
            gridScale = new Vector3(gridDim.x / volumeSize.x, gridDim.y / volumeSize.y, gridDim.z / volumeSize.z);
            rcpGridDim = new Vector4(0.5f / gridDim.x, 0.5f / gridDim.y, 0.5f / gridDim.z, 0.5f / gridDim.w);

            volumeMaterial = new Material(volume);
            if (rayDataSmall != null)
            {
                Vector4 rayTexelSize = Vector4.zero;
                rayTexelSize.x = 1.0f / rayDataSmall.width;
                rayTexelSize.y = 1.0f / rayDataSmall.height;
                volumeMaterial.SetVector("TexelSize", rayTexelSize);
                volumeMaterial.mainTexture = rayDataSmall;
            }
            else
            {
                volumeMaterial.mainTexture = rayData;
            }

            rayMarchMaterial = new Material(raymarch);
            rayMarchMaterial.SetTexture("_Volume", volumeTexture);
            rayMarchMaterial.SetTexture("_JitterTex", jitterTex);
            rayMarchMaterial.SetVector("_GridDim", gridDim);
            rayMarchMaterial.SetVector("_rcpGridDim", rcpGridDim);

            matBlurShader = new Material(blur);
            matBlurShader.hideFlags = HideFlags.HideAndDontSave;
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private void CreateRenderTargets(int raycastResolution)
        {
            int targetWidth = Screen.width;
            int targetHeight = Screen.height;

            if (raycastResolution > 0)
            {
                float rcpRes = 1.0f / raycastResolution;
                targetWidth = Mathf.RoundToInt(rcpRes * targetWidth);
                targetHeight = Mathf.RoundToInt(rcpRes * targetHeight);
            }

            rayData = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBHalf)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            idRayData = new RenderTargetIdentifier(rayData);

            if (raycastResolution > 1)
            {
                rayDataSmall = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.ARGBHalf)
                {
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                idRayDataSmall = new RenderTargetIdentifier(rayDataSmall);

                rayEdges = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.RFloat)
                {
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                idRayEdges = new RenderTargetIdentifier(rayEdges);

            }

            rayMarchRenderTarget = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.ARGBHalf)
            {
                filterMode = FilterMode.Trilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            idRayMarchRenderTarget = new RenderTargetIdentifier(rayMarchRenderTarget);

            rayMarchLayer = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.Default)
            {
                filterMode = FilterMode.Trilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            idRayMarchLayer = new RenderTargetIdentifier(rayMarchLayer);
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private void CreateJitterTexture(int resolution)
        {
            jitterTex = new Texture2D(resolution, resolution, TextureFormat.ARGB32, false)
            {
                name = "JitterTex",
                hideFlags = HideFlags.HideAndDontSave
            };

            Color32[] pixels = jitterTex.GetPixels32();
            for (int i = 0; i < pixels.Length; ++i)
            {
                pixels[i] = new Color(Random.value, Random.value, Random.value, Random.value);
            }

            jitterTex.SetPixels32(pixels);
            jitterTex.Apply();
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        public void Destroy()
        {
            foreach(var kv in cameraCommands)
            {
                Camera camera = kv.Key;
                if (camera == null)
                    continue;

                if (kv.Value != null)
                {
                    camera.RemoveCommandBuffer(cameraEvent, kv.Value);
                }
            }
            cameraCommands.Clear();

            FreeRenderTarget(ref rayMarchRenderTarget);
            FreeRenderTarget(ref rayEdges);
            FreeRenderTarget(ref rayDataSmall);
            FreeRenderTarget(ref rayData);
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private static void FreeRenderTarget(ref RenderTexture rt)
        {
            if (rt != null)
            {
                rt.Release();
                rt = null;
            }
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        public void OnRenderObject(Camera camera, MeshRenderer renderer, float edgeDetectionThreshold, int numBlurPasses)
        {
            if (!cameraCommands.TryGetValue(camera, out CommandBuffer commandBuffer) || currBlurPasses != numBlurPasses)
            {
                currBlurPasses = numBlurPasses;
                if (commandBuffer == null)
                {
                    commandBuffer = new CommandBuffer() { name = "RenderVolume" };
                    camera.AddCommandBuffer(cameraEvent, commandBuffer);
                    cameraCommands.Add(camera, commandBuffer);
                }
                else
                {
                    commandBuffer.Clear();
                }
                FillCommandBuffer(commandBuffer, renderer);
            }

            volumeMaterial.SetFloat("EdgeThreshold", edgeDetectionThreshold);

            UpdateTracingParameters(camera, renderer.localToWorldMatrix);
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private void FillCommandBuffer(CommandBuffer cmdVolume, MeshRenderer renderer)
        {
            // Step 1: 
            // Render first the backfaces and then the front faces of the mesh
            // This will create a rendertarget which stores the ray origins (RGB) and 
            // tracing depth (W) for the scene
            cmdVolume.SetRenderTarget(idRayData);
            cmdVolume.ClearRenderTarget(false, true, Color.clear);
            cmdVolume.DrawRenderer(renderer, volumeMaterial, 0, VOLUME_PASS_BACK_FACES);
            cmdVolume.DrawRenderer(renderer, volumeMaterial, 0, VOLUME_PASS_FRONT_FACES);

            // Step 2:
            // If we do the raymarching at a smaller resolution than the current one,
            // run and edge filter over it to determine parts of the images which would 
            // create artifacts because of the lower resolution of the final raymarched image
            if (rayDataSmall != null)
            {
                cmdVolume.Blit(idRayData, idRayDataSmall);
                cmdVolume.Blit(idRayDataSmall, idRayEdges, volumeMaterial, VOLUME_PASS_EDGE_FILTER);
            }
            
            // Step 3:
            // Do the actual raymarching and create an intermediate output
            if (rayDataSmall != null)
            {
                cmdVolume.Blit(idRayDataSmall, idRayMarchRenderTarget, rayMarchMaterial, RAYMARCH_PASS_RAYCAST_VOLUME);
            }
            else
            {
                cmdVolume.Blit(idRayData, idRayMarchRenderTarget, rayMarchMaterial, RAYMARCH_PASS_RAYCAST_VOLUME);
            }

            // Step 4:
            // Combine the raymarching result with the edge cases which will be raycasted at a higher resolution
            if (rayEdges != null)
            {
                cmdVolume.SetGlobalTexture("_EdgeLookup", idRayEdges);
                cmdVolume.SetGlobalTexture("_RayTexture", idRayData);
                cmdVolume.Blit(idRayMarchRenderTarget, idRayMarchLayer, rayMarchMaterial, RAYMARCH_PASS_COMBINE_SCENE_EDGEFILTER);
            }
            else
            {
                cmdVolume.Blit(idRayMarchRenderTarget, idRayMarchLayer, rayMarchMaterial, RAYMARCH_PASS_COMBINE_SCENE_SIMPLE);
            }

            // Step 5:
            // Blur the resulting texture, which will then be rendered into the scene afterwards
            if (currBlurPasses > 0)
            {
                int idTempBlur = Shader.PropertyToID("_BlurTemp");
                cmdVolume.GetTemporaryRT(idTempBlur, -1, -1, 0, FilterMode.Trilinear, RenderTextureFormat.Default);

                for (int i = 0; i < currBlurPasses; ++i)
                {
                    cmdVolume.Blit(idRayMarchLayer, idTempBlur, matBlurShader, BLUR_PASS_HORIZONTAL);
                    cmdVolume.Blit(idTempBlur, idRayMarchLayer, matBlurShader, BLUR_PASS_VERTICAL);
                }

                cmdVolume.ReleaseTemporaryRT(idTempBlur);
            }
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private void UpdateTracingParameters(Camera camera, Matrix4x4 mWorld)
        {
            Matrix4x4 V = camera.worldToCameraMatrix;

            // Camera position in Volume-Space
            Matrix4x4 MV = V * mWorld;
            Matrix4x4 invMV = Matrix4x4.Inverse(MV);
            Vector3 camPos = invMV.MultiplyPoint3x4(Vector3.zero) + 0.5f * Vector3.one;
            rayMarchMaterial.SetVector("_CameraTS", camPos);

            // Inverse World-View-Projection matrix
            Matrix4x4 P = GL.GetGPUProjectionMatrix(camera.projectionMatrix, true);
            Matrix4x4 invMVP = Matrix4x4.Inverse(P * MV);
            rayMarchMaterial.SetMatrix("_invMVP", invMVP);

            // Scale factor to determine the amount of samples along a ray
            Matrix4x4 mGrid = Matrix4x4.Scale(gridScale) * V;
            Vector3 gridAxis = mGrid.GetColumn(2);
            float sampleScale = 2.0f * gridAxis.magnitude;
            rayMarchMaterial.SetFloat("_SampleScale", sampleScale);

            // calculate near plane position for raycasting when inside volume
            Vector3 camNearPos = CaluclateNearPlanePosition(camera);
            rayMarchMaterial.SetVector("_NearPlanePos", camNearPos);
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private Vector3 CaluclateNearPlanePosition(Camera camera)
        {
            float yFac = Mathf.Tan(camera.fieldOfView * Mathf.Deg2Rad * 0.5f);
            float xFac = camera.aspect * yFac;

            float near = camera.nearClipPlane;

            Vector3 nearPos = Vector3.forward * near;
            nearPos += Vector3.right * near * xFac;
            nearPos += Vector3.up * near * yFac;

            return nearPos;
        }

        //-----------------------------------------------------------------------------------------------------------------------------------------
        public void Apply(float absorption, float density, float jitterStrength)
        {
            rayMarchMaterial.SetFloat("_Absorption", absorption);
            rayMarchMaterial.SetFloat("_Density", density);
            rayMarchMaterial.SetFloat("_JitterStrength", jitterStrength);
        }
    }
}