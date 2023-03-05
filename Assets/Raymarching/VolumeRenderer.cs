using UnityEngine;
using UnityEngine.Rendering;

//------------------------------------------------------------------------------------------------------------------------------------------------------
namespace VolumeRendering
{
    //--------------------------------------------------------------------------------------------------------------------------------------------------
    public class VolumeRenderer : MonoBehaviour
    {
        //----------------------------------------------------------------------------------------------------------------------------------------------
        [Header("Volume Init")]
        [SerializeField]
        private ComputeShader computeShader = null;
        [SerializeField]
        private int maxResolution = 128;
        [SerializeField]
        private float gridSize = 0.1f;
        //----------------------------------------------------------------------------------------------------------------------------------------------
        [Header("Raymarching")]
        [SerializeField]
        private Shader volume = null;
        [SerializeField]
        private Shader raymarch = null;
        [SerializeField]
        private int raycastResolution = 2;
        [SerializeField]
        private Shader blur = null;
        //----------------------------------------------------------------------------------------------------------------------------------------------
        [Header("Runtime Parameter")]
        [SerializeField]
        private float absorption = 0.5f;
        [SerializeField]
        private float density = 0.75f;
        [SerializeField]
        private float edgeDetectionThreshold = 0.01f;
        [SerializeField]
        private int numBlurPasses = 0;
        [SerializeField]
        private float jitterStrength = 1.0f;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        [Header("Debug")]
        public bool showGrid = false;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private RayMarcher rayMarcher;
        private MeshRenderer meshRenderer;
        private Material volumeMaterial;

        private RenderTexture volumeTexture;

        private int kernelUpdate;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void Awake()
        {
            meshRenderer = GetComponent<MeshRenderer>();
            
            // setup material which will render the final image of the
            // raymarch to the scene
            volumeMaterial = meshRenderer.material;
            volumeMaterial.DisableKeyword("_ALPHATEST_ON");
            volumeMaterial.EnableKeyword("_ALPHABLEND_ON");
            volumeMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            volumeMaterial.renderQueue = 3000;
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private void Start()
        {
            // calculate resolution of grid depending on size of the current gameobject
            Vector3 scale = transform.localScale;
            GetResolution(scale, gridSize, maxResolution, out int width, out int height, out int depth);

            // create 3d texture which is used as a source for raymarching
            volumeTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf);
            volumeTexture.dimension = TextureDimension.Tex3D;
            volumeTexture.volumeDepth = depth;
            volumeTexture.enableRandomWrite = true;
            volumeTexture.wrapMode = TextureWrapMode.Clamp;
            volumeTexture.filterMode = FilterMode.Point;
            volumeTexture.Create();

            // initialize the raymarching
            rayMarcher = new RayMarcher(volumeTexture, scale, raycastResolution, volume, raymarch, blur);

            // assign raymarching output texture
            volumeMaterial.mainTexture = rayMarcher.Result;

            // intialize and run a compute shader to create the volume texture
            threadSetup.CalculateThreadCount(width, height, depth);

            int kInit = computeShader.FindKernel("Init");
            computeShader.SetInts("resolution", width, height, depth);
            computeShader.SetFloats("rcpResolution", 1.0f / width, 1.0f / height, 1.0f / depth);
            computeShader.SetTexture(kInit, "outputFloat4", volumeTexture);
            Dispatch(kInit);

            kernelUpdate = computeShader.FindKernel("Update");
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void OnDestroy()
        {
            if (rayMarcher != null)
            {
                rayMarcher.Destroy();
                rayMarcher = null;
            }
            if (volumeTexture != null)
            {
                volumeTexture.Release();
                volumeTexture = null;
            }
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void Update()
        {
            if (kernelUpdate != 0)
            {
                computeShader.SetFloats("time", Time.deltaTime, Time.time);
                computeShader.SetTexture(kernelUpdate, "outputFloat4", volumeTexture);
                Dispatch(kernelUpdate);
            }

            if (rayMarcher != null)
            {
                rayMarcher.Apply(absorption, density, jitterStrength);
            }
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void OnWillRenderObject()
        {
            if (rayMarcher != null)
            {
                rayMarcher.OnRenderObject(Camera.current, meshRenderer, edgeDetectionThreshold, numBlurPasses);
            }
        }

#if UNITY_EDITOR
        //----------------------------------------------------------------------------------------------------------------------------------------------
        void OnGUI()
        {
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void OnDrawGizmosSelected()
        {
            Matrix4x4 mOld = Gizmos.matrix;
            Gizmos.matrix = transform.localToWorldMatrix;

            Gizmos.color = Color.cyan;
            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);

            if (showGrid)
            {
                DrawCubeGrid(transform.localScale, gridSize, maxResolution);
            }

            Gizmos.matrix = mOld;
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        public static void DrawCubeGrid(Vector3 scale, float gridScale, int maxRes)
        {
            Gizmos.color = Color.cyan;

            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);

            int[] dim = new int[3];
            GetResolution(scale, gridScale, maxRes, out dim[0], out dim[1], out dim[2]);

            Vector3 size = new(1.0f / dim[0], 1.0f / dim[1], 1.0f / dim[2]);

            DrawCubeGrid(size, 0, dim);
            DrawCubeGrid(size, 1, dim);
            DrawCubeGrid(size, 2, dim);
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private static void DrawCubeGrid(Vector3 size, int i, int[] dim)
        {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;

            Vector3 p1 = Vector3.zero;
            Vector3 p2 = Vector3.zero;

            p1[k] = -0.5f;
            p2[k] = 0.5f;
            for (int l = 0; l < dim[i]; ++l)
            {
                p1[i] = -0.5f + size[i] * l;
                p2[i] = p1[i];

                p1[j] = -0.5f;
                p2[j] = p1[j];
                Gizmos.DrawLine(p1, p2);

                p1[j] = 0.5f;
                p2[j] = p1[j];
                Gizmos.DrawLine(p1, p2);
            }

            p1[i] = -0.5f;
            p2[i] = 0.5f;
            for (int l = 0; l < dim[k]; ++l)
            {
                p1[k] = -0.5f + size[k] * l;
                p2[k] = p1[k];

                p1[j] = -0.5f;
                p2[j] = p1[j];
                Gizmos.DrawLine(p1, p2);

                p1[j] = 0.5f;
                p2[j] = p1[j];
                Gizmos.DrawLine(p1, p2);
            }
        }
#endif
        //----------------------------------------------------------------------------------------------------------------------------------------------
        //
        //----------------------------------------------------------------------------------------------------------------------------------------------
        private const int NUM_THREADS_X = 8;
        private const int NUM_THREADS_Y = 8;
        private const int NUM_THREADS_Z = 8;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private struct ThreadSetup
        {
            public int numX, numY, numZ;
            public void CalculateThreadCount(int xRes, int yRes, int zRes)
            {
                numX = CalculateNumThreads(xRes, NUM_THREADS_X);
                numY = CalculateNumThreads(yRes, NUM_THREADS_Y);
                numZ = CalculateNumThreads(zRes, NUM_THREADS_Z);
            }
        }
        private ThreadSetup threadSetup;

        //-----------------------------------------------------------------------------------------------------------------------------------------
        private static int CalculateNumThreads(int resolution, int threadcount)
        {
            int numThreads = (resolution + threadcount - 1) / threadcount;
            return Mathf.Max(1, numThreads);
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private void Dispatch(int kernel)
        {
            computeShader.Dispatch(kernel, threadSetup.numX, threadSetup.numY, threadSetup.numZ);
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private static Vector3 CalculateGridSize(Vector3 size, float gridScale, int maxRes)
        {
            GetResolution(size, gridScale, maxRes, out int width, out int height, out int depth);
            return new Vector3(size.x / width, size.y / height, size.z / depth);
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------
        private static void GetResolution(Vector3 size, float gridSize, int maxRes, out int width, out int height, out int depth)
        {
            size /= gridSize;

            if (size.x >= size.y && size.x >= size.z)
            {
                width = SetDimension(size.x, maxRes);
                height = SetDimension(size.y * width / size.x);
                depth = SetDimension(size.z * width / size.x);
            }
            else if (size.y >= size.x && size.y >= size.z)
            {
                height = SetDimension(size.y, maxRes);
                width = SetDimension(size.x * height / size.y);
                depth = SetDimension(size.z * height / size.y);
            }
            else
            {
                depth = SetDimension(size.z, maxRes);
                width = SetDimension(size.x * depth / size.z);
                height = SetDimension(size.y * depth / size.z);
            }
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------
        private static readonly int[] SnapValues = { 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 128 };

        //---------------------------------------------------------------------------------------------------------------------------------------------
        private static int SetDimension(float fValue, int maxRes = 0)
        {
            int iValue = Mathf.RoundToInt(fValue);
            if (iValue <= SnapValues[0])
            {
                iValue = SnapValues[0];
            }
            else if (iValue >= SnapValues[SnapValues.Length - 1])
            {
                iValue = SnapValues[SnapValues.Length - 1];
            }
            else
            {
                for (int i = 1; i < SnapValues.Length; ++i)
                {
                    int dp = iValue - SnapValues[i - 1];
                    int dn = SnapValues[i] - iValue;
                    if (dp >= 0 && dn >= 0)
                    {
                        iValue = dp < dn ? SnapValues[i - 1] : SnapValues[i];
                        break;
                    }
                }
            }
            if (maxRes > 0)
            {
                iValue = Mathf.Min(maxRes, iValue);
            }
            return iValue;
        }
    }
}
