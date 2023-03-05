using UnityEngine;
using UnityEngine.Rendering;

//------------------------------------------------------------------------------------------------------------------------------------------------------
namespace VolumeRendering
{
    //--------------------------------------------------------------------------------------------------------------------------------------------------
    public class VolumeRenderer : MonoBehaviour
    {
        //----------------------------------------------------------------------------------------------------------------------------------------------
        [SerializeField]
        private Setup setup = new Setup();
        [SerializeField]
        private Resolution resolution = Resolution.DEFAULT;
        [SerializeField]
        private FilterMode volumeTextureFilter = FilterMode.Bilinear;
        [SerializeField]
        private float absorption = 0.5f;
        [SerializeField]
        private float density = 0.75f;
        [SerializeField]
        private ComputeShader computeShader = null;
        [Header("Debug")]
        public bool showGrid = false;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private RayMarcher rayMarcher;
        private MeshRenderer meshRenderer;
        private Material volumeMaterial;

        private RenderTexture volumeTexture;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void Awake()
        {
            meshRenderer = GetComponent<MeshRenderer>();
            
            volumeMaterial = meshRenderer.material;
            volumeMaterial.DisableKeyword("_ALPHATEST_ON");
            volumeMaterial.EnableKeyword("_ALPHABLEND_ON");
            volumeMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            volumeMaterial.renderQueue = 3000;
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        private void Start()
        {
            Vector3 scale = transform.localScale;

            Setup.GetResolution(scale, resolution.gridSize, resolution.maxRes, out int width, out int height, out int depth);

            volumeTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf);
            volumeTexture.dimension = TextureDimension.Tex3D;
            volumeTexture.volumeDepth = depth;
            volumeTexture.enableRandomWrite = true;
            volumeTexture.wrapMode = TextureWrapMode.Clamp;
            volumeTexture.filterMode = volumeTextureFilter;
            volumeTexture.Create();

            rayMarcher = new RayMarcher(setup, resolution, volumeTexture, scale);

            volumeMaterial.mainTexture = rayMarcher.Result;

            threadSetup.CalculateThreadCount(width, height, depth);

            int kCSMain = computeShader.FindKernel("CSMain");
            computeShader.SetInts("resolution", width, height, depth);
            computeShader.SetFloats("rcpResolution", 1.0f / width, 1.0f / height, 1.0f / depth);
            computeShader.SetTexture(kCSMain, "outputFloat4", volumeTexture);
            Dispatch(kCSMain);
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
            if (rayMarcher != null)
            {
                rayMarcher.Apply(absorption, density);
            }
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        void OnWillRenderObject()
        {
            if (rayMarcher != null)
            {
                rayMarcher.OnRenderObject(Camera.current, meshRenderer);
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
                DrawCubeGrid(transform.localScale, resolution.gridSize, resolution.maxRes);
            }

            Gizmos.matrix = mOld;
        }

        //----------------------------------------------------------------------------------------------------------------------------------------------
        public static void DrawCubeGrid(Vector3 scale, float gridScale, int maxRes)
        {
            Gizmos.color = Color.cyan;

            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);

            int[] dim = new int[3];
            Setup.GetResolution(scale, gridScale, maxRes, out dim[0], out dim[1], out dim[2]);

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
    }
}
