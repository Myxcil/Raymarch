using UnityEngine;
using UnityEngine.Rendering;
using System;

//--------------------------------------------------------------------------------------------------------------------------------------------------
namespace VolumeRendering
{
    //----------------------------------------------------------------------------------------------------------------------------------------------
    [Serializable]
    public struct Resolution
    {
        public int maxRes;
        public float gridSize;
        public int raycastResolution;
        public int numBlurPasses;

        public static readonly Resolution DEFAULT = new Resolution()
        {
            maxRes = 128,
            gridSize = 0.1f,
            raycastResolution = 2,
            numBlurPasses = 1,
        };
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------
    [Serializable]
    public class Setup
    {
        public Shader shaderVolume;
        public Shader shaderRaymarch;
        public Shader blurShader;
        public float edgeDetectionThreshold = 0.02f;
        public CameraEvent cameraEvent = CameraEvent.AfterForwardOpaque;

        //----------------------------------------------------------------------------------------------------------------------------------------------
        public static Vector3 CalculateGridSize(Vector3 size, float gridScale, int maxRes)
        {
            GetResolution(size, gridScale, maxRes, out int width, out int height, out int depth);
            return new Vector3(size.x / width, size.y / height, size.z / depth);
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------
        public static void GetResolution(Vector3 size, float gridSize, int maxRes, out int width, out int height, out int depth)
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