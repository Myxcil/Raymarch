// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

//---------------------------------------------------------------------------------------------------------------------------------------------
Shader "Hidden/Raymarching/Volume" 
{
	Properties{
		_MainTex("", 2D) = "" {}
	}
	
	//-----------------------------------------------------------------------------------------------------------------------------------------
	CGINCLUDE
	#include "UnityCG.cginc"

	//-----------------------------------------------------------------------------------------------------------------------------------------
	uniform sampler2D _CameraDepthTexture;

	//-----------------------------------------------------------------------------------------------------------------------------------------
	struct appdata
	{
		float3	pos		: POSITION;
		float2	uv		: TEXCOORD0;
	};

	//-----------------------------------------------------------------------------------------------------------------------------------------
	struct v2f
	{
		float4	pos			: SV_POSITION;
		float4	uv			: TEXCOORD0;
		float3	gridPos		: TEXCOORD1;
		float	depth		: TEXCOORD2;
	};

	//-----------------------------------------------------------------------------------------------------------------------------------------
	struct v2fQuad
	{
		float4	pos			: SV_POSITION;
		float2	uv			: TEXCOORD0;
	};

	//-----------------------------------------------------------------------------------------------------------------------------------------
	v2f vertVolume(appdata IN)
	{
		float4 pos = float4(IN.pos, 1);

		v2f OUT;
		OUT.pos = UnityObjectToClipPos(pos);
		OUT.uv = ComputeScreenPos(OUT.pos);
		OUT.gridPos = IN.pos + 0.5;
		OUT.depth = OUT.pos.w;

		return OUT;
	}

	//-----------------------------------------------------------------------------------------------------------------------------------------
	float getVolumeDepth(v2f IN)
	{
		return IN.depth;
	}

	//-----------------------------------------------------------------------------------------------------------------------------------------
	float getSceneDepth(v2f IN)
	{
		float depth = tex2D(_CameraDepthTexture, IN.uv / IN.uv.w);
		return LinearEyeDepth(depth);
	}
	ENDCG

	//-----------------------------------------------------------------------------------------------------------------------------------------
	SubShader
	{
		//--------------------------------------------------------------------------------------------------------------------------------------
		Pass // 0: render back-faces , store minimum depth in alpha
		{
			Name "Render Backfaces"

			ZWrite Off
			AlphaTest Off
			ZTest Off

			Cull Front
			Blend Off

			CGPROGRAM
			#pragma vertex vertVolume
			#pragma fragment fragBack

			half4 fragBack(v2f IN) : COLOR
			{
				float volumeZ = getVolumeDepth(IN);
				float sceneZ = getSceneDepth(IN);
				float depth = min(sceneZ, volumeZ);
				return half4(0, -2, 0, depth);
			}
			ENDCG
		}

		//--------------------------------------------------------------------------------------------------------------------------------------
		Pass // 1: render front faces, store texture coord rgb, add negative depth in alpha (=> actual depth of the volume at that pixel)
		{
			Name "Render Frontfaces"

			Cull Back
			Blend One Zero, One One

			CGPROGRAM
			#pragma vertex vertVolume
			#pragma fragment fragFront

			half4 fragFront(v2f IN) : COLOR
			{
				float volumeZ = getVolumeDepth(IN);
				float sceneZ = getSceneDepth(IN);
				if (sceneZ < volumeZ)
					return half4(-2,0,0,0);

				return float4(IN.gridPos, -volumeZ);
			}
			ENDCG
		}

		//--------------------------------------------------------------------------------------------------------------------------------------
		Pass // 2: edge detection filter
		{
			Name "Edge filter"

			Cull Off
			Blend Off

			CGPROGRAM
			#pragma target 5.0
			#pragma vertex vertEdge
			#pragma fragment fragEdge

			//-----------------------------------------------------------------------------------------------------------------------------------------
			uniform sampler2D _MainTex;
			uniform float2 TexelSize;
			uniform float EdgeThreshold;

			//-----------------------------------------------------------------------------------------------------------------------------------------
			struct v2fEdge
			{
				float4	pos		: SV_POSITION;
				float2	uv00	: TEXCOORD0;
				float2	uv01	: TEXCOORD1;
				float2	uv02	: TEXCOORD2;
				float2	uv10	: TEXCOORD3;
				float2	uv12	: TEXCOORD4;
				float2	uv20	: TEXCOORD5;
				float2	uv21	: TEXCOORD6;
				float2	uv22	: TEXCOORD7;
			};

			//-----------------------------------------------------------------------------------------------------------------------------------------
			v2fEdge vertEdge(appdata IN)
			{
				v2fEdge OUT;
				OUT.pos = UnityObjectToClipPos(float4(IN.pos,1));

				float2 center = IN.uv;
				
				OUT.uv00 = center + float2(-TexelSize.x,-TexelSize.y);
				OUT.uv01 = center + float2(-TexelSize.x, 0);
				OUT.uv02 = center + float2(-TexelSize.x, TexelSize.y);

				OUT.uv10 = center + float2(0, -TexelSize.y);
				OUT.uv12 = center + float2(0, TexelSize.y);

				OUT.uv20 = center + float2(TexelSize.x,-TexelSize.y);
				OUT.uv21 = center + float2(TexelSize.x, 0);
				OUT.uv22 = center + float2(TexelSize.x, TexelSize.y);

				return OUT;
			}

			float EdgeDetectScalar(float sx, float sy, float threshold)
			{
				float dist = (sx*sx + sy*sy);
				return (dist > threshold*_ProjectionParams.z) ? 1 : 0;
			}

			float4 fragEdge(v2fEdge IN) : COLOR
			{
				float4 col;
				float g00,g01,g02;
				float g10,g12;
				float g20,g21,g22;

				col = tex2D(_MainTex, IN.uv00);
				g00 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv01);
				g01 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv02);
				g02 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv10);
				g10 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv12);
				g12 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv20);
				g20 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv21);
				g21 = (col.g >= 0) ? col.a : -col.a;

				col = tex2D(_MainTex, IN.uv22);
				g22 = (col.g >= 0) ? col.a : -col.a;

				float sx = 0;
				sx -= g00;
				sx -= g01 * 2;
				sx -= g02;
				sx += g20;
				sx += g21 * 2;
				sx += g22;

				float sy = 0;
				sy -= g00;
				sy -= g10 * 2;
				sy -= g20;
				sy += g02;
				sy += g12 * 2;
				sy += g22;

				float e = EdgeDetectScalar(sx, sy, EdgeThreshold);

				return float4(e,e,e,1);
			}
			ENDCG
		}
	}
}