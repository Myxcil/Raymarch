//--------------------------------------------------------------------------------------------------------------------------------------------------
Shader "Hidden/Raymarching" 
{
	Properties{
		_MainTex("", 2D) = "" {}
	}
		
	//----------------------------------------------------------------------------------------------------------------------------------------------
	CGINCLUDE
	#pragma target 5.0
	#include "UnityCG.cginc"
	#pragma exclude_renderers gles

	//--------------------------------------------------------------------------------------------------------------------------------------
	uniform sampler2D _MainTex;
	uniform sampler2D _RayTexture;
	uniform sampler3D _Volume;
	
	uniform sampler2D _JitterTex;
	uniform float _JitterStrength;

	uniform float3 _Absorption;
	uniform float _Density;

	uniform float4x4 _invMVP;
	uniform float4 _CameraTS;
	uniform float4 _GridDim;
	uniform float4 _rcpGridDim;
	uniform float _SampleScale;

	uniform float3 _NearPlanePos;

	//--------------------------------------------------------------------------------------------------------------------------------------
	struct appdata
	{
		float3	pos			: POSITION;
	};

	//--------------------------------------------------------------------------------------------------------------------------------------
	struct v2f 
	{
		float4	pos			: SV_POSITION;
		float2	uv			: TEXCOORD0;
		float3	gridPos		: TEXCOORD1;
		float3	view		: TEXCOORD2;
	};

	//--------------------------------------------------------------------------------------------------------------------------------------
	v2f vertQuad(appdata IN)
	{
		v2f OUT;
		OUT.pos = UnityObjectToClipPos(float4(IN.pos,1));
		OUT.uv = IN.pos.xy;

		float3 p;
		p.x = 2 * IN.pos.x - 1;
		p.y = 1 - 2 * IN.pos.y;
		p.z = 1;

		float4 gp = float4(p, 1) *  _ProjectionParams.y;
		OUT.gridPos = mul(_invMVP, gp) + 0.5;

		OUT.view = _NearPlanePos * p;
		OUT.view /= OUT.view.z;

		return OUT;
	}
	
	//--------------------------------------------------------------------------------------------------------------------------------------
	void SampleVolume(float weight, float4 p, inout float4 result)
	{
		float4 s = tex3Dlod(_Volume, p);
		result.xyz *= 1 - saturate(s.xyz * _Absorption);
		result.w += _Density * s.z * (1 - result.w);
	}

	//--------------------------------------------------------------------------------------------------------------------------------------
	float4 Raycast(sampler2D rayTexture, float2 uv, float3 gridPos, float3 view)
	{
		float4 ray = tex2D(rayTexture, uv);
		if (ray.x < -1)
			return 0;

		ray.w *= length(view);

		if (ray.y < -1)
		{
			ray.xyz = gridPos;
			ray.w -= _ProjectionParams.y;
		}

		float fSamples = ray.w * _SampleScale;
		int nSamples = floor(fSamples);
		if (nSamples <= 0)
			return 0;

		float offset = 1.0f + _JitterStrength * tex2D(_JitterTex, uv * 0.25).r;
		float3 stepVec = normalize((ray.xyz - _CameraTS.xyz) * _GridDim.xyz) * _rcpGridDim.xyz;
		float4 p = float4(ray.xyz + stepVec * offset, 0);

		float4 result = float4(1, 1, 1, 0);
		for(int i=0; i < nSamples; ++i)
		{	
			SampleVolume(1.0f, p, result);
			if (result.w > 0.999)
			{
				break;
			}

			p.xyz += stepVec;
		}
		if (i == nSamples)
		{
			SampleVolume(frac(fSamples), p, result);
		}

		return result;
	}

	//------------------------------------------------------------------------------------------------------------------------------------------
	float4 CalculateFinalColor(float4 data, float2 uv, float3 gridPos, float3 view)
	{
		return data;
	}
	ENDCG

	//----------------------------------------------------------------------------------------------------------------------------------------------
	SubShader  
	{
		//------------------------------------------------------------------------------------------------------------------------------------------
		Pass // 0: Raycast volume texture, store data in target texture
		{
			Name "Raycast Volume"

			//--------------------------------------------------------------------------------------------------------------------------------------
			Cull Back
			ZTest Less
			ZWrite Off
    		Blend Off
			
			//--------------------------------------------------------------------------------------------------------------------------------------
			CGPROGRAM
			#pragma vertex vertQuad
			#pragma fragment frag

			half4 frag(v2f IN) : COLOR
			{
				return Raycast(_MainTex, IN.uv, IN.gridPos, IN.view);
			}
			ENDCG
		} 

		//------------------------------------------------------------------------------------------------------------------------------------------
		Pass // 1: Render layer, raycast "problematic" edges again in higher resolution
		{
			Name "Raycast Edges"

			//--------------------------------------------------------------------------------------------------------------------------------------
			Cull Off
			ZTest Always
			ZWrite Off

			Blend Off

			//--------------------------------------------------------------------------------------------------------------------------------------
			CGPROGRAM
			#pragma vertex vertQuad
			#pragma fragment frag

			//--------------------------------------------------------------------------------------------------------------------------------------
			uniform sampler2D _EdgeLookup;
		
			//--------------------------------------------------------------------------------------------------------------------------------------
			float4 frag(v2f IN) : COLOR
			{
				float edge = tex2D(_EdgeLookup, IN.uv);
				
				float4 data;
				if (edge > 0)
				{
					data = Raycast(_RayTexture, IN.uv, IN.gridPos, IN.view);
				}
				else
				{
					data = tex2D(_MainTex, IN.uv);
				}
				return CalculateFinalColor(data, IN.uv, IN.gridPos, IN.view);
			}
			ENDCG
		} 

		//------------------------------------------------------------------------------------------------------------------------------------------
		Pass // 2: render layer, raycast at full resolution
		{
			Name "Raycast FullRes"

			//--------------------------------------------------------------------------------------------------------------------------------------
			Cull Off
			ZTest Always
			ZWrite Off

			Blend Off
							
			//--------------------------------------------------------------------------------------------------------------------------------------
			CGPROGRAM
			#pragma vertex vertQuad
			#pragma fragment frag

			//--------------------------------------------------------------------------------------------------------------------------------------
			float4 frag(v2f IN) : COLOR
			{
				float4 data = Raycast(_RayTexture, IN.uv, IN.gridPos, IN.view);
				return CalculateFinalColor(data, IN.uv, IN.gridPos, IN.view);
			}
			ENDCG
		} 
	}
}
