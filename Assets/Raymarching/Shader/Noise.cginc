//--------------------------------------------------------------------------------------------------------------------------------------------------
// https://thebookofshaders.com/10/
//--------------------------------------------------------------------------------------------------------------------------------------------------
float random(float3 v)
{
	return frac(sin(dot(v.xyz, float3(12.9898, 78.233, 159.351))) * 43758.5453123);
}

float2 random2(float3 v) 
{
    return frac(sin(float2(dot(v.xy,float2(127.1,311.7)),dot(v.zy,float2(269.5,183.3))))*43758.5453);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
float noise(float3 v)
{
	float3 i = floor(v);
	float3 f = frac(v);

	float3 t = f * f * (3.0 - 2.0 * f); // cubic hermite
	
	float x0y0z0 = random(i);
	float x1y0z0 = random(i + float3(1,0,0));
	float x0y1z0 = random(i + float3(0,1,0));
	float x1y1z0 = random(i + float3(1,1,0));
	float x0y0z1 = random(i + float3(0,0,1));
	float x1y0z1 = random(i + float3(1,0,1));
	float x0y1z1 = random(i + float3(0,1,1));
	float x1y1z1 = random(i + float3(1,1,1));

	float y0z0 = lerp(x0y0z0, x1y0z0, t.x);
	float y1z0 = lerp(x0y1z0, x1y1z0, t.x);
	float y0z1 = lerp(x0y0z1, x1y0z1, t.x);
	float y1z1 = lerp(x0y1z1, x1y1z1, t.x);

	float z0 = lerp(y0z0, y1z0, t.y);
	float z1 = lerp(y0z1, y1z1, t.y);

	return lerp(z0, z1, t.z);
}
