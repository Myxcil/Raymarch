//--------------------------------------------------------------------------------------------------------------------------------------------------
// https://thebookofshaders.com/10/
//--------------------------------------------------------------------------------------------------------------------------------------------------
float PerlinNoise(float2 v)
{
	float2 i = floor(v);
	float2 f = frac(v);

	float2 t = f * f * (3.0 - 2.0 * f); // cubic hermite
	
	float x0y0 = random(i);
	float x1y0 = random(i + float2(1,0));
	float x0y1 = random(i + float2(0,1));
	float x1y1 = random(i + float2(1,1));

	float y0 = lerp(x0y0, x1y0, t.x);
	float y1 = lerp(x0y1, x1y1, t.x);

	return lerp(y0, y1, t.y);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
float PerlinNoise(float3 v)
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
