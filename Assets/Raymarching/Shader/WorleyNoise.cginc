//--------------------------------------------------------------------------------------------------------------------------------------------------
float WorleyNoise(float2 v)
{
	float2 i_v = floor(v);
	float2 f_v = frac(v);

	float minDist = 1.0;
	for (int y = -1; y <= 1; ++y)
	{
		for (int x = -1; x <= 1; ++x)
		{
			float2 neighbor = float2(x,y);
			float2 p = random2(i_v + neighbor);
			float2 diff = neighbor + p - f_v;
			float dist = length(diff);
			minDist = min(minDist,dist);
		}
	}
	return minDist;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
float WorleyNoise(float3 v)
{
	float3 i_v = floor(v);
	float3 f_v = frac(v);

	float minDist = 1.0;
	for(int z=-1; z <= 1; ++z)
	{
		for (int y = -1; y <= 1; ++y)
		{
			for (int x = -1; x <= 1; ++x)
			{
				float3 neighbor = float3(x,y,z);
				float3 p = random3(i_v + neighbor);
				float3 diff = neighbor + p - f_v;
				float dist = length(diff);
				minDist = min(minDist,dist);
			}
		}
	}
	return minDist;
}

