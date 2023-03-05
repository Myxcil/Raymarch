//--------------------------------------------------------------------------------------------------------------------------------------------------
float3 random3(float3 v) 
{
    return frac(sin(float3( dot(v,float3(127.1,311.7,281.3)), dot(v,float3(269.5,183.3,67.7)), dot(v,float3(12.9898, 78.233, 159.351))))*43758.5453);
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
