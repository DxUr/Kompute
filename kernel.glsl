#version 430 core

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer In1 {
    float InData1[];
};

layout(binding = 1) readonly buffer In2 {
    float InData2[];
};

layout(binding = 2) buffer Out {
    float OutData[];
};

uniform ivec3 dims;

#define GET_VEC3(bff, uv) vec3(bff[(uv.y * dims.x + uv.x) * dims.z],\
    bff[(uv.y * dims.x + uv.x) * dims.z + 1],\
    bff[(uv.y * dims.x + uv.x) * dims.z + 2])
#define SET_VEC3(bff, v, uv) bff[(uv.y * dims.x + uv.x) * dims.z] = v.x,\
    bff[(uv.y * dims.x + uv.x) * dims.z + 1] = v.y,\
    bff[(uv.y * dims.x + uv.x) * dims.z + 2] = v.z
#define SET_VAL(bff, v, uv) bff[uv.y * dims.x + uv.x] = v



vec3[2] gaussianBlur(uvec2 coord) {
	float[9] kernel = float[9](
		1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
		2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
		1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
	);

	const ivec2 offset[9] = ivec2[](
		ivec2(-1,  1), ivec2( 0,  1), ivec2( 1,  1),
		ivec2(-1,  0), ivec2( 0,  0), ivec2( 1,  0),
		ivec2(-1, -1), ivec2( 0, -1), ivec2( 1, -1)
	);

	vec3 r1 = vec3(0.0);
	vec3 r2 = vec3(0.0);

	for (int i = 0; i < 9; i++) {
		uvec2 coordOffset = coord + offset[i];
		r1 += GET_VEC3(InData1, coordOffset) * kernel[i];
		r2 += GET_VEC3(InData2, coordOffset) * kernel[i];
	}

	return vec3[](r1, r2);
}

vec3[2] sobelEdgeDetection(uvec2 coord) {
	// Sobel kernels for X and Y directions
	float kernelX[9] = float[9](
		-1.0,  0.0,  1.0,
		-2.0,  0.0,  2.0,
		-1.0,  0.0,  1.0
	);

	float kernelY[9] = float[9](
		-1.0, -2.0, -1.0,
		0.0,  0.0,  0.0,
		1.0,  2.0,  1.0
	);

	const ivec2 offset[9] = ivec2[](
		ivec2(-1,  1), ivec2( 0,  1), ivec2( 1,  1),
		ivec2(-1,  0), ivec2( 0,  0), ivec2( 1,  0),
		ivec2(-1, -1), ivec2( 0, -1), ivec2( 1, -1)
	);

	vec2 grad1 = vec2(0.0);
    vec2 grad2 = vec2(0.0);

	// Convolve the kernel over the texture
	for (int i = 0; i < 9; i++) {
		uvec2 coordOffset = coord + offset[i];
		vec3 c1 = GET_VEC3(InData1, coordOffset);
		vec3 c2 = GET_VEC3(InData2, coordOffset);
		float i1 = dot(c1, vec3(0.299, 0.587, 0.114));  // Convert to grayscale
        float i2 = dot(c2, vec3(0.299, 0.587, 0.114));
        const vec2 ker = vec2(kernelX[i], kernelY[i]);
        grad1 += ker * i1;
        grad2 += ker * i2;
	}

	// Calculate gradient magnitude (edge strength)
	float e1 = length(grad1);
	float e2 = length(grad2);
	return vec3[2](vec3(e1), vec3(e2));
}

vec3 threshold(vec3 value, float thresh) {
	return step(vec3(thresh), value);  // Keep pixels above the threshold
}

vec3 computeMotionMask(vec3 edge, vec3 blur, float edgeWeight, float blurWeight) {
    // Normalize edge and blur to ensure they are within range [0, 1]
    vec3 normalizedEdge = edge / max(dot(edge, vec3(1.0)), 1e-6);
    vec3 normalizedBlur = blur / max(dot(blur, vec3(1.0)), 1e-6);
    
    // Combine the edge and blur results
    vec3 mask = mix(normalizedBlur, normalizedEdge, edgeWeight);
    
    // Optionally, apply some additional processing if needed
    return clamp(mask, 0.0, 1.0);
}


void main() {
    const uvec2 Coord = gl_GlobalInvocationID.xy;
    /*
     * Start Shader Code
     */

	// Step 1: Calculate Sobel edge detection
	vec3[2] edge = sobelEdgeDetection(Coord);
	vec3[2] blur = gaussianBlur(Coord);
    
    vec3 outColor = max(abs(edge[0] - edge[1]), abs(blur[0] - blur[1]));
    outColor = threshold(outColor, 0.2);

    float v = (outColor.x + outColor.y + outColor.z) * 255.0 / 3.0;

    /*
     * End Shader Code
     */
    
    SET_VAL(OutData, v, Coord);
}
