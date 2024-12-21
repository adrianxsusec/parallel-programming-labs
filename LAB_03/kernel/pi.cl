__kernel void pi(__global float* output, const long n) {
    int gid = get_global_id(0);
    int gsize = get_global_size(0);

    float sum = 0.0;
    for (float i = gid; i < n; i += gsize) {
        float x = (i - 0.5) / n;
        sum += 1.0 / (1.0 + x * x);
    }

    output[gid] = 4.0 * sum / n;
}