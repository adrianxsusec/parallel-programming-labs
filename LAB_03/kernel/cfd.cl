__kernel void jacobistep(__global float *psi, __global float *psitmp, int m, int n) {
    int global_id = get_global_id(0);
    int c = m + 2;
    if (global_id >= m + 3 && global_id <= m * m + 2 * m + n && (global_id % (m + 2) != m + 1 && global_id % (m + 2) != 0)) {
        psitmp[global_id] = 0.25 * (psi[global_id - c] + psi[global_id + c] + psi[global_id - 1] + psi[global_id + 1]);
    }
}


__kernel void copy(__global float *psitmp, __global float *psi, int m, int n) {
    int global_id = get_global_id(0);
    if (global_id >= m + 3 && global_id <= m * m + 2 * m + n && (global_id % (m + 2) != m + 1 && global_id % (m + 2) != 0)) {
        psi[global_id] = psitmp[global_id];
    }
}


__kernel void deltasq(__global float *psi, __global float *psitmp, __global float *sum, int m, int n) {
    int global_id = get_global_id(0);
    if (global_id >= m + 3 && global_id <= m * m + 2 * m + n && (global_id % (m + 2) != m + 1 && global_id % (m + 2) != 0)) {
        float temp = psitmp[global_id] - psi[global_id];
        sum[global_id] += (temp * temp);
    }
}
