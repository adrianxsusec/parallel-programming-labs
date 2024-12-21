int is_prime(int n) {
        if (n < 2)
            return 0;
        for (int i = 2; i <= sqrt((float)n); i++) {
            if (n % i == 0)
                return 0;
        }
        return 1;
    }

__kernel void primes(__global const int *arr, __global int *out, const int elems_per_thread, const int is_atomic) {
    int gid = get_global_id(0);
    int count = 0;
    for (int i = 0; i < elems_per_thread; i++) {
        if (is_prime(arr[gid * elems_per_thread + i])) {
            count++;
        }
    }
    if (is_atomic) {
        atomic_add(out, count);
    }
    else {
        out += count;
    }
}