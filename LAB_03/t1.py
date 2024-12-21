import time
import numpy as np
import pyopencl as cl
import argparse


def main():
    parser = argparse.ArgumentParser(description="LAB_03 Task 1")
    parser.add_argument("-N", type=int, help="Exponent of 2 - number of elements in the array")
    args = parser.parse_args()

    context = cl.create_some_context(interactive=False)
    device = context.get_info(cl.context_info.DEVICES)[0]
    queue = cl.CommandQueue(context)

    N = 2 ** args.N
    G = int(2 ** (args.N // 2)) # work items
    L = 16 # group size

    numbers = np.array(range(1, N + 1), dtype=np.int32)

    with open("kernel/primes.cl") as f:
        kernel = f.read()

    program = cl.Program(context, kernel).build()

    mem_flags = cl.mem_flags
    numbers_buffer = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=numbers)
    atomic_buffer = cl.Buffer(context, mem_flags.WRITE_ONLY, N, hostbuf=None)
    non_atomic_buffer = cl.Buffer(context, mem_flags.WRITE_ONLY, N, hostbuf=None)

    start_time = time.time()
    program.primes(queue, [G], [L], numbers_buffer, atomic_buffer, np.int32(N // G), np.int32(1))
    queue.finish()
    atomic_duration = time.time() - start_time

    start_time = time.time()
    program.primes(queue, [G], [L], numbers_buffer, non_atomic_buffer, np.int32(N // G), np.int32(0))
    queue.finish()
    non_atomic_duration = time.time() - start_time

    atomic_result = bytearray(4)
    cl.enqueue_copy(queue, atomic_result, atomic_buffer).wait()
    non_atomic_result = np.empty(G, dtype=np.int32)
    cl.enqueue_copy(queue, non_atomic_result, non_atomic_buffer).wait()

    print("Atomic time: ", atomic_duration)
    print("Non-atomic time: ", non_atomic_duration)
    print("Atomic primes: ", int.from_bytes(atomic_result, byteorder='little'))
    print("Non-atomic primes: ", np.sum(non_atomic_result))


if __name__ == "__main__":
    main()
