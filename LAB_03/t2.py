import argparse
import time

import numpy as np
import pyopencl as cl


def main():
    parser = argparse.ArgumentParser(description="LAB_03 Task 2")
    parser.add_argument("-N", type=int, help="10 ^ N")
    args = parser.parse_args()

    context = cl.create_some_context(interactive=False)
    device = context.get_info(cl.context_info.DEVICES)[0]
    queue = cl.CommandQueue(context)

    with open("kernel/pi.cl") as f:
        kernel = f.read()

    args_n = args.N if args.N else 4
    N = 10 ** args_n
    G = 2 ** 12
    L = 64

    program = cl.Program(context, kernel).build()
    mem_flags = cl.mem_flags
    pi_result = cl.Buffer(context, mem_flags.READ_WRITE, size=G * np.dtype(np.float32).itemsize, hostbuf=None)

    start_time = time.time()
    program.pi(queue, [G], [L], pi_result, np.int64(N))
    queue.finish()
    duration = time.time() - start_time

    pi = np.empty(G, dtype=np.float32)
    cl.enqueue_copy(queue, pi, pi_result).wait()
    pi = sum(pi)
    print(f"Time: {duration}s")
    print(f"Pi: {pi}")
    print(f"Error: {abs(np.pi - pi)}")


if __name__ == "__main__":
    main()
