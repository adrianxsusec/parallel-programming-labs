import argparse
import time

import numpy as np
import pyopencl as cl


def psi_boundary(psi: np.array, m, n, b, h, w):
    for i in range(b + 1, b + w):
        psi[i * (m + 2) + 0] = i - b

    for i in range(b + w, m + 1):
        psi[i * (m + 2) + 0] = w

    for j in range(1, h + 1):
        psi[(m + 1) * (m + 2) + j] = w

    for j in range(h + 1, h + 2):
        psi[(m + 1) * (m + 2) + j] = w - j + h

    return psi


def main():
    parser = argparse.ArgumentParser(description="LAB_03 Task 3")
    parser.add_argument("-S", type=int, help="Scale factor")
    parser.add_argument("-N", type=int, help="Number of iterations")
    args = parser.parse_args()

    print_freq = 1
    error = None
    tolerance = 0.0

    scale_factor = args.S if args.S else 64
    num_iter = args.N if args.N else 1000

    bbase = 10
    hbase = 15
    wbase = 5
    mbase = 32
    nbase = 32

    irrotational = 1
    check_err = False

    if not check_err:
        print(f"{scale_factor=}, {num_iter=}")
    else:
        print(f"{scale_factor=}, {num_iter=}, {tolerance=}")

    print("Irrotational flow")

    b = bbase * scale_factor
    h = hbase * scale_factor
    w = wbase * scale_factor
    m = mbase * scale_factor
    n = nbase * scale_factor

    print(f"Running CFD on {m} x {n} grid")

    psi = np.zeros((m + 2) * (n + 2), dtype=np.float32)
    error_array = np.zeros(((m + 2) * (n + 2)), dtype=np.float32)
    psi = psi_boundary(psi=psi, m=m, n=n, b=b, h=h, w=w)

    b_norm = 0.0
    for i in range(0, m + 2):
        for j in range(0, n + 2):
            b_norm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j]
    b_norm = np.sqrt(b_norm)

    print(b_norm)

    platform = cl.get_platforms()[0]
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)

    with open("kernel/cfd.cl", "r") as f:
        kernel_source = f.read()

    program = cl.Program(context, kernel_source)
    program.build()

    # device = platform.get_devices()[0]
    # print("Device name:", device.name)
    # double_support = device.get_info(cl.device_info.DOUBLE_FP_CONFIG)
    # if double_support:
    #     print("Double precision support: Yes")
    # else:
    #     print("Double precision support: No")
    #
    # print(program.all_kernels())

    psi_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=psi)
    psi_tmp_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=psi.nbytes)
    error_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=error_array.nbytes)

    print("Starting main loop!")
    time_start = time.time()

    print(num_iter)
    for i in range(1, num_iter + 1):
        res = program.jacobistep(queue, (psi.size,), None, psi_buffer, psi_tmp_buffer, np.int32(m), np.int32(n))
        res.wait()

        if check_err or i == num_iter:
            res = program.deltasq(queue, (psi.size,), None, psi_buffer, psi_tmp_buffer, error_buffer, np.int32(m),
                                  np.int32(n))
            res.wait()
            cl.enqueue_copy(queue, error_array, error_buffer)

            error = np.sqrt(np.sum(error_array)) / b_norm

        if check_err and error < tolerance:
            print(f"Converged in {i} iterations")
            break

        if i % print_freq == 0:
            if not check_err:
                print(f"Iteration {i}")
            else:
                print(f"Iteration {i}, error {error}")

        res = program.copy(queue, (psi.size,), None, psi_tmp_buffer, psi_buffer, np.int32(m), np.int32(n))
        res.wait()

    total_time = time.time() - time_start
    iteration_time = total_time / num_iter

    print("Finished main loop!")
    print(f"Error: {error}")
    print(f"Total time: {total_time}")
    print(f"Time per iteration: {iteration_time}")


if __name__ == "__main__":
    main()
