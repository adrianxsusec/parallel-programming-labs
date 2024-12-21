import time

import numpy as np


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


def deltasq(psi, psitmp, error_array, m, n):
    for global_id in range(m + 3, m * m + 2 * m + n):
        if global_id % (m + 2) != m + 1 and global_id % (m + 2) != 0:
            temp = psitmp[global_id] - psi[global_id]
            error_array[global_id] += (temp * temp)


def copy(psitmp, psi, m, n):
    for global_id in range(m + 3, m * m + 2 * m + n):
        if global_id % (m + 2) != m + 1 and global_id % (m + 2) != 0:
            psi[global_id] = psitmp[global_id]


def jacobistep(psi, psitmp, m, n):
    c = m + 2
    for global_id in range(m + 3, m * m + 2 * m + n):
        if global_id % (m + 2) != m + 1 and global_id % (m + 2) != 0:
            psitmp[global_id] = 0.25 * (
                    psi[global_id - c] + psi[global_id + c] + psi[global_id - 1] + psi[global_id + 1])


def main():
    scale_factor = int(input("Faktor skaliranja: "))
    num_iter = int(input("Broj iteracija: "))

    error = None

    bbase = 10
    hbase = 15
    wbase = 5
    mbase = 32
    nbase = 32

    b = bbase * scale_factor
    h = hbase * scale_factor
    w = wbase * scale_factor
    m = mbase * scale_factor
    n = nbase * scale_factor

    psi = np.zeros((m + 2) * (n + 2), dtype=np.float32)
    psitmp = np.zeros((m + 2) * (n + 2), dtype=np.float32)
    error_array = np.zeros(((m + 2) * (n + 2)), dtype=np.float32)
    psi = psi_boundary(psi=psi, m=m, n=n, b=b, h=h, w=w)

    b_norm = 0.0
    for i in range(0, m + 2):
        for j in range(0, n + 2):
            b_norm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j]
    b_norm = np.sqrt(b_norm)

    time_start = time.time()

    for i in range(1, num_iter + 1):
        jacobistep(psi, psitmp, m, n)
        if i == num_iter:
            deltasq(psi, psitmp, error_array, m, n)
            error = np.sqrt(np.sum(error_array)) / b_norm
        copy(psitmp, psi, m, n)

    print(f"Greška: {error}")
    print(f"Ukupno vrijeme: {time.time() - time_start}")


if __name__ == "__main__":
    main()
