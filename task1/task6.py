import numpy as np


def vect_run_length_encode(x):
    """
     Vectorized implementation of Run-length encoding

     Input parameter:
        x: 1d-numpy.array, input array

     Output parameters:
        numbers: 1d-numpy.array, different numbers in x
        counts: 1d-numpy.array, count of repeated numbers
    """
    numbers = []
    x_size = x.shape[0]

    tmp = np.diff(x)
    num_inds = np.where(tmp != 0)[0]
    numbers = np.append(numbers, x[num_inds])
    numbers = np.append(numbers, x[x_size - 1])

    tmp = np.array([1])
    tmp = np.append(tmp, np.diff(x))
    tmp = np.append(tmp, 1)
    cnt_inds = np.where(tmp != 0)[0]
    counts = np.diff(cnt_inds)

    return numbers, counts


def run_length_encode(x):
    """
     Implementation with loops of Run-length encoding

     Input parameter:
         x: 1d-numpy.array, input array
     Output parameters:
        numbers: 1d-numpy.array, different numbers in x
        counts: 1d-numpy.array, count of repeated numbers
    """
    numbers = []
    counts = []
    number = x[0]
    cnt = 0
    x_size = x.shape[0]

    for i in range(x_size):
        if number == x[i]:
            cnt += 1
        else:
            numbers.append(number)
            counts.append(cnt)
            cnt = 1
            number = x[i]
    numbers.append(number)
    counts.append(cnt)
    numbers = np.array(numbers)
    counts = np.array(counts)

    return numbers, counts


def other_method(x):
    """
     The third implementation of Run-length encoding

     Input parameter:
        x: 1d-numpy.array, input array

     Output parameters:
        numbers: 1d-numpy.array, different numbers in x
        counts: 1d-numpy.array, count of repeated numbers
    """
    x_size = x.shape[0]

    tmp = np.array([1])
    tmp = np.append(tmp, np.diff(x))
    tmp = np.append(tmp, 1)
    counts = np.diff(np.where(tmp != 0)[0])
    cnt_size = counts.shape[0]

    cnt_it = 0
    cnt = counts[0]
    numbers = []

    for i in range(x_size):
        if cnt == 1:
            numbers.append(x[i])
            cnt_it += 1
            if cnt_it == cnt_size:
                break
            cnt = counts[cnt_it]
        else:
            cnt -= 1
    numbers = np.array(numbers)

    return numbers, counts
