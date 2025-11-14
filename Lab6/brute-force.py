from itertools import product

def find_partition(numbers: list[int]):
    lowest_difference = 1e9
    best_mask = []
    for mask in product([0, 1], repeat=len(numbers)):
        s1 = 0
        s2 = 0
        for number, bit in zip(numbers, mask):
            if bit:
                s1 += number
            else:
                s2 += number
        diff = abs(s1 - s2)
        if diff < lowest_difference:
            lowest_difference = diff
            best_mask = mask
            if lowest_difference == 0:
                break

    partition_1 = []
    partition_2 = []
    for number, bit in zip(numbers, best_mask):
            if bit:
                partition_1.append(number)
            else:
                partition_2.append(number)

    return partition_1, partition_2, sum(partition_2) - sum(partition_1)

problem_1_1_data = [3, 5, 6, 7, 9, 11, 12, 13]
problem_1_2_data = [4, 7, 10, 11, 12, 14, 15, 18]
problem_2_data = [82, 75, 91, 68, 79, 95, 88, 73, 84, 77, 92, 69, 81, 87, 90, 76, 83, 71, 85, 80]

print(find_partition(problem_1_1_data))
print(find_partition(problem_1_2_data))
print(find_partition(problem_2_data))