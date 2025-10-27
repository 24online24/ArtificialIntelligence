import random
import numpy as np


def one_point_crossover(p1, p2):
    length = len(p1)
    crossover_point = random.randint(2, length - 1)
    child = p1[:crossover_point]

    for city in p2:
        if city not in child:
            child.append(city)

    return child


def two_point_crossover(p1, p2):
    length = len(p1)
    left, right = sorted(random.sample(range(length), 2))
    child = [-1] * length
    child[left:right] = p1[left:right]
    used = set(p1[left:right])

    j = 0
    for i in range(length):
        if child[i] == -1:
            while p2[j] in used:
                j += 1
            child[i] = p2[j]
            used.add(p2[j])

    return child


def n_point_crossover(p1, p2, n):
    length = len(p1)
    crossover_points = sorted(random.sample(range(1, length), n))
    child = [-1] * length

    last_point = 0
    take_from_p1 = True
    for point in crossover_points + [length]:
        if take_from_p1:
            for i in range(last_point, point):
                child[i] = p1[i]
        last_point = point
        take_from_p1 = not take_from_p1
    used = set(child)

    j = 0
    for i in range(length):
        if child[i] == -1:
            while p2[j] in used:
                j += 1
            child[i] = p2[j]
            used.add(p2[j])

    return child


def uniform_crossover(p1, p2):
    length = len(p1)
    child = [-1] * length
    used = set()

    mask = [random.choice([True, False]) for _ in range(length)]

    for i in range(length):
        if mask[i]:
            child[i] = p1[i]
            used.add(p1[i])

    j = 0
    for i in range(length):
        if child[i] == -1:
            while p2[j] in used:
                j += 1
            child[i] = p2[j]
            used.add(p2[j])

    return child


def random_selection(population):
    return random.choice(population)


def roulette_wheel_selection(population):
    total_fitness = sum(1.0 / chromosome.fitness for chromosome in population)
    selection_point = random.uniform(0, total_fitness)
    accumulated_fitness = 0.0
    for chromosome in population:
        accumulated_fitness += 1.0 / chromosome.fitness
        if accumulated_fitness >= selection_point:
            return chromosome


def tournament_selection(population, tournament_size=30):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda chromosome: chromosome.fitness)


def rank_selection(population):
    sorted_population = sorted(population, key=lambda chromosome: chromosome.fitness)
    ranks = list(range(len(sorted_population), 0, -1))
    rank_sum = sum(ranks)
    probabilities = [rank / rank_sum for rank in ranks]
    selection_point = random.random()
    accumulated_probability = 0.0
    for chromosome, probability in zip(sorted_population, probabilities):
        accumulated_probability += probability
        if accumulated_probability >= selection_point:
            return chromosome


def elite_selection(population, elite_size=5):
    sorted_population = sorted(population, key=lambda chromosome: chromosome.fitness)
    return sorted_population[:elite_size]


GENERATIONS_NUMBER = 1000
INITIAL_TEMPERATURE = 100
FINAL_TEMPERATURE = 0.1
DELTA = (INITIAL_TEMPERATURE - FINAL_TEMPERATURE) / GENERATIONS_NUMBER


def boltzmann_selection(population, generation):
    def _normalize_fitness(fitness_values):
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        return [1 - ((f - min_fitness) / (max_fitness - min_fitness)) for f in fitness_values]

    temperature = max(FINAL_TEMPERATURE, INITIAL_TEMPERATURE - generation * DELTA)
    fitness_values = [chromosome.fitness for chromosome in population]
    normalized_scores = _normalize_fitness(fitness_values)
    exponentials = [np.exp(score / temperature) for score in normalized_scores]
    exponentials_sum = sum(exponentials)
    probabilities = [e / exponentials_sum for e in exponentials]
    selection_point = random.random()
    accumulated_probability = 0.0
    for chromosome, probability in zip(population, probabilities):
        accumulated_probability += probability
        if accumulated_probability >= selection_point:
            return chromosome
