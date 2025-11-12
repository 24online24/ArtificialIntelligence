import random
import json
import copy

with open('problem_1.json') as f:
    problem_1_data = json.load(f)

CAPACITY = problem_1_data['capacity']
item_data: list[dict] = problem_1_data['objects']

ITEMS: list[tuple[int, int]] = [(item['value'], item['weight']) for item in item_data]

MUTATION_RATE: float = 0.05
CROSSOVER_RATE: float = 0.8


class Chromosome:
    def __init__(self, selected: list[bool] | None = None):
        if selected is not None:
            self.selected = selected.copy()
        else:
            self.selected = [random.random() < 0.5 for _ in range(len(ITEMS))]
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self) -> int:
        total_value = 0
        total_weight = 0
        for gene, (value, weight) in zip(self.selected, ITEMS):
            if gene:
                total_value += value
                total_weight += weight
                if total_weight > CAPACITY:
                    return 0
        return total_value

    def mutate(self):
        self.selected = [
            not gene if random.random() < MUTATION_RATE else gene
            for gene in self.selected
        ]
        self.fitness = self.calculate_fitness()

    def decode(self) -> tuple[int, int, list[int]]:
        total_weight = 0
        for (_, weight), gene in zip(ITEMS, self.selected):
            if gene:
                total_weight += weight
        selected_indices = [i for i, gene in enumerate(self.selected) if gene]
        return self.fitness, total_weight, selected_indices


def create_population(pop_size: int) -> list[Chromosome]:
    return [Chromosome() for _ in range(pop_size)]


def tournament_selection(population: list[Chromosome], tournament_size: int) -> Chromosome:
    participants = random.sample(population, tournament_size)
    return max(participants, key=lambda chromosome: chromosome.fitness)


def uniform_crossover(parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
    child_1_genes, child_2_genes = [], []
    for gene_parent_1, gene_parent_2 in zip(parent1.selected, parent2.selected):
        if random.random() < 0.5:
            child_1_genes.append(gene_parent_1)
            child_2_genes.append(gene_parent_2)
        else:
            child_1_genes.append(gene_parent_2)
            child_2_genes.append(gene_parent_1)
    return Chromosome(child_1_genes), Chromosome(child_2_genes)


def genetic_algorithm(population_size: int, generations: int) -> Chromosome:
    population = create_population(population_size)
    best_chromosome = copy.deepcopy(population[0])

    for generation in range(generations):
        best_in_generation = max(population, key=lambda chromosome: chromosome.fitness)

        if best_in_generation.fitness > best_chromosome.fitness:
            best_chromosome = copy.deepcopy(best_in_generation)
        if generation % 100 == 0:
            print(f"Generation: {generation} | Best current: {best_in_generation.fitness} | Best ever: {best_chromosome.fitness}")
        next_population = [copy.deepcopy(best_chromosome)]

        while len(next_population) < population_size:
            parent1 = tournament_selection(population, 5)
            parent2 = tournament_selection(population, 5)

            if random.random() < CROSSOVER_RATE:
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = Chromosome(parent1.selected.copy()), Chromosome(parent2.selected.copy())

            child1.mutate()
            child2.mutate()

            next_population.extend([child1, child2])

        population = next_population

    return best_chromosome


if __name__ == '__main__':
    random.seed(42)

    best_solution = genetic_algorithm(200, 2000)
    value, weight, indices = best_solution.decode()
    print(f"\nBest solution: value={value} weight={weight}/{CAPACITY} items={indices}")
