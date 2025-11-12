import random
import json
import copy

with open('problem_1.json') as f:
    problem_1_data = json.load(f)

CAPACITY = problem_1_data['capacity']
item_data: list[dict] = problem_1_data['objects']

ITEMS: list[tuple[int, int]] = [(item['value'], item['weight']) for item in item_data]

MUTATION_RATE: float = 1 / len(ITEMS)
CROSSOVER_RATE: float = 0.8
TOURNAMENT_SIZE: int = 5


class Chromosome:
    """Binary representation of a knapsack solution.

    The chromosome is a list of booleans where ``True`` means the corresponding
    item is selected. Fitness equals the total value of selected items, or
    ``0`` if the total weight exceeds the knapsack capacity.
    """

    def __init__(self, selected: list[bool] | None = None):
        """Create a chromosome.

        Args:
            selected: Optional predefined gene list. If ``None``, a random
                selection is generated.
        """
        if selected is not None:
            self.selected = selected.copy()
        else:
            self.selected = [random.random() < 0.5 for _ in range(len(ITEMS))]
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self) -> int:
        """Compute the chromosome's fitness.

        Returns:
            Total value of selected items if the combined weight does not
            exceed ``CAPACITY``; otherwise ``0``.
        """
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
        """Flip each gene with probability ``MUTATION_RATE`` and update fitness."""
        self.selected = [
            not gene if random.random() < MUTATION_RATE else gene
            for gene in self.selected
        ]
        self.fitness = self.calculate_fitness()

    def decode(self) -> tuple[int, int, list[int]]:
        """Translate the chromosome into a readable solution.

        Returns:
            A tuple ``(value, weight, indices)`` where ``indices`` lists the
            positions of selected items.
        """
        total_weight = 0
        for (_, weight), gene in zip(ITEMS, self.selected):
            if gene:
                total_weight += weight
        selected_indices = [i for i, gene in enumerate(self.selected) if gene]
        return self.fitness, total_weight, selected_indices


def create_population(pop_size: int) -> list[Chromosome]:
    """Generate an initial population of random chromosomes.

    Args:
        pop_size: Number of chromosomes to create.

    Returns:
        List of ``Chromosome`` objects.
    """
    return [Chromosome() for _ in range(pop_size)]


def tournament_selection(population: list[Chromosome]) -> Chromosome:
    """Select a parent using tournament selection.

    Args:
        population: Current population of chromosomes.
        tournament_size: Number of individuals competing in each tournament.

    Returns:
        The chromosome with the highest fitness among the sampled participants.
    """
    participants = random.sample(population, TOURNAMENT_SIZE)
    return max(participants, key=lambda chromosome: chromosome.fitness)


def uniform_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
) -> tuple[Chromosome, Chromosome]:
    """Perform uniform crossover between two parents.

    Each gene is independently taken from either parent with equal probability.

    Args:
        parent1: First parent chromosome.
        parent2: Second parent chromosome.

    Returns:
        Two new ``Chromosome`` objects representing the children.
    """
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
    """Run the genetic algorithm to solve the knapsack problem.

    Args:
        population_size: Number of chromosomes in each generation.
        generations: How many generations to evolve.

    Returns:
        The best chromosome discovered during the run.
    """
    population = create_population(population_size)
    best_chromosome = copy.deepcopy(population[0])

    for generation in range(generations):
        best_in_generation = max(population, key=lambda c: c.fitness)

        if best_in_generation.fitness > best_chromosome.fitness:
            best_chromosome = copy.deepcopy(best_in_generation)

        if generation % 100 == 0:
            print(
                f"Generation: {generation} | Best current: "
                f"{best_in_generation.fitness} | Best ever: "
                f"{best_chromosome.fitness}"
            )

        next_population = [copy.deepcopy(best_chromosome)]

        while len(next_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < CROSSOVER_RATE:
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1 = Chromosome(parent1.selected.copy())
                child2 = Chromosome(parent2.selected.copy())

            child1.mutate()
            child2.mutate()

            next_population.extend([child1, child2])

        population = next_population[:population_size]

    return best_chromosome


if __name__ == '__main__':
    random.seed(42)

    best_solution = genetic_algorithm(200, 2000)
    value, weight, indices = best_solution.decode()
    print(
        f"\nBest solution: value={value} weight={weight}/{CAPACITY} "
        f"items={indices}"
    )
