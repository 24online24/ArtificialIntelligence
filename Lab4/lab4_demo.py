import random
from dataclasses import dataclass

import operators


def fitness(c):
    return int(c, 2) ** 2


def turneu(P):
    a, b = random.sample(P, 2)
    return min(a, b, key=fitness)


def copil(a, b, pc, pm):
    c = a
    if random.random() < pc:
        k = random.randint(1, 4)
        c = a[:k] + b[k:]
    c = "".join(
        str(1 - int(bit)) if random.random() < pm else bit
        for bit in c
    )
    # Toate valorile 0..31 sunt valide in exercitiul curent.
    return c


def ga_min_square(N=8, G=30, pc=0.8, pm=0.05):
    """Minimizeaza x^2 pe intregii 0..31; rezultatul nu este garantat optim."""
    P = [format(random.randint(0, 31), "05b") for _ in range(N)]
    best = min(P, key=fitness)
    for _ in range(G):
        P = [copil(turneu(P), turneu(P), pc, pm) for _ in range(N)]
        best = min(best, min(P, key=fitness), key=fitness)
    return best, int(best, 2), fitness(best)


@dataclass
class Individ:
    name: str
    fitness: float  # COST: o valoare mai mica este mai buna.


def demo_permutari():
    random.seed(42)
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [6, 5, 4, 3, 2, 1]
    print("Permutari: fiecare oras trebuie sa apara exact o data.")
    for name, function in [
        ("one_point", operators.one_point_crossover),
        ("two_point", operators.two_point_crossover),
        ("n_point (3)", lambda a, b: operators.n_point_crossover(a, b, 3)),
        ("uniform", operators.uniform_crossover),
    ]:
        child = function(p1, p2)
        print(f"  {name}: {child}; valid={sorted(child) == sorted(p1)}")


def demo_selectie():
    random.seed(42)
    P = [Individ("A", 10), Individ("B", 30), Individ("C", 60)]
    print("\nSelectie pentru MINIMIZARE: costuri A=10, B=30, C=60.")
    total = sum(1 / c.fitness for c in P)
    print("  Ponderi ruleta: 1/10, 1/30, 1/60")
    print("  Probabilitati:", [round((1 / c.fitness) / total, 4) for c in P])
    for name, function in [
        ("random", operators.random_selection),
        ("roulette", operators.roulette_wheel_selection),
        ("tournament (2)", lambda p: operators.tournament_selection(p, 2)),
        ("rank", operators.rank_selection),
        ("Boltzmann, generatia 1000", lambda p: operators.boltzmann_selection(p, 1000)),
    ]:
        selected = function(P)
        print(f"  {name}: {selected.name}, cost={selected.fitness}")
    print("  elite (1):", [c.name for c in operators.elite_selection(P, 1)])
    print("  O extragere este un exemplu, nu garantia unui rezultat.")


def main():
    demo_permutari()
    demo_selectie()
    random.seed(42)
    bits, x, value = ga_min_square()
    print(f"\nGA pentru x^2: cromozom={bits}, x={x}, f(x)={value}")
    print("Reper exact: x=0, f(x)=0. Alte rulari pot da rezultate suboptime.")


if __name__ == "__main__":
    main()
