import json
import random


def generate_knapsack_instance_problem_1(num_objects=100, min_weight=5, max_weight=100):
    objects = []

    for i in range(1, num_objects + 1):
        weight = random.randint(min_weight, max_weight)
        value = random.randint(weight * 2, weight * 15)
        objects.append({"id": i, "weight": weight, "value": value})

    total_weight = sum(obj["weight"] for obj in objects)
    capacity = int(total_weight * 0.4)

    return {"capacity": capacity, "objects": objects}


def generate_knapsack_instance_problem_2(
    num_projects=100,
    min_value=5000,
    max_value=30000,
):
    projects = []

    for i in range(1, num_projects + 1):
        value = random.randint(min_value, max_value)

        cost = random.randint(int(value * 0.25), int(value * 0.45))

        base_time = cost / 1000
        time = random.randint(int(base_time * 5), int(base_time * 15))
        time = max(10, min(time, 180))

        projects.append(
            {
                "id": i,
                "value": value,
                "cost": cost,
                "time": time,
            }
        )

    total_cost = sum(p["cost"] for p in projects)
    total_time = sum(p["time"] for p in projects)

    budget_total = int(total_cost * 0.6)
    time_total = int(total_time * 0.65)

    return {
        "budget_total": budget_total,
        "time_total": time_total,
        "projects": projects,
    }


problem_1 = generate_knapsack_instance_problem_1()
with open("problem_1.json", "w") as f:
    json.dump(problem_1, f, indent=4)

problem_2 = generate_knapsack_instance_problem_2()
with open("problem_2.json", "w") as f:
    json.dump(problem_2, f, indent=4)
