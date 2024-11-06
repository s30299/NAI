import random


def select_roulette_rule(fitnesses):
    sum_fit = 0
    selected = []
    for f in fitnesses: sum_fit += f
    for i in range(len(fitnesses)):
        r = random.random()*sum_fit
        sf = 0
        for j in range(len(fitnesses)):
            f = fitnesses[j]
            sf+=f
            if sf>r:
                selected.append(j) # append index of individual
                break
    return selected


def crossover(a,b,n):

    if random.randint(0, 100)>n:
        return a,b
    cross_point = random.randint(0,len(a)-1)
    new_a = [ a[i] if i < cross_point else b[i] for i in range(len(a)) ]
    new_b = [ b[i] if i < cross_point else a[i] for i in range(len(a)) ]
    return new_a, new_b

def mutation(a,b,n):
    a=mut_chance(a,n)
    b=mut_chance(b,n)
    return a,b
def mut_chance(a,n):
        for i in a:
            if random.randint(1, 100) <= n:
                if a[i] == 0:
                    a[i] = 1
                else:
                    a[i] = 0
        return a

def genetic_algorithm(fitness, random_solution, iterations, pop_size,chance_of_crossover, chance_of_mutation):
    population = [ random_solution() for _ in range(pop_size) ]

    for i in range(iterations): 
        fitnesses = [ fitness(population[i]) for i in range(len(population)) ]
        print("f:", fitnesses)
        selected = select_roulette_rule(fitnesses)
        print("  ", selected)
        new_population = []
        for i in range(int(len(population)/2)):
            a,b = crossover(population[selected[i*2]],population[selected[i*2+1]],chance_of_crossover)
            a,b = mutation(a,b,chance_of_mutation)
            new_population.append(a)
            new_population.append(b)
        # todo: mutation and probabilities of mutation and crossover
        population = new_population
    return population


def random_packing(n):
    return [ random.randint(0,1) for _ in range(n) ]


def value_knapsack(knapsack, packing ):
    """Function calculating knapsack value"""
    value = 0
    weight = 0
    for i,item in enumerate(knapsack['items']):
        value += item['value'] if packing[i] == 1 else 0
        weight += item['weight'] if packing[i] == 1 else 0
        if weight > knapsack['capacity']:
            return 0
    return value


def main():
    knapsack = {
        "capacity": 10,
        "items":[{"weight":1,"value":2}, {"weight":2,"value":3},
                 {"weight":5,"value":6}, {"weight":4,"value":10},
                 # {"weight": 3, "value": 2}, {"weight": 32, "value": 3},
                 # {"weight": 4, "value": 6}, {"weight": 24, "value": 10},
                 # {"weight": 50, "value": 52}, {"weight": 20, "value": 3},
                 # {"weight": 60, "value": 46}, {"weight": 34, "value": 10},
                 # {"weight": 70, "value": 21}, {"weight": 22, "value": 3},
                 # {"weight": 80, "value": 66}, {"weight": 14, "value": 100},
                 # {"weight": 90, "value": 2}, {"weight": 20, "value": 3},
                 # {"weight": 100, "value": 6}, {"weight": 19, "value": 10},
                 # {"weight": 11, "value": 2}, {"weight": 18, "value": 3},
                 # {"weight": 120, "value": 6}, {"weight": 17, "value": 10},
                 # {"weight": 13, "value": 2}, {"weight": 16, "value": 3},
                 # {"weight": 14, "value": 6}, {"weight": 150, "value": 10},
                 ]
    }
    genetic_algorithm(lambda x: value_knapsack(knapsack, x ),
                      lambda: random_packing(len(knapsack['items'])),
                      50,
                      100,
                      60,
                      1)


if __name__ == "__main__":
    main()