import copy

import pyomo.environ as pyo


#########################################################
# Worker Optimization
#########################################################
model = pyo.ConcreteModel()

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Shifts of each day
shifts = ['morning', 'evening', 'night']  # 3 shifts of 8 hours
days_shifts = {day: shifts for day in days}  # dict with day as key and list of its shifts as value

# Worker ids
workers = ['W' + str(i) for i in range(1, 11)]  # 10 workers available, more than needed
model.worker = pyo.Set(initialize=workers, doc="List of worker ids.")

# Will worker be working on a given day and given shift? default is NO.
model.works = pyo.Var(((worker, day, shift) for worker in workers for day in days for shift in days_shifts[day]), within=pyo.Binary, initialize=0)

# Is the worker needed at all?
model.needed = pyo.Var(workers, within=pyo.Binary, initialize=0)

model.constraints = pyo.ConstraintList()

# TODO: number of workers should be dynamic based on estimated number of clients
# TODO: add a forecasting model input to define constraints more dynamically
# TODO: package as a class and parametrize
# All shifts must be assigned
for day in days:
    for shift in days_shifts[day]:
        if day in days[:-1] and shift in ['morning', 'evening']:
            # Two workers during morning and evening shifts on Mon-Sat inclusive
            model.constraints.add(sum(model.works[worker, day, shift] for worker in workers) == 2)
        else:
            # One worker during every shift on Sunday and Mon-Sat night shifts
            model.constraints.add(sum(model.works[worker, day, shift] for worker in workers) == 1)

# A worker shouldn't work more than 40 hours per week
for worker in workers:
    # for each worker, the number of shifts and days they work
    model.constraints.add(sum(8 * model.works[worker, day, shift] for day in days for shift in days_shifts[day]) <= 40)
    
# Rest between two shifts is 12 hours (i.e., at least two shifts)
for worker in workers:
    for j in range(len(days)):
        # no more than a single shift in a given day
        model.constraints.add(sum(model.works[worker, days[j], shift] for shift in days_shifts[days[j]]) <= 1)
        
        # if evening, rest during night and next morning (sum(evening + night + next morning <= 1))        
        model.constraints.add(sum([
            model.works[worker, days[j], "evening"],
            model.works[worker, days[j], "night"],
            model.works[worker, days[(j + 1) % 7], "morning"]
        ]) <= 1)
        
        # if night (sum(night + next morning + next evening <= 1))
        model.constraints.add(sum([
            model.works[worker, days[j], "night"],
            model.works[worker, days[(j + 1) % 7], "morning"],
            model.works[worker, days[(j + 1) % 7], "evening"]
        ]) <= 1)     

# Constraint to remove workers that are not necessary at all
# If a worker has any shift - they should be 1, else they should be 0, how can we achieve this?

# Objective: minimize number of engaged workers
def objective(model):
    c = len(workers)
    return sum(c * model.needed[worker] for worker in workers)

# add objective function to the model. rule (pass function) or expr (pass expression directly)
model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)

opt = pyo.SolverFactory('cbc')  # choose a solver
results = opt.solve(model)  # solve the model with the selected solver

workers_needed = [v.value for k, v in model.needed.items()]
schedule = dict()
for k, v in model.works.items():
    if v.value:
        schedule[k] = True        
        


#########################################################
# Food Optimization
#########################################################

nutrients = {
    "calories": {"min": 1800, "max": 2200},
    "protein":{"min": 91, "max": float('inf')},
    "fat": {"min": 0, "max": 65},
    "sodium": {"min": 0, "max": 1779}
}

foods = {
    "hamburger": {"cost": 2.49, "volume": 1, "calories": 410, "protein": 24, "fat": 26, "sodium": 730},
    "chicken": {"cost": 2.89, "volume": 2, "calories": 420, "protein": 32, "fat": 10, "sodium": 1190},
    "hot dog": {"cost": 1.50, "volume": 3, "calories": 560, "protein": 20, "fat": 32, "sodium": 1800},
    "fries": {"cost": 1.89, "volume": 2, "calories": 380, "protein": 4, "fat": 19, "sodium": 270},
    "macaroni": {"cost": 2.09, "volume": 1, "calories": 320, "protein": 12, "fat": 10, "sodium": 930},
    "pizza": {"cost": 1.99, "volume": 2, "calories": 320, "protein": 15, "fat": 12, "sodium": 820},
    "salad": {"cost": 2.49, "volume": 3, "calories": 320, "protein": 31, "fat": 12, "sodium": 1230},
    "milk": {"cost": 0.89, "volume": 2, "calories": 100, "protein": 8, "fat": 2.5, "sodium": 125},
    "ice cream": {"cost": 1.59, "volume": 1, "calories": 330, "protein": 8, "fat": 10, "sodium": 180}
}

volume_limit = 75.0

model = pyo.ConcreteModel()
model.nutrients = pyo.Set(initialize=nutrients.keys(), doc="Nutrients")
model.foods = pyo.Set(initialize=foods.keys(), doc="Foods")

model.nutrients_min = pyo.Param(model.nutrients, initialize={k:v["min"] for k, v in nutrients.items()}, within=pyo.NonNegativeReals, default=0.0)
model.nutrients_max = pyo.Param(model.nutrients, initialize={k:v["max"] for k, v in nutrients.items()}, within=pyo.NonNegativeReals, default=float('inf'))

# Food cost parameters
model.foods_costs = pyo.Param(model.foods, initialize={k:v["cost"] for k, v in foods.items()}, within=pyo.PositiveReals)
# Volume per serving of food
model.vol = pyo.Param(model.foods, initialize={k:v["volume"] for k, v in foods.items()}, within=pyo.PositiveReals)
# Maximum volume of food consumed
model.vol_max = pyo.Param(initialize=volume_limit, within=pyo.PositiveReals)

# Food nutrient parameters
foods_nutrients = dict()
for f, v in foods.items():
    for n in nutrients.keys():
        foods_nutrients[(f, n)] = v[n]
model.foods_nutrients = pyo.Param(model.foods, model.nutrients, initialize=foods_nutrients, within=pyo.NonNegativeReals)
# Decision variable, which food to purchase
model.foods_to_buy = pyo.Var(model.foods, domain=pyo.NonNegativeIntegers)

# Objective: minimize costs of purchased food
def cost_rule(model):
    return sum(model.foods_costs[f] * model.foods_to_buy[f] for f in model.foods)
model.cost = pyo.Objective(sense=pyo.minimize, rule=cost_rule)

# Nutrition constraints
def nutrient_limit_rule(model, n):
    nutritional_value = sum(model.foods_nutrients[f, n] * model.foods_to_buy[f] for f in model.foods)
    return model.nutrients_min[n] <= nutritional_value <= model.nutrients_max[n]
model.nutrient_limit = pyo.Constraint(model.nutrients, rule=nutrient_limit_rule)

# Limit the volume of food consumed
def volume_rule(model):
    return sum(model.vol[f] * model.foods_to_buy[f] for f in model.foods) <= model.vol_max
model.volume_constraint = pyo.Constraint(rule=volume_rule)

# Solve
from pyomo.opt import SolverFactory
opt = SolverFactory("cbc")
results = opt.solve(model)
print(results)

for k, v in model.foods_to_buy.items():
    print(k, v.value)



