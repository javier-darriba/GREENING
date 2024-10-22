import logging
import math
from datetime import datetime
import numpy as np
import multiprocessing as mp
from functools import partial

def process_init(seed, nprocess, lock):
    with lock:
        np.random.seed(nprocess.value + seed)
        nprocess.value += 1

def particle_update(fn_args, global_best_position, particle):
    np.random.seed(particle.seed)
    # UPDATE POSITION
    # Stickiness
    stk = 1.0 - particle.current_life / particle.max_life

    # Flipping probability
    p = particle.i_m * (1 - stk) + particle.i_p * abs(particle.best_position.astype(int) - particle.position.astype(
        int)) + particle.i_g * abs(global_best_position.astype(int) - particle.position.astype(int))

    # Update position and value
    rand_flip = np.random.rand(len(global_best_position)) < p
    np.invert(particle.position, where=rand_flip, out=particle.position)

    # Update current life and flip the bits that exceed max_life
    particle.current_life += 1
    particle.current_life[np.logical_or(
        rand_flip, particle.current_life >= particle.max_life)] = 0

    # UPDATE VALUE
    # Call fitness function
    particle.value = particle.fitness(particle.position, fn_args)
    if math.isnan(particle.value) or particle.value is None:
        logging.error("The fitness function value is NaN or None")

    if particle.best_value is None or particle.value < particle.best_value:
        particle.best_value = particle.value
        particle.best_position = particle.position

    particle.seed += 1

    return particle


def swarm_update(fn_args, swarm, pool):
    # Get best value
    for particle in swarm.particles:
        if swarm.best_value is None or particle.value < swarm.best_value:
            swarm.best_value = particle.value
            swarm.best_position = particle.position
    # Update particles
    swarm.particles = pool.map(partial(
        particle_update, fn_args, swarm.best_position), swarm.particles, chunksize=10)

    return swarm


class Particle:
    def __init__(self, num_features, max_life, fitness, i_m, i_p, i_g):
        self.position = np.random.choice([True, False], size=num_features)
        self.best_position = self.position
        self.current_life = np.zeros(num_features)
        self.max_life = max_life
        self.value = None
        self.best_value = None
        self.fitness = fitness
        self.seed = np.random.randint(9999)
        self.i_m = i_m
        self.i_p = i_p
        self.i_g = i_g

    def __str__(self):
        return "position: " + str(self.position) + "\nbest_position: " + str(self.best_position) + "\ncurrent_life: " + str(self.current_life) + "\nmax_life: " + str(self.max_life) + "\nvalue: " + str(self.value) + "\nbest_value: " + str(self.best_value) + "\nfitness: " + str(self.fitness)

    def update_value(self, fn_args):
        # Call fitness function
        self.value = self.fitness(self.position, fn_args)
        if math.isnan(self.value) or self.value is None:
            logging.error("The fitness function value is NaN or None")

        if self.best_value is None or self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position

        return self.best_value, self.best_position

    def update_position(self, global_best_position):
        # Stickiness
        stk = 1.0 - self.current_life / self.max_life

        # Flipping probability
        p = self.i_m * (1 - stk) + self.i_p * abs(self.best_position.astype(int) -
                                                  self.position.astype(int)) + self.i_g * abs(global_best_position.astype(int) - self.position.astype(int))

        # Update position and value
        rand_flip = np.random.rand(len(global_best_position)) < p
        np.invert(self.position, where=rand_flip, out=self.position)

        # Update current life and flip the bits that exceed max_life
        self.current_life += 1
        self.current_life[np.logical_or(
            rand_flip, self.current_life >= self.max_life)] = 0



# Swarm
class Swarm:
    def __init__(self, swarm_size, num_features, i_m, i_p, i_g, fitness, max_life):
        if i_m + i_p + i_g != 1:
            logging.error("The sum i_m, i_p and i_g must be exactly 1")
        self.best_position = None
        self.best_value = None
        self.particles = []
        for i in range(swarm_size):
            self.particles.append(
                Particle(num_features, max_life, fitness, i_m, i_p, i_g))

    def update(self, fn_args):
        # Sequential
        for particle in self.particles:
            value, position = particle.update_value(fn_args)

            if self.best_value is None or value < self.best_value:
                self.best_value = value
                self.best_position = position

            particle.update_position(self.best_position)




def sbpso(seed, nprocess, lock, num_features, fn, trace=True, minimize=True, report=10, max_it=1000, swarm_size=-1, i_m=0.1154, i_p=0.4423, i_g=0.4423, max_life=40, max_it_stagnate=math.inf, **fn_args):

    # Default swarm size
    if swarm_size <= 0:
        swarm_size = 10+2*math.sqrt(len(num_features))

    # Minimize or maximize fitness function
    func = fn if minimize else -fn
    func = fn

    # Initialize swarm
    swarm = Swarm(swarm_size, num_features, i_m, i_p, i_g, func, max_life)

    # Open parallel pool
    with mp.Pool(initializer=process_init, initargs=(seed, nprocess, lock)) as pool:
        # Evolve swarm
        time_start = datetime.now()
        it_stagnate = it = 0
        while True:
            previous_best_value = swarm.best_value

            # Not parallel
            # swarm.update(fn_args)

            # Parallel
            swarm = swarm_update(fn_args, swarm, pool)

            it += 1

            if swarm.best_value == previous_best_value:
                it_stagnate += 1
            else:
                it_stagnate = 0

            # Log updated results info
            if trace and it % report == 0:
                logging.info("It: " + str(it) + " | Value: " +
                             str(swarm.best_value) + " | Features: " +
                             str(sum(swarm.best_position)) + " Mean time per iteration: " +
                             str((datetime.now() - time_start)/it))
                
                
                with open("xxxxxxxx.txt", "a") as myfile:
                    myfile.write("\nIt: " + str(it) + " | Value: " +
                             str(swarm.best_value) + " | Features: " +
                             str(sum(swarm.best_position)) + " Mean time per iteration: " +
                             str((datetime.now() - time_start)/it))

            # Check termination condition
            if it >= max_it or it_stagnate >= max_it_stagnate:
                break

    # Return results
    return swarm.best_position, swarm.best_value, it, (datetime.now() - time_start)/it
