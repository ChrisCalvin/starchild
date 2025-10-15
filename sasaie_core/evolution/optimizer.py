"""
Defines the EvolutionaryOptimizer for evolving RG-Morphisms.
"""

from typing import List, Literal
import random
import torch
from copy import deepcopy

from sasaie_core.evolution.morphisms import RGMorphism
from sasaie_core.evolution.registry import GraphComponentRegistry
from sasaie_core.planning.efe import EFECalculator

class EvolutionaryOptimizer:
    """
    Optimizes a population of RG-Morphisms using evolutionary algorithms.
    """

    def __init__(self, registry: GraphComponentRegistry, efe_calculator: EFECalculator):
        """
        Initializes the EvolutionaryOptimizer.

        Args:
            registry: The graph component registry.
            efe_calculator: The EFE calculator.
        """
        self.registry = registry
        self.efe_calculator = efe_calculator

    def _perform_uniform_crossover(self, parent1: RGMorphism, parent2: RGMorphism) -> (RGMorphism, RGMorphism):
        """
        Performs uniform crossover on the state dicts of two morphisms.

        Args:
            parent1: The first parent RGMorphism.
            parent2: The second parent RGMorphism.

        Returns:
            A tuple containing two offspring RGMorphisms.
        """
        offspring1 = parent1.clone(copy_weights=False)
        offspring2 = parent2.clone(copy_weights=False)
        p1_state_dict, p2_state_dict = parent1.state_dict(), parent2.state_dict()
        o1_state_dict, o2_state_dict = offspring1.state_dict(), offspring2.state_dict()

        for key in p1_state_dict:
            if random.random() < 0.5:
                o1_state_dict[key] = deepcopy(p1_state_dict[key])
                o2_state_dict[key] = deepcopy(p2_state_dict[key])
            else:
                o1_state_dict[key] = deepcopy(p2_state_dict[key])
                o2_state_dict[key] = deepcopy(p1_state_dict[key])
        
        offspring1.load_state_dict(o1_state_dict)
        offspring2.load_state_dict(o2_state_dict)
        return offspring1, offspring2

    def _perform_sbx_crossover(self, parent1: RGMorphism, parent2: RGMorphism, eta: float = 2.0) -> (RGMorphism, RGMorphism):
        """
        Performs Simulated Binary Crossover (SBX).

        Args:
            parent1: The first parent RGMorphism.
            parent2: The second parent RGMorphism.
            eta: The distribution index for SBX.

        Returns:
            A tuple containing two offspring RGMorphisms.
        """
        offspring1 = parent1.clone(copy_weights=False)
        offspring2 = parent2.clone(copy_weights=False)
        p1_state_dict, p2_state_dict = parent1.state_dict(), parent2.state_dict()
        o1_state_dict, o2_state_dict = offspring1.state_dict(), offspring2.state_dict()

        for key in p1_state_dict:
            if p1_state_dict[key].is_floating_point():
                p1_val, p2_val = p1_state_dict[key], p2_state_dict[key]
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                o1_state_dict[key] = 0.5 * ((1 + beta) * p1_val + (1 - beta) * p2_val)
                o2_state_dict[key] = 0.5 * ((1 - beta) * p1_val + (1 + beta) * p2_val)
            else: # For non-floating point params, do uniform crossover
                if random.random() < 0.5:
                    o1_state_dict[key] = deepcopy(p1_state_dict[key])
                    o2_state_dict[key] = deepcopy(p2_state_dict[key])
                else:
                    o1_state_dict[key] = deepcopy(p2_state_dict[key])
                    o2_state_dict[key] = deepcopy(p1_state_dict[key])

        offspring1.load_state_dict(o1_state_dict)
        offspring2.load_state_dict(o2_state_dict)
        return offspring1, offspring2

    def _perform_gaussian_mutation(self, individual: RGMorphism, mutation_strength: float) -> RGMorphism:
        """
        Performs Gaussian mutation on an individual RG-Morphism.

        Args:
            individual: The RGMorphism to mutate.
            mutation_strength: The standard deviation of the Gaussian noise.

        Returns:
            The mutated RGMorphism.
        """
        mutated_individual = individual.clone(copy_weights=True)
        mutated_state_dict = mutated_individual.state_dict()
        for key in mutated_state_dict:
            if mutated_state_dict[key].is_floating_point():
                noise = torch.randn_like(mutated_state_dict[key]) * mutation_strength
                mutated_state_dict[key] += noise
        mutated_individual.load_state_dict(mutated_state_dict)
        return mutated_individual

    def _perform_polynomial_mutation(self, individual: RGMorphism, eta: float = 20.0) -> RGMorphism:
        """
        Performs polynomial mutation on an individual RG-Morphism.

        Args:
            individual: The RGMorphism to mutate.
            eta: The distribution index for polynomial mutation.

        Returns:
            The mutated RGMorphism.
        """
        mutated_individual = individual.clone(copy_weights=True)
        mutated_state_dict = mutated_individual.state_dict()
        for key in mutated_state_dict:
            if mutated_state_dict[key].is_floating_point():
                param = mutated_state_dict[key]
                u = torch.rand_like(param)
                delta = torch.zeros_like(param)
                
                lt_mask = u < 0.5
                pow_val = 1.0 / (eta + 1.0)
                
                # Calculate delta for all elements first
                delta_all = torch.zeros_like(param)

                # For u < 0.5
                lt_mask = u < 0.5
                if lt_mask.any(): # Only compute if mask is not empty
                    val1 = 2.0 * u[lt_mask]
                    delta_all[lt_mask] = torch.pow(val1, pow_val) - 1.0

                # For u >= 0.5
                gt_mask = ~lt_mask
                if gt_mask.any(): # Only compute if mask is not empty
                    val2 = 2.0 * (1.0 - u[gt_mask])
                    delta_all[gt_mask] = 1.0 - torch.pow(val2, pow_val)
                
                mutated_state_dict[key] += delta_all
                
                mutated_state_dict[key] += delta
        
        mutated_individual.load_state_dict(mutated_state_dict)
        return mutated_individual

    def evolve(
        self,
        population: List[RGMorphism],
        fitness_scores: List[float],
        elitism_count: int = 2,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        crossover_type: Literal['uniform', 'sbx'] = 'sbx',
        mutation_type: Literal['gaussian', 'polynomial'] = 'polynomial',
        mutation_strength: float = 0.1,
        eta: float = 20.0
    ) -> List[RGMorphism]:
        """
        Evolves the population of RG-Morphisms based on their fitness scores.

        Args:
            population: The current population of RG-Morphisms.
            fitness_scores: The fitness scores of the morphisms in the population.
            elitism_count: The number of best individuals to carry over to the next generation.
            crossover_rate: The probability of performing crossover.
            mutation_rate: The probability of performing mutation on an individual.
            crossover_type: The type of crossover to use ('uniform' or 'sbx').
            mutation_type: The type of mutation to use ('gaussian' or 'polynomial').
            mutation_strength: The strength of Gaussian mutation.
            eta: The eta parameter for SBX crossover and polynomial mutation.

        Returns:
            A new, evolved population of RG-Morphisms.
        """
        print(f"Evolving RG-Morphisms using {crossover_type} crossover and {mutation_type} mutation...")
        new_population = []
        population_size = len(population)
        
        # --- Elitism ---
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        elites = [ind.clone(copy_weights=True) for ind, score in sorted_population[:elitism_count]]
        new_population.extend(elites)

        # --- Selection ---
        selected_parents = []
        for _ in range(population_size - elitism_count):
            tournament_contenders = random.sample(sorted_population, 2)
            winner = max(tournament_contenders, key=lambda x: x[1])[0]
            selected_parents.append(winner)
        
        # --- Crossover ---
        offspring_population = []
        random.shuffle(selected_parents)
        for i in range(0, len(selected_parents), 2):
            if i + 1 < len(selected_parents) and random.random() < crossover_rate:
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                if crossover_type == 'sbx':
                    offspring1, offspring2 = self._perform_sbx_crossover(parent1, parent2, eta=eta)
                else: # uniform
                    offspring1, offspring2 = self._perform_uniform_crossover(parent1, parent2)
                offspring_population.extend([offspring1, offspring2])
            else: # No crossover, just clone parents
                offspring_population.append(selected_parents[i].clone(copy_weights=True))
                if i + 1 < len(selected_parents):
                    offspring_population.append(selected_parents[i+1].clone(copy_weights=True))

        # --- Mutation ---
        mutated_offspring = []
        for individual in offspring_population:
            if random.random() < mutation_rate:
                if mutation_type == 'polynomial':
                    mutated_ind = self._perform_polynomial_mutation(individual, eta=eta)
                else: # gaussian
                    mutated_ind = self._perform_gaussian_mutation(individual, mutation_strength=mutation_strength)
                mutated_offspring.append(mutated_ind)
            else:
                mutated_offspring.append(individual)

        new_population.extend(mutated_offspring)
        # Ensure population size is maintained
        return new_population[:population_size]
