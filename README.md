# SIR Modeling

Repository Structure:
* The report is in the file `Report.pdf`
* All the simulation code (including the runner) is in `simulation.py`
* `parameter_analysis.py` contains the code to run the simulation (with no vaccination) over tao within [0,4] and kappa within [1,5] and generate a heatmap of the stopping (convergence times) for analysis.
* `vaccination_polices.py` contains some options for the vaccination policies and a way to handle hyper-parameters of vaccination policies.
* `neat_policy.py` runs the NEAT algorithm to create a neural network vaccination policy that will automatically (through Neuro-Evolution) design a policy that maximizes the amount of susceptible people left at the converged solution.
* `vaccination_tester.py` handles testing a vaccination policy by running the simulation multiple times with it, finding optimal hyper-parameters, and then plotting the resulting simulation using the optimal vaccination hyper-parameters.
* `init.bat` is a windows batch script to create the following directories (required to run the code): `results`, `results/sims`, `results/analysis`, `results/models`
* `clear_results.bat` is a windows batch script to clear all the results in `results/sims`