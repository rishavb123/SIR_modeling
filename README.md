# CX4230-MiniProject1

a)
Team Members: Rishav Bhagat and Ethan Povlot

Team Number: 7

b) 
The report is in the file `Report.pdf`

* 1.1 is in pages 1,2, and the very top of 3 (sections `SIR Model` and `Fixed Point Analysis`)
* 1.2 is on page 3 (sections `Simulation Results` and `Analysis`)
* 1.3 is on page 3 (section `Stop T Heatmaps`)
* 1.4 is on pages 4,5, and top of page 6 (sections `Vaccinations`, `NEAT`, `NEAT Results`, `Analysis`, `Designed Policy Results`)
* 1.5 is on page 6 and 7 (section `Designed Policy Results`)

c)
* All the simulation code (including the runner) is in `simulation.py`
* `parameter_analysis.py` contains the code to run the simulation (with no vaccination) over tao within [0,4] and kappa within [1,5] and generate a heatmap of the stopping (convergence times) for analysis.
* `vaccination_polices.py` contains some options for the vaccination policies and a way to handle hyper-parameters of vaccination policies.
* `neat_policy.py` runs the NEAT algorithm to create a neural network vaccination policy that will automatically (through Neuro-Evolution) design a policy that maximizes the amount of susceptible people left at the converged solution.
* `vaccination_tester.py` handles testing a vaccination policy by running the simulation multiple times with it, finding optimal hyper-parameters, and then plotting the resulting simulation using the optimal vaccination hyper-parameters.