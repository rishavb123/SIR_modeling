import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

plt.style.use("dark_background")

def f(t, x, tao, kappa):
    s, i, r = x
    return [
        -tao * s * i,
        tao * s * i - i / kappa,
        i / kappa
    ]

def run_simulation(s0=0.99, i0=0.01, r0=0, tao=0.8, kappa=4, log=False):
    assert i0 + s0 + r0 == 1, "Initial conditions must sum to 1"
    
    x0 = [s0, i0, r0]
    
    def stopping_condition(t, x, tao, kappa):
        s, i, r = x
        return i - 1e-4
        
    stopping_condition.terminal = True
    
    start_t = 0
    end_t = 100
    N = 1000
    
    result = scipy.integrate.solve_ivp(f, (0, 100), x0, events=[stopping_condition], args=(tao, kappa), t_eval=np.linspace(start_t, end_t, N))
    
    if log:
        print(result)
    
    return result.t, result.y[0], result.y[1], result.y[2], result.t_events[0][0], result

def plot_sim_results(s0=0.99, i0=0.01, r0=0, tao=0.8, kappa=4, log=False):
    t, s, i, r, stop_t, sol = run_simulation(
        s0, i0, r0, tao, kappa, log
    )

    title = f"SIR Model s0={s0}, i0={i0}, r0={r0}, tao={tao}, kappa={kappa}"

    print("Stopping Condition at", stop_t)
    
    plt.plot(t, s, label="Susceptible %")
    plt.plot(t, i, label="Infected %")
    plt.plot(t, r, label="Recovered %")

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("proportion of population")
    
    plt.legend()
    plt.show()
    
    return sol


def main():
    sol = plot_sim_results()
    
    
if __name__ == "__main__":
    main()

