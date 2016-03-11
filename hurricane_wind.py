import numpy as np
import scipy as sp
import scipy.integrate as integ
import matplotlib.pyplot as plt

def coriolis_param(latitude):
    #Given a latitude, returns the coriolis frequency

    omega = 7.2921e-5   #rotational frequency of the earth
    return 2*omega*np.sin(latitude*2*np.pi/360)

def solve_r_max(f, r_ref, v_ref, v_max):
    #Finds the radius of maximum wind speed based on a reference
    #radius and speed r_ref and v_ref, the max wind speed v_max, 
    #and the coriolis parameter f.
    #This relies on the assumption that C_k/C_d = 1
    
    a = v_ref - .5*f*r_ref
    b = -2*r_ref*v_max
    c = v_ref*r_ref**2 + (f*r_ref**3)/2
    #When C_k/C_d = 1, r_max is the solution of a quadratic equation.
    r_max = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return r_max

def integ_m_out(f, chi, r_0, r_a, res):
    #Based on outer model parameters, integrates the solution
    #for the convection-free outer regime of the hurricane.

    M_prime = lambda M, t: chi*((M - .5*f*t**2)**2)/(r_0**2 - t**2)
    t = np.linspace(r_0, r_a, res)
    y0 = .5*f*r_0**2 + (.5/res)*(r_a - r_0)*chi*(f**2)*(r_0)**2
    M = [.5*f*r_0**2]
    M.extend(integ.odeint(M_prime, y0, t[1:], printmessg=False))
    v = []
    for i, c in enumerate(M):
        v.append((c - .5*f*(t[i]**2))/t[i])
    return v

def solve_r_0(f, r_a, v_a, chi, r_guess, res):
    #Use a shooting method to find an appropriate value for r_0
    TOLERANCE = 1e-7
    r_0 = r_guess
    v_bound = 0
    while np.abs(v_a - v_bound) > TOLERANCE:
        v_bound = integ_m_out(f, chi, r_0, r_a, res)[-1]
        #dvdr is the derivative with respect to r_a, not r_0.
        #I use this because I don't know the derivative with respect to r_0.
        dvdr = (r_a*v_bound**2)/(r_0**2 - r_a**2) - v_bound/r_a - f
        r_0 = r_0 - 10*(v_a - v_bound)/dvdr 
        print 'r_0', r_0, v_a - v_bound 
    return r_0

def eval_v_in(f, v_max, r_max, r_in):
    #Evaluates the azimuthal wind velocity in the inner convection-dominated
    # region of the hurricane.
    M_in = 2*(r_max*v_max + .5*f*r_max**2)/((r_max/r_in)**2 + 1)
    v_in = (M_in - .5*f*r_in**2)/r_in
    return v_in



def solve_hurricane(latitude, r_ref, v_ref, v_max, chi, res, plotTitle='test'):
    """Solves for r_0, v_a, r_a, and r_max, the four unknown parameters
    of the model.  The parameters taken here are used throughout, so here
    is an explanation of each:
    latitude: Used to calculate the coriolis parameter.
    r_ref, v_ref: A reference velocity and radius. MUST COME FROM INNER REGIME.
    v_max: maximum wind speed.  r_max is the radius at which v_max occurs.
    chi: A parameter with dimensions s/m.  Drag coefficient C_d divided by rate
    of ekman transport.
    res: resolution used for integrating the outer regime solution."""
    def slope_dr(chi, f, r_a, v_a, r_0):
        #Finds the derivative of the difference in slope between
        #the outer and inner regions of the hurricane.
        #Should be analytically correct, but may contain mistakes.
        pvpr = chi*((r_a*v_a)**2)/((r_0**2 - r_a**2)*r_a) - f - v_a/r_a
        slope_dr = chi*2*(r_a**3)*(v_a**2)/(r_0**2 - r_a**2)
        temp_val = 2*r_a*v_a**2 + 2*pvpr*v_a*r_a**2
        slope_dr = (slope_dr + temp_val)/(r_0**2 - r_a**2)
        temp_val = (v_a + .5*f*r_a)*(3*(r_a/r_max)**2 + 1)/((r_a/r_max)**2 + 1)
        temp_val = temp_val - (v_a + r_a*pvpr + f*r_a)*2
        temp_val = temp_val/(r_a*((r_a/r_max)**2 + 1))
        return slope_dr + temp_val
    #Initialize variables.
    TOLERANCE = 1e-5
    f = coriolis_param(latitude)
    r_max = solve_r_max(f, r_ref, v_ref, v_max)
    r_a = 3*r_max
    r_0 = 15*r_max
    slope_diff = 1
    
    while(np.abs(slope_diff) > TOLERANCE):
        #This while loop tests for continuity in the derivative
        #at the boundary of the inner and outer regimes.
        v_a = eval_v_in(f, v_max, r_max, r_a)
        r_0 = solve_r_0(f, r_a, v_a, chi, r_0, res)
        slope_diff = -(v_a + .5*f*r_a)*2/((r_a/r_max)**2 + 1)
        slope_diff = slope_diff + (chi*(r_a*v_a)**2)/(r_0**2 - r_a**2)
        
        #The factor of 1/2 in the second term is intended to avoid over
        #shooting caused by bad initial guesses.
        r_a = r_a - slope_diff/(slope_dr(chi, f, r_a, v_a, r_0)/2)
        print 'r_a', r_a, slope_diff

    plot_velocity(f, chi, r_0, r_a, r_max, v_max, 1000, plotTitle)
    return r_0, r_a, r_max, v_max

def plot_velocity(f, chi, r_0, r_a, r_max, v_max, res, title, 
                  r_vals=None, v_vals=None):
    """Plots the full model of the hurricane. r_vals and v_vals are plotted
    as black dots on the graph, intended to represent real-world observations"""

    #Initialize variables
    r_in = np.linspace(0, r_a, res)
    r_out = np.linspace(r_0, r_a, res)
    v_in = []

    #evaluate velocities
    for c in r_in:
        v_in.append(eval_v_in(f, v_max, r_max, c))
    v_out = integ_m_out(f, chi, r_0, r_a, res)
    
    #Convert velocities to knots and radii to nautical miles
    v_in = np.array(v_in)*1000/.514
    v_out = np.array(v_out)*1000/.514
    r_in = r_in/1.852
    r_out = r_out/1.852
    
    #Here we set up the plot.
    fig = plt.figure(1)
    plt.subplot(1,1,1)
    plt.plot(r_in, v_in, 'b-', r_out, v_out, 'r-')
    if r_vals:
        plt.plot(r_vals, v_vals, 'ko') 
    plt.xlabel("Nautical Miles from Center")
    plt.ylabel("Wind Speed in Knots")
    plt.title('Wind Speed Profile for ' + title)
    fig.savefig(title + '.pdf')
    plt.show()
