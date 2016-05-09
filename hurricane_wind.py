import numpy as np
import scipy as sp
import scipy.integrate as integ
import matplotlib.pyplot as plt

def coriolis_param(latitude):
    #Given a latitude, returns the coriolis frequency

    omega = 7.2921e-5   #rotational frequency of the earth
    return 2*omega*np.sin(latitude*2*np.pi/360)

def get_chi(v_ref):
    #Return chi given v, using w_cool = 2 mm/s and the piecewise linear
    #fit given for C_d in Chavas et al.
    if(v_ref < 6):
        return .308
    elif (v_ref < 35.4):
        return .02905*v_ref + .1307
    else:
        return 1.2

def get_alpha(v_max):
    #return the quadratic fit on C_k/C_d given in Chavas et al.
    return .00055*v_max**2 - .0259*v_max + .763

def solve_r_max(f, r_ref, v_ref, v_max):
    #Finds the radius of maximum wind speed based on a reference
    #radius and speed r_ref and v_ref, the max wind speed v_max, 
    #and the coriolis parameter f.
    #This relies on the assumption that C_k/C_d = 1
    
    alpha = get_alpha(v_max)
    r_guess = r_ref/2.0
    r_step = r_guess
    m_ref = r_ref*v_ref + .5*f*r_ref**2
    while (r_step > .01):  #this tolerance is more precise than necessary
        m_max = r_guess*v_max + .5*f*r_guess**2
        test = ((2*(r_ref/r_guess)**2)/(2 - alpha + alpha*(r_ref/r_guess)**2))
        test = m_max*(test)**(2-alpha)
        r_step = r_step/2
        if (test > m_ref):
            r_guess = r_guess - r_step
        else:
            r_guess = r_guess + r_step
    print 'r_guess =' 
    print r_guess
    return r_guess

def integ_m_out(f, r_0, r_a, res):
    #Based on outer model parameters, integrates the solution
    #for the convection-free outer regime of the hurricane.

    def M_prime(M,t):
        v = (M-.5*f*t**2)/t
        chi = get_chi(v)
        return (chi*(t*v)**2)/(r_0**2 - t**2)

    delta_t = (r_0 - r_a)/res
    y0 = .5*f*(r_0 - delta_t)**2 + f*(delta_t)*(r_0 - delta_t)
    t = np.linspace(r_0, r_a, res + 1)
    M = [.5*f*r_0**2]
    M.extend(integ.odeint(M_prime, y0, t[1:], printmessg=False))
    v = np.empty(( len(M) ))
    for i, c in enumerate(M):
        v[i] = (c - .5*f*(t[i]**2))/t[i]
    
    return v

def solve_r_0(f, r_a, v_a, r_guess, res):
    #Use a shooting method to find an appropriate value for r_0
    TOLERANCE = 1e-7
    r_0 = r_guess
    v_bound = 0
    step = r_guess/2
    while np.abs(v_a - v_bound) > TOLERANCE:
        v_bound = integ_m_out(f, r_0, r_a, res)[-1]
        r_0 = r_0
        if(v_a - v_bound < 0):
            r_0 = r_0 - step
            step = step/2
        else:
            r_0 = r_0 + step
    return r_0

def eval_v_in(f, v_max, r_max, r_in):
    #Evaluates the azimuthal wind velocity in the inner convection-dominated
    # region of the hurricane.
    alpha = get_alpha(v_max)
    M_max = v_max*r_max + .5*f*r_max**2
    M_in = M_max*( ((r_in/r_max)**2)/ \
                   (2 - alpha + alpha*(r_in/r_max)**2) )**(2-alpha)
    v_in = (M_in - .5*f*r_in**2)/r_in
    return v_in



def solve_hurricane(latitude, r_ref, v_ref, v_max, res, plotTitle='test'):
    """Solves for r_0, v_a, r_a, and r_max, the four unknown parameters
    of the model.  The parameters taken here are used throughout, so here
    is an explanation of each:
    latitude: Used to calculate the coriolis parameter.
    r_ref, v_ref: A reference velocity and radius. MUST COME FROM INNER REGIME.
    v_max: maximum wind speed.  r_max is the radius at which v_max occurs.
    chi: A parameter with dimensions s/m.  Drag coefficient C_d divided by rate
    of ekman transport.
    res: resolution used for integrating the outer regime solution."""
#    def slope_dr(chi, f, r_a, v_a, r_0):
        #Finds the derivative of the difference in slope between
        #the outer and inner regions of the hurricane.
        #Should be analytically correct, but may contain mistakes.
#        pvpr = chi*((r_a*v_a)**2)/((r_0**2 - r_a**2)*r_a) - f - v_a/r_a
#        slope_dr = chi*2*(r_a**3)*(v_a**2)/(r_0**2 - r_a**2)
#        temp_val = 2*r_a*v_a**2 + 2*pvpr*v_a*r_a**2
#        slope_dr = (slope_dr + temp_val)/(r_0**2 - r_a**2)
#        temp_val = (v_a + .5*f*r_a)*(3*(r_a/r_max)**2 + 1)/((r_a/r_max)**2 + 1)
#        temp_val = temp_val - (v_a + r_a*pvpr + f*r_a)*2
#        temp_val = temp_val/(r_a*((r_a/r_max)**2 + 1))
#        return slope_dr + temp_val
    #Initialize variables.
    TOLERANCE = 1e-8
    f = coriolis_param(latitude)
    r_max = solve_r_max(f, r_ref, v_ref, v_max)
    r_a = 2*r_max
    r_0 = 15*r_max
    step = r_a
    slope_diff = 1
    step_factor = 1
    while(step > .01):
        #This while loop tests for continuity in the derivative
        #at the boundary of the inner and outer regimes.
        v_a = eval_v_in(f, v_max, r_max, r_a)
        r_0 = solve_r_0(f, r_a, v_a, r_0, res)
        chi = get_chi(v_a)
        slope_diff = -(v_a + .5*f*r_a)*2/((r_a/r_max)**2 + 1)
        slope_diff = slope_diff + (chi*(r_a*v_a)**2)/(r_0**2 - r_a**2)
        
        if (slope_diff > 0):
            step = step/2
            step_factor = .5
            r_a = r_a - step
        else:
            step = step*step_factor
            r_a = r_a + step
        print 'r_a', r_a, slope_diff

    plot_velocity(f, r_0, r_a, r_max, v_max, 1000, plotTitle)
    return r_0, r_a, r_max, v_max

def plot_velocity(f, r_0, r_a, r_max, v_max, res, title, 
                  r_vals=None, v_vals=None):
    """Plots the full model of the hurricane. r_vals and v_vals are plotted
    as black dots on the graph, intended to represent real-world observations"""

    #Initialize variables
    r_in = np.linspace(0, r_a, res)
    r_out = np.linspace(r_0, r_a, res + 1)
    v_in = []

    #evaluate velocities
    for c in r_in:
        v_in.append(eval_v_in(f, v_max, r_max, c))
    v_out = integ_m_out(f, r_0, r_a, res)
    
    #Convert velocities to knots and radii to nautical miles
    v_in = np.array(v_in)/.514
    v_out = np.array(v_out)/.514
    r_in = r_in/1852
    r_out = r_out/1852
    
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
