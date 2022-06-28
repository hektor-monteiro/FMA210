#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:25:38 2022

@author: hmonteiro
"""

import numpy as np

###############################################################################
# Constants
###############################################################################
class Constants:
     pass
 
cst = Constants()

cst.m_H         = 1.673532499e-27
cst.m_e         = 9.10938188e-31
cst.AU          = 1.4959787066e11
cst.G           = 6.673e-11
cst.c           = 2.99792458e08
cst.mu_0        = 4.*np.pi*1e-07
cst.epsilon_0   = 1/(cst.mu_0*cst.c**2)
cst.e_C         = 1.602176462e-19
cst.h           = 6.62606876e-34
cst.hbar        = cst.h/2./np.pi
cst.k_B         = 1.3806503e-23
cst.sigma       = 2*np.pi**5*cst.k_B**4/(15*cst.c**2*cst.h**3)
cst.a_rad       = 4*cst.sigma/cst.c
cst.a_rad_o3    = cst.a_rad/3
cst.four_ac_o3  = 4*cst.a_rad_o3*cst.c
cst.M_Sun       = 1.9891e30
cst.S_Sun       = 1.365e3
cst.L_Sun       = 4*np.pi*cst.AU**2*cst.S_Sun


###############################################################################
# Physics
###############################################################################

def Mean_Molecular_Weight(X, Y, Z):
#       General Description:
#       ====================
#           Calculate the mean molecular weight of the gas

    mu = 1/(2*X + 3*Y/4 + Z/2)           # Assume complete ionization, Eq. (10.16)
    
    return mu
    
def PTgradient(P0, P1, T0, T1):
    
#       General Description:
#       ====================
#           Compute the pressure gradient with respect to temperature to 
#           determine whether convection is required. Limit value of 
#           dlnPdlnT for output purposes.
    
    dlnPdlnT_val = ((T0 + T1)/(P0 + P1))*((P0 - P1)/(T0 - T1))
    
    if (dlnPdlnT_val > 99.9):
        dlnPdlnT_val = 99.9
        
    return dlnPdlnT_val

def Specific_Heat_Ratio():
#       General Description:
#       ====================
#           Compute the ratio C_P/C_V
    
    monatomic = 5/3.0
    gamma = monatomic
    
    return gamma


def Density(T, P, mu, step_size_condition):
    
#       General Description:
#       ====================
#           Density computes the density of the gas, assuming the ideal gas law 
#           and radiation pressure. A negative value for the density indicates 
#           that an error was detected in the routine

        # USE Zone_Quantities, ONLY : step_size_condition

        P_gas = P - cst.a_rad_o3*T**4                       # Eq. (10.20)
        
        if (P_gas <= 0 and T > 0):
            
            if (step_size_condition == 0):              # Do something desperate
                P_gas = P
            elif (step_size_condition == 1):
                P_gas = 0.001*P
            elif (step_size_condition == 2):
                P_gas = 0.0001*P

        
        if (T > 0 and P_gas > 0):
            rho = P_gas*mu*cst.m_H/(cst.k_B*T)                    # Eq. (10.11)
        else:
            rho = -1

        if (rho < 0):
            print('A negative density was computed!')
            print('Terminating calculation with: T= %8.3f P= %8.3f Pgas= %8.3f '%(T, P, P_gas))
            
        return rho


def Opacity(T, rho, X, Z):    
#       General Description:
#       ====================
#           Opacity computes an approximation of the Rosseland Mean Opacity, 
#           based on approximation formulae

    g_ff = 1                    # the free-free Gaunt factor is on the order of unity
    A_bf = 4.34e21
    A_ff = 3.68e18
    A_es = 0.02
    A_Hm = 7.9e-34

    tog_bf = 0.708*(rho*(1 + X))**0.2                       # Taken from Novotny (1973), p. 469

    kappa_bf = (A_bf/tog_bf)*Z*(1 + X)*rho/T**3.5           # Eq. (9.22)
    kappa_ff = A_ff*g_ff*(1 - Z)*(1 + X)*rho/T**3.5         # Eq. (9.23)
    kappa_es = A_es*(1 + X)                                 # Eq. (9.27)
    
    if ((T > 3000 and T < 6000) and (rho > 1E-10 and rho < 1E-5) and (Z > 0.001 and Z < 0.03)):
        kappa_Hminus = A_Hm*(Z/0.02)*np.sqrt(rho)*T**9      # Eq. (9.28)
    else:
        kappa_Hminus = 0

    kappa = kappa_bf + kappa_ff + kappa_es + kappa_Hminus
    
    return kappa

def Optical_Depth_Change(kappa, kappam, rho, rhom, r, rm):
#       General Description:
#       ====================
#           Compute the change in optical depth across the zone

    dtau = -(kappa*rho + kappam*rhom)*(r - rm)/2            # Eq. (9.15)
    return dtau

def Nuclear(T, rho, X, Z):
#       General Description:
#       ====================
#           Nuclear computes the nuclear energy generation rates for the 
#           proton-proton chains, the CNO cycle, and helium burning.

    fpp, f3a = 1, 1
    onethird = 1/3.0 
    twothirds = 2*onethird
    fourthirds = 4*onethird
    fivethirds = 5*onethird

    A_pp = 0.241                 # reaction rate coefficients
    A_CNO = 8.67e20
    A_He = 50.9      

    T6 = T*1.0E-06
    T8 = T*1.0E-08

#       PP chains (see Hansen and Kawaler, Eq. 6.65, 6.73, and 6.74)
    psipp = 1 + 1.412E8*(1/X - 1)*np.exp(-49.98*T6**(-onethird))
    Cpp = 1 + 0.0123*T6**onethird + 0.0109*T6**twothirds + 0.000938*T6
    eps_pp = A_pp*rho*X*X*fpp*psipp*Cpp*T6**(-twothirds)*np.exp(-33.80*T6**(-onethird))    # Eq. (10.46)

#       CNO cycle (Kippenhahn and Weigert, Eq. 18.65)
    XCNO = Z/2
    CCNO = 1 + 0.0027*T6**onethird - 0.00778*T6**twothirds - 0.000149*T6  
    eps_CNO = A_CNO*rho*X*XCNO*CCNO*T6**(-twothirds)*np.exp(-152.28*T6**(-onethird))       # Eq. (10.58)

#       Helium burning (Kippenhahn and Weigert, Eq. 18.67)
    Y = 1.0 - X - Z
    eps_He = A_He*rho**2*Y**3/T8**3*f3a*np.exp(-44.027/T8)                                 # Eq. (10.62)

#       Combined energy generation rate
    epsilon = eps_pp + eps_CNO + eps_He
    
    return epsilon


###############################################################################
# Stellar_Structure_Equations
###############################################################################

#   Hydrostatic Equilibrium
def dPdr(M_r, rho, r):
      return -cst.G*rho*M_r/r**2                # Eq. (10.6)

#   Mass Conservation
def dMdr(r, rho):
      return (4.0*np.pi*rho*r**2)               # Eq. (10.7)

#   Luminosity Gradient
def dLdr(r, rho, epsilon):
      return 4.0*np.pi*r**2*rho*epsilon        # Eq. (10.36)

#   Temperature Gradient
def dTdr(kappa, rho, T, L_r, r, mu, M_r, gamma, dlnPdlnT):
    
    gamma_ratio = gamma/(gamma - 1)
    if (dlnPdlnT > gamma_ratio):                                # radiation criterion,   Eq. (10.95)
        dTdr = -(kappa*rho/T**3)*(L_r/(4.0*np.pi*r**2))/cst.four_ac_o3    # radiation,   Eq. (10.68)
        rc_flag = "r"
    else:
        dTdr = -(1/gamma_ratio)*(mu*cst.m_H/cst.k_B)*(cst.G*M_r/r**2)     # adiabatic convection,  Eq. (10.89)
        rc_flag = "c"
    
    return dTdr, rc_flag


#  This subroutine returns the required derivatives for RUNGE, the
#  Runge-Kutta integration routine.
def STRUCT_EQS(f0,r,f,X,Y,Z,step_size_condition):
    

    dfdr=np.zeros(4, dtype=np.float64)
    P0, M_r0, L_r0, T0   = f0
    P, M_r, L_r, T   = f
        
    mu  = Mean_Molecular_Weight(X, Y, Z)
    rho = Density(T, P, mu, step_size_condition)
    
    if (rho < 0.):
        print('Density calculation error in FUNCTION Structure_Eqns')
        return dfdr
    else:
        dfdr[0] = dPdr(M_r, rho, r)
        
        dfdr[1] = dMdr(r, rho)
        
        epsilon  = Nuclear(T, rho, X, Z)
        dfdr[2] = dLdr(r, rho, epsilon)
        
        kappa    = Opacity(T, rho, X, Z)
        gamma    = Specific_Heat_Ratio()
        dlnPdlnT = PTgradient(P0, P, T0, T)
        dfdr[3],rc_flag = dTdr(kappa, rho, T, L_r, r, mu, M_r, gamma, dlnPdlnT)
        
        return dfdr


# Runge-kutta algorithm
def RUNGE(f_0, dfdr, r_0, deltar, X, Y, Z, step_size_condition):

      f_temp=np.zeros(4, dtype=np.float64)
      f_i=np.zeros(4, dtype=np.float64)

      k1 = deltar*dfdr
      
      df1 = STRUCT_EQS(f_0, r_0 + deltar/2, f_0 + k1/2, X, Y, Z, step_size_condition)
      k2 = deltar*df1

      df2 = STRUCT_EQS(f_0, r_0 + deltar/2, f_0 + k2/2, X, Y, Z, step_size_condition)
      k3 = deltar*df2

      df3 = STRUCT_EQS(f_0, r_0 + deltar, f_0 + k3, X, Y, Z, step_size_condition)
      k4 = deltar*df3

      f_i = f_0 + (k1/6 + k2/3 + k3/3 + k4/6)

      return f_i

###############################################################################
# Boundary conditions
###############################################################################

def Surface(i, Ms, Ls, M, L, r, X, Z, dr, P, T, rho, kappa, eps, step_size_condition):

#       General Description:
#       ====================
#           Estimate the temperature and pressure of the outermost zone from the zero boundary condition.
#           Electron scattering and H- ion contributions to the opacity are neglected for simplification.


    g_ff = 1                    # the free-free Gaunt factor is on the order of unity
    A_bf = 4.34E21
    A_ff = 3.68E18              # Bound-free and free-free coefficients
    maximum   = 1.0E-8          # Maximum change in Ms and Ls over surface zone
    j_max = 50
    
    Y  = 1.0 - X - Z
    mu = Mean_Molecular_Weight(X, Y, Z)
    gamma = Specific_Heat_Ratio()
    gamma_ratio = gamma/(gamma - 1)

    j = 0
    r1 = r[i-1] + dr
    
    while(j < j_max):        
        
#           Compute the temperature and pressure for the radiative boundary condition
        rc_flag = "r"
        T1 = cst.G*Ms*(mu*cst.m_H/(4.25*cst.k_B))*(1/r1 - 1/r[i-1])                # Eq. (L.2); radiative assumption
        
        if (i < 2):
            tog_bf = 0.01                                                    # Assume small value for surface
        else:
            tog_bf = 2.82*(rho[i-1]*(1 + X))**0.2                                 # Taken from Novotny (1973), p. 469
            
        Aop = (A_bf/tog_bf)*Z*(1+X) + A_ff*g_ff*(1-Z)*(1+X)                  # From Eq. (9.22) and (9.23)
        P1 = np.sqrt((1/4.25)*(16*np.pi/3)*(cst.G*Ms/Ls)*(cst.a_rad*cst.c*cst.k_B/(Aop*mu*cst.m_H)))*T1**4.25  # Eq. (L.1)

#           If the zone is convective, recompute the adiabatic temperature and pressure
        dlnPdlnT1 = PTgradient(P[i-1], P1, T[i-1], T1)
        
        if (dlnPdlnT1 < gamma_ratio and i > 2):
            rc_flag = "c"
            kPadiabatic = P[i-1]/T[i-1]**gamma_ratio
            T1 = cst.G*M[i]*(mu*cst.m_H/(cst.k_B*gamma_ratio))*(1/r1 - 1/r[i-1])       # Eq. (L.3)
            P1 = kPadiabatic*T**gamma_ratio                                 # Eq. (10.83)
            
#           Compute remaining surface quantities
        rho1 = Density(T1, P1, mu, step_size_condition)
        if (rho1 < 0):
            good_surface = False
            break

        kappa1   = Opacity(T1, rho1, X, Z)
        XCNO    = Z/2
        eps1 = Nuclear(T1, rho1, X, Z)
            
#           Test to be sure that variations in M_r and L_r are not too large
        M_r = M[i-1] + dMdr(r1, rho1)*dr
        L_r = L[i-1] + dLdr(r1, rho1, eps1)*dr
        
        if (np.abs((M[i] - M_r)/M[i]) < maximum and np.abs((L[i] - L_r)/L[i]) < maximum):
            good_surface = True
            
            # print(P1,T1,rho1,dlnPdlnT1,M_r,L_r,np.abs((M[i] - M_r)/M[i]))
            break
        
#           If changes in M_r and L_r were too large, repeat with one-half the step size
            
        j = j + 1
        if (j > j_max):
            print("Unable to converge in SUBROUTINE Surface --- Exiting")
            good_surface = False
            break

        dr = dr/2
        r1 = r[i-1] + dr
        

    if (good_surface == False):
        print("The last values obtained by SUBROUTINE Surface were: ")
        print("     M_r = %8.3f   dM_r/Ms = %8.3f"%(M_r, (M[i] - M_r)/M[i]))
        print("     L_r = %8.3f   dL_r/Ls = %8.3f"%(L_r, (L[i] - L_r)/L[i]))
        
    return M_r, L_r, r1, P1, T1, rho1, kappa1, eps1, dlnPdlnT1

def Core(M, L, P, T, X, Z, r):

#       General Description:
#       ====================
#           This routine extrapolates from the inner-most zone to obtain estimates of core conditions in the star

    converged = 1.0E-8
    i_max = 50

    rho_0     = M/(4*np.pi/3*r**3)             # Average density of the central ball
    P_0       = P + (2*np.pi/3)*cst.G*rho_0**2*r**2    # Central pressure, Eq. (L.4)
    epsilon_0 = L/M                           # Average energy generation rate of the central ball
        
#       Find core temperature by Newton-Raphson method (including radiation pressure)
    Y   = 1.0 - X - Z
    mu  = Mean_Molecular_Weight(X, Y, Z)

    if (rho_0 > 0):
        i = 0
        T_0 = T
        good_T = True
        while(i < i_max):
            f_T0 = rho_0*cst.k_B*T/(mu*cst.m_H) + cst.a_rad_o3*T**4 - P_0
            dfdT_0 = rho_0*cst.k_B/(mu*cst.m_H) + 4*cst.a_rad_o3*T**3
            dT = f_T0/dfdT_0
            if (np.abs(dT/T_0) < converged):
                break
            T_0 = T_0 + dT
            if (i > i_max):
                print("Unable to converge on core temperature in SUBROUTINE Core --- Exiting")
                good_T = False
                break
            i = i + 1
    else:
        T_0 = -T
        good_T = False


    if (good_T):
        kappa_0  = Opacity(T_0, rho_0, X, Z)
        dlnPdlnT = PTgradient(P, P_0, T, T_0)
        gamma    = Specific_Heat_Ratio()
    else:
        kappa_0  = -99.9
        dlnPdlnT = -99.9
        
    return rho_0, epsilon_0, P_0, T_0, kappa_0, dlnPdlnT

def f(T):
    f = rho_0*k*T/(mu*m_H) + a_o3*T**4 - P_0    # f = Ideal Gas Law + Radiation Pressure - core P = 0
    return f

def dfdT(T):
    dfdT = rho_0*k/(mu*m_H) + 4*a_o3*T**3       # df/dT
    return dfdT

###############################################################################
# running the model
###############################################################################

def StatStar(Msolar,Lsolar,Teff,X,Z, do_plots=True):
    max_zones = 8000
    # calculate Stellar radius
    Rs = np.sqrt(Lsolar*cst.L_Sun/(4*np.pi*cst.sigma*Teff**4))
    Ms = Msolar*cst.M_Sun
    Ls = Lsolar*cst.L_Sun
    
    # define parameter arrays
    L = np.zeros(max_zones, dtype=np.float64)
    M = np.zeros(max_zones, dtype=np.float64)
    P = np.zeros(max_zones, dtype=np.float64)
    T = np.zeros(max_zones, dtype=np.float64)
    r = np.zeros(max_zones, dtype=np.float64)
    tau = np.zeros(max_zones, dtype=np.float64)
    rho = np.zeros(max_zones, dtype=np.float64)
    kappa = np.zeros(max_zones, dtype=np.float64)
    eps = np.zeros(max_zones, dtype=np.float64)
    dlnPdlnT = np.zeros(max_zones, dtype=np.float64)

    # initial zone values    
    tau[0] = 2/3
    rho[0] = 0.
    kappa[0] = Opacity(Teff, 1e-8, 0.7, 0.008)
    eps[0] = 0.
    dlnPdlnT[0] = 99.9
    
    M[0] = Ms
    L[0] = Ls
    P[0] = 2*cst.G*Ms/Rs**2/3/kappa[0]
    T[0] = Teff
    r[0] = Rs

    M[1] = Ms
    L[1] = Ls
    P[1] = 0.
    T[1] = 0.

    dr = -(1/1000)*Rs
    
    # calculate surface condition. Only the firt layer after boundary will use these approximations
    step_size_condition = 0
    M[1], L[1], r[1], P[1], T[1], rho[1], kappa[1], eps[1], dlnPdlnT[1] = Surface(1, Ms, Ls, M, L, r, X, Z, dr, P, T, rho, kappa, eps, step_size_condition)

    # calculate optical depth change
    tau[1] = tau[0] + Optical_Depth_Change(kappa[1], kappa[0], rho[1], rho[0], r[1], r[0])
    
    # i=1
    # print('zone   r        tau       M        L          T         P        rho       kap       eps       dlnPdlnT')
    # print('-------------------------------------------------------------------------------------------------------')
    # print('%i %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.1f '%(i,r[i],tau[i], M[i], L[i],  T[i], P[i], rho[i], kappa[i], eps[i], dlnPdlnT[i]))


#----------------------------------------------------------------------------
#       Start integration towards the center
#----------------------------------------------------------------------------
    # variables to hold derivatives
    f_im1=np.zeros(4, dtype=np.float64)
    dfdr=np.zeros(4, dtype=np.float64)
    f_i=np.zeros(4, dtype=np.float64)
    conv_flag = False
    core_flag = False
    end_M_flag = False
    end_R_flag = False
    end_L_flag = False
    end_var_flag = False
    degen_flag = False
    end_zone_flag = False
    dpdr_flag = False
    
    Y        = 1.0 - X - Z
    mu       = Mean_Molecular_Weight(X, Y, Z)
    gamma    = Specific_Heat_Ratio()

    for i in range(2,max_zones):

        #       calculate initial derivatives        
        f_im1[0] = P[i-1]
        f_im1[1] = M[i-1]
        f_im1[2] = L[i-1]
        f_im1[3] = T[i-1]
                
        dfdr[0] = dPdr(M[i-1], rho[i-1], r[i-1])
        dfdr[1] = dMdr(r[i-1], rho[i-1])
        dfdr[2] = dLdr(r[i-1], rho[i-1], eps[i-1])
        dfdr[3],rc_flag = dTdr(kappa[i-1], rho[i-1], T[i-1], L[i-1], r[i-1], mu, M[i-1], gamma, dlnPdlnT[i-1])

        # apply Runge-Kutta integration
        f_i = RUNGE(f_im1, dfdr, r[i-1], dr, X, Y, Z, step_size_condition)
        
        # check convergence in zone
        if (f_i[1]/Ms < 0.01 and f_i[2]/Ls < 0.1 and np.abs(r[i-1] + dr)/Rs < 0.01 
            and f_i[0]>0. and f_i[2]>0. and f_i[3]>0. and r[i-1] + dr > 0.):
            conv_flag = True
            
        # check if maximum of zones has been reached            
        if (i == max_zones-1):
            print('The model reached number of zones')
            end_zone_flag = True

        # end of star reached before convergence
        #----------------------------------------------------------------------
        if (f_i[1] < 0.):
            print('The model reached the mass limit before convergence')
            end_M_flag = True

        if ((r[i-1] + dr) < 0.):
            print('The model reached the radius limit before convergence')
            end_R_flag = True
        
        if (f_i[2] < 0.):
            print('The model reached negative luminosity before convergence')
            end_L_flag = True
        #----------------------------------------------------------------------

                        

        # update zone values
        r[i] = r[i-1] + dr
        P[i] = f_i[0]
        M[i] = f_i[1]
        L[i] = f_i[2]
        T[i] = f_i[3]
        dlnPdlnT[i] = PTgradient(P[i-1], P[i], T[i-1], T[i])
        rho[i] = Density(T[i], P[i], mu, step_size_condition)
        kappa[i] = Opacity(T[i], rho[i], X, Z)
        eps[i] = Nuclear(T[i], rho[i], X, Z)
        tau[i] = tau[i-1] + Optical_Depth_Change(kappa[i], kappa[i-1], rho[i], rho[i-1], r[i], r[i-1])
        
        # check if dP/dr tends to zero at core           
        if (conv_flag == True and np.abs(dPdr(M[i], rho[i], r[i])) > 1.0e4):
            #print('The model has large dP/dr at the core')
            dpdr_flag = True
            conv_flag = True

        # check to see if gas is degenerate
        mu_e=2/(1+X)
        rho_degen = np.pi/3*( (cst.k_B*T[i]/(mu*cst.m_H)) * (20*cst.m_e*(cst.m_H*mu_e)**(5/3)/cst.h**2))**(3/2)        
        if (rho[i] > rho_degen):
            print('The gas turned degenerate before convergence')
            degen_flag = True
        
        # check for large variation at the end
        tol = 0.5
        if (conv_flag == True and (np.abs(P[i]-P[i-1])/P[i-1] > tol or np.abs(T[i]-T[i-1])/P[i-1] > tol
                                    or np.abs(M[i]-M[i-1])/M[i-1] > tol)):
            print('The model has large variations at the end zones')
            end_var_flag = True
            conv_flag = False


        # if zone converged and core reached exit
        if (conv_flag == True):
            print('The Model converged!')
            istop = i
            #################################################################
            # get core conditions
            rhocor = M[i]/(4.0e0/3.0e0*np.pi*r[i]**3)              
            epscor = L[i]/M[i]
            Pcore  = P[i] + 2.0e0/3.0e0*np.pi*cst.G*rhocor**2*r[i]**2
            Tcore  = (Pcore - cst.a_rad_o3*T[i]**4)*mu*cst.m_H/(rhocor*cst.k_B)
            
            # core using Statstar function
            rhocor1, epscor1, Pcore1, Tcore1, kappacor1, dlnPdlnTcor1 = Core(M[i], L[i], P[i], T[i], X, Z, r[i])
            
            print('')
            print('Extrapolated Core conditions:')
            print('-----------------------------------------------------------')
            print('zone     Rho           Eps           P             T')
            print('-----------------------------------------------------------')
            print('last     %8.3e     %8.3e     %8.3e     %8.3e'%(rho[i],eps[i],P[i],T[i]))
            print('core     %8.3e     %8.3e     %8.3e     %8.3e \n'%(rhocor,epscor,Pcore,Tcore))
            break
        elif(end_M_flag == True or end_L_flag == True or end_R_flag == True or 
             end_var_flag == True or degen_flag == True or end_zone_flag == True):
            istop = i
            conv_flag = False
            break

        #print('%i %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.3e %8.1f '%(i,r[i],tau[i], M[i], L[i],  T[i], P[i], rho[i], kappa[i], eps[i], dlnPdlnT[i]))
        #print(i, M[i]/Ms, L[i]/Ls, r[i]/Rs, dr/r[i])
        #print(i,(P[i]-P[i-1])/P[i-1], (T[i]-T[i-1])/T[i-1], (L[i]-L[i-1])/L[i-1], (M[i]-M[i-1])/M[i-1])
            
            
#               Is it time to change step size?

        # check for thermodinamic equilibrium
        crosssec_photon = 1.e-28 # m2
        mean_free_path = mu*cst.m_H/rho[i]/crosssec_photon
        dtdri,aux = dTdr(kappa[i], rho[i], T[i], L[i], r[i], mu, M[i], gamma, dlnPdlnT[i])
        
        #print(mean_free_path, dr, dr/(0.01*T[i]/dtdri), degen_flag, rho[i]/rho_degen)
        
        # if (step_size_condition == 0 and M[i] < 0.99*Ms):
        #     dr = -Rs/100
        #     step_size_condition = 1
    
        # elif (step_size_condition == 1 and np.abs(dr) > 5*r[i]):
        #     dr = dr/10.
        #     step_size_condition = 2
            
        # vary step size according to variable change
        tolvar = 0.05
        if (np.abs(P[i]-P[i-1])/P[i-1] > tolvar or np.abs(T[i]-T[i-1])/T[i-1] > tolvar or
            np.abs(L[i]-L[i-1])/L[i-1] > tolvar or np.abs(M[i]-M[i-1])/M[i-1] > tolvar):
            dr = dr/2
        elif(np.abs(P[i]-P[i-1])/P[i-1] < tolvar/10 or np.abs(T[i]-T[i-1])/T[i-1] < tolvar/10 or
            np.abs(L[i]-L[i-1])/L[i-1] < tolvar/10 or np.abs(M[i]-M[i-1])/M[i-1] < tolvar/10):
            dr = dr*2
            
            
            
                        
    # save only usable array
    r = r[0:istop+1]
    P = P[0:istop+1]
    M = M[0:istop+1]
    L = L[0:istop+1]
    T = T[0:istop+1]
    dlnPdlnT = dlnPdlnT[0:istop+1] 
    rho =  rho[0:istop+1]
    kappa = kappa[0:istop+1]
    eps = eps[0:istop+1]
    tau = tau[0:istop+1]
    
#    print('Rough estimate of central pressure: %8.3e %8.3e'%(3*cst.G*Ms**2/8/np.pi/Rs**4,np.mean(P)))

            
#----------------------------------------------------------------------------
#       End integration towards the center
#----------------------------------------------------------------------------
        
    if do_plots:
        import matplotlib.pyplot as plt
        plt.figure()          
        plt.plot(r,L/L.max(),'-o',label='luminosity')
        plt.plot(r,T/T.max(),'-o',label='temperature')
        plt.plot(r,P/P.max(),'-o',label='pressure')
        plt.plot(r,rho/rho.max(),'-o',label='density')
        plt.plot(r,eps/eps.max(),'-o',label='Eps')          
        plt.legend()
        plt.show()
        
    dpdr_val = np.abs(dPdr(M[istop], rho[istop], r[istop]))
    if(np.abs(L[i]/Ls) < 0.05):
        val = 0.05
    else:
        val = np.abs(L[i]/Ls)
        
    residual = np.abs(dpdr_val * val)
    #residual = np.mean([np.abs(M[istop]/Ms), np.abs(L[istop]/Ls), np.abs(r[istop]/Rs)]) 
    
    #print(dpdr_val , val)

    if (M[istop] < 0. or L[istop] < 0. or r[istop] < 0.):
        neg_flag = True
    else:
        neg_flag = False
        
    ###########################################################################
    # Write file
    ###########################################################################
    f=open('starmodl_py.dat','w')
    f.write('   r        Qm       L_r       T        P        rho      kap      eps     fl   dlnPdlnT\n')

    for ic in range(0,istop+1):
        i = ic #istop+1 - ic
        Qm = 1.0e0 - M[i]/Ms    # Total mass fraction down to radius


        if (dlnPdlnT[i] < gamma):
            rcf = ' c '
        else:
            rcf = ' r '
        if (np.abs(dlnPdlnT[i]) > 99.):
            rcf= ' * ' 

        s='{0:7.3E}    {1:7.3E}    {2:7.3E}    {3:7.3E}    {4:7.3E}    {5:7.3E}    {6:7.3E}    {7:6.3E}    {8:1s}    {9:5.1f}\n'.format(r[i], Qm, L[i], T[i], P[i], rho[i], kappa[i],eps[i], rcf, dlnPdlnT[i])
        f.write(s)
    
    if conv_flag:
        return conv_flag, dpdr_val, residual, [rho[istop],eps[istop],T[istop],P[istop]] #  [rhocor1,epscor1,Tcore1,Pcore1] # 
    else:
        return conv_flag, dpdr_val, residual, [99,99,99,99]


#------------------------------------------------------------------------------

def main():

#
#  Enter desired stellar parameters
#


      getinp=1  # read in input
      if (getinp == 1):
            Msolar=float(input(' Enter the mass of the star (in solar units):'))
            Lsolar=float(input(' Enter the luminosity of the star (in solar units):'))
            Te=float(input(' Enter the effective temperature of the star (in K):'))
            Y=-1.0
            while (Y < 0.0):
                X=float(input(' Enter the mass fraction of hydrogen (X):'))
                Z=float(input(' Enter the mass fraction of metals (Z):'))
                Y = 1.e0 - X - Z
                if Y < 0:
                    print('You must have X + Z <= 1. Please reenter composition.')

      conv, dpdr_val, res, core_vals=StatStar(Msolar,Lsolar,Te,X,Z)

main()


