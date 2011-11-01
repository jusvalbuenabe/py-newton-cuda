#!/usr/local/epd/bin/python
#----------------------------------------------------------------
#                Classical Symplectic propagator  test  1D
#----------------------------------------------------------------
from SymplecticPropagator import *
#-----------------------------------------------------------------

#-------------------------------------------------------------------------------------------
#                     Harmonic Oscillator 1D
#-------------------------------------------------------------------------------------------
class Harmonic_Oscillator_1D (Base_Symplectic_4) :
                """
                Harmonic oscillator in one dimension by the symplectic integrator of fourth order
                """
                def dTdp (self, p) : return p/self.m
                def dVdq (self, q):  return q/self.m
                def __init__(self):
                        self.degrees_of_freedom = 2
                        self.omega = 1.
                        self.m = 1.
                        self.potential_str   = ' (%f**2*x**2)/(2*%f) '%(self.omega,self.m)
                        self.hamiltonian_str = ' (px**2)/(2*%f)'%self.m + ' + ' + self.potential_str
                        #
                        DIM1 = 10
                        xMin = -10.
                        xMax =  10.
                        pMin = -10.
                        pMax =  10.
                        xRange0 = np.linspace(xMin, xMax, DIM1)
                        pRange0 = np.linspace(pMin, pMax, DIM1)
                        x_Mesh, p_Mesh = np.array(np.meshgrid( xRange0 , pRange0 ))
                        self.x_init = np.append( x_Mesh.flatten() , p_Mesh.flatten() )
                        #       Initial density 
                        x_mu = 0.
                        p_mu = 0.
                        x_sigma = 3.



#==================================================================
if __name__ == "__main__": 

	harmOsc1D = Harmonic_Oscillator_1D()

	harmOsc1D.Set_TimeTrack( T=6.0, skipFrame = 10 ,  fileName = "_1DHarmOsc.h5" )

	harmOsc1D.Run()



