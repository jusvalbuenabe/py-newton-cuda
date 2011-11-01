#!/usr/local/epd/bin/python
# -*- coding: utf-8 -*-
#          Classical Symplectic propagator  test 2D
#----------------------------------------------------------------
from SymplecticPropagator import * 

#------------------------------------------------------------------
#   Defining the system: Harmonic oscillator in two dimensions
#-----------------------------------------------------------------

class Harmonic_Oscillator_2D (Base_Symplectic_4) :
                """
                Harmonic oscillator in two dimensions by the symplectic integrator of fourth order
                """
                def dTdp (self, p) : return p/self.m

                def dVdq (self, q) :
                        DIM = q.size
                        DIM1 = DIM/2
                        typeSize = q.dtype.itemsize
                        x = gpuarray.empty(DIM1,q.dtype)
                        y = gpuarray.empty(DIM1,q.dtype)
                        out   = gpuarray.empty(DIM, q.dtype)
                        cuda.memcpy_dtod(x.ptr, q.ptr                 , typeSize*DIM1 )
                        cuda.memcpy_dtod(y.ptr, q.ptr + typeSize*DIM1 , typeSize*DIM1 )
                        #--------------------The gradient is------------------
                        x = self.omega_x*x
                        y = self.omega_y*y
                        #--------------------------------------------------------------------
                        cuda.memcpy_dtod( out.ptr                    , x.ptr , typeSize*DIM1)
                        cuda.memcpy_dtod( out.ptr +  typeSize*DIM1   , y.ptr , typeSize*DIM1)
                        return out

                def Set_InitialConditions(self,DIM1=10,xMin=-10.,xMax=10.,yMin=-10.,yMax=10.,pxMin=-10.,pxMax=10.,pyMin=-10.,pyMax=10.):
                        """
                        The optional arguments are:
                                DIM1 : The dimension of one side of the four-dimensional array of initial positions in the discretization of /
				       the phase space. This implies that the total number of particles is DIM1**4
                                xMin,xMax,pMin... : The extreme values of the four-dimensional hypercube in the phase space
                        
                        """
                        self.degrees_of_freedom = 4
                        self.omega_x = 1.
                        self.omega_y = 1.
                        self.m = 1.
                        self.potential_str   = ' (%f**2*x**2 + %f**2*y**2)/(2.*%f) '%(self.omega_x,self.omega_y,self.m)
                        self.hamiltonian_str = ' (px**2 + py**2)/(2*%f)'%self.m + ' + ' + self.potential_str
                        #      Initial condition in the phase space of the ensemble of particles 
                        x_Mesh,y_Mesh,px_Mesh,py_Mesh = PhaseSpaceMesh2D(xMin,xMax,yMin,yMax,pxMin,pxMax,pyMin,pyMax,DIM1)
                        self.x_init = np.append( x_Mesh.flatten(), np.append(y_Mesh.flatten(), np.append(px_Mesh.flatten(), py_Mesh.flatten()) )  )
                        #       Initial density in the phase space
                        x_mu, y_mu         = [0. , 0.]
                        px_mu, py_mu       = [0. , 0.]
                        x_sigma, y_sigma   = [5. , 5.]
                        px_sigma, py_sigma = [5. , 5.]
                        self.density = np.exp( -(x_Mesh.flatten()-x_mu)**2/(2*x_sigma**2)  -(y_Mesh.flatten()-y_mu)**2/(2*y_sigma**2)   -(py_Mesh.flatten()-py_mu)**2/(2*py_sigma**2)  -(px_Mesh.flatten()-px_mu)**2/(2*px_sigma**2)  )


#----------------------------------------------------------------
if __name__ == "__main__": 

	# Creating an instance
	harmOsc2D = Harmonic_Oscillator_2D()

	# Setting up the initial conditions 
	harmOsc2D.Set_InitialConditions(DIM1=10)

	# Setting up the time-track parameters
	harmOsc2D.Set_TimeTrack( T=6.0,  DIM_time = 1000 , skipFrame = 10 ,  fileName = "_2DHarmOsc.h5" )

	# Running 
	harmOsc2D.Run()

	# Visualizing the result
	#%run 2D_classical_frame_viwer.py _2DHarmOsc.h5

