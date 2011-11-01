#!/usr/local/epd/bin/python
#------------------------------------------------------------
#                Classical Symplectic propagator on GPU
#-------------------------------------------------------------

import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

import h5py


#====================================================================================================

class Base_Symplectic_Transformations :
	"""
	This is an abstract class that implements two symplectic transformations used 
	in the symplectic integrator schemes.

	Virtual methods used by the class:
		dTdp (P) -- the derivative of the kinetic enerhy w.r.t. the momentum
		dVdq (q) -- the derivative of the potential w.r.t. the corrdiante 
	
	Virtual properties used by the class:
		dt	-- the time step for a single iteration
	"""

	def DT_GPU(self, X, c):
		DIM = X.size
		floatSize = X.dtype.itemsize
		q = gpuarray.empty(DIM/2,X.dtype)
		p = gpuarray.empty(DIM/2,X.dtype)
		XNext = gpuarray.empty(DIM,X.dtype)
		cuda.memcpy_dtod(q.ptr,X.ptr, floatSize*DIM/2 )
		cuda.memcpy_dtod(p.ptr,X.ptr + floatSize*DIM/2 , floatSize*DIM/2)
		qNext = q + c*self.dt* self.dTdp(p)
		pNext = p
		cuda.memcpy_dtod( XNext.ptr                    , qNext.ptr , floatSize*DIM/2)
		cuda.memcpy_dtod( XNext.ptr +  floatSize*DIM/2 , pNext.ptr , floatSize*DIM/2)
        	return XNext

	def DV_GPU(self, X, d):
		DIM = X.size
        	floatSize = X.dtype.itemsize
        	q = gpuarray.empty(DIM/2,X.dtype)
        	p = gpuarray.empty(DIM/2,X.dtype)
        	XNext = gpuarray.empty(DIM,X.dtype)
        	cuda.memcpy_dtod(q.ptr,X.ptr, floatSize*DIM/2 )
        	cuda.memcpy_dtod(p.ptr,X.ptr + floatSize*DIM/2 , floatSize*DIM/2)
		qNext = q
        	pNext = p - d*self.dt* self.dVdq(q)
		cuda.memcpy_dtod( XNext.ptr                    , qNext.ptr , floatSize*DIM/2)
        	cuda.memcpy_dtod( XNext.ptr +  floatSize*DIM/2 , pNext.ptr , floatSize*DIM/2)
        	return XNext


#---------------------------------------------------------------
#            First Order
#---------------------------------------------------------------

def SymplecticPropagatorStep1_GPU( X_GPU , dt , dTdp, dVdq):
        """
        Symplectic propagator step, order 1.
        """
        NextX =     SymplecticDV_GPU(X_GPU, dt,dVdq,1.)
        return SymplecticDT_GPU( NextX ,dt,dTdp,1.)

def SymplecticPropagator1_GPU( X ,  dTdp, dVdq, t0,T, DIM_time , FileName ):
        f = h5py.File(FileName)
        DIM_qp =  X.__len__()
        dt = (T-t0)/(DIM_time-1.);
        tRange = np.linspace(t0,T,DIM_time)
        trajectory  = np.zeros( (DIM_time,DIM_qp), float )
        NextX = gpuarray.to_gpu(X)
        tCounter = 0
        for t in tRange:
                trajectory[tCounter] = NextX.get()
                tCounter+=1
                NextX = SymplecticPropagatorStep1_GPU( NextX , dt , dTdp, dVdq)
        f.create_dataset("pq", data = trajectory   )
        f.close()
        return trajectory


#------------------------------------------------------------
#           Fourth Order
#------------------------------------------------------------

class Base_Symplectic_Single_Iteration_4 (Base_Symplectic_Transformations) :
	"""
	This base class implements a single iteration for the symplectic integrator of 4th order.
	"""
	
	def Iteration (self, X_GPU):
        	"""
		Symplectic propagator, order 4.
        	"""
        	a = 1./(2. - 2.**(1./3.))
        	c1 = 0.5*a
        	c2 = (1.-2.**(1./3.))*c1
        	c4 = c1
        	c3 = c2
        	d1 = a
        	d2 = - 2.**(1./3.)*a
        	d3 = d1
        	d4 = 0.
        	NextX = self.DT_GPU( self.DV_GPU(    X_GPU, d1 ),  c1 )
        	NextX = self.DT_GPU( self.DV_GPU(NextX, d2 )  , c2 )
        	NextX = self.DT_GPU( self.DV_GPU(NextX, d3 )  , c3 )
        	NextX = self.DT_GPU( self.DV_GPU(NextX, d4 )  , c4 )
        	return NextX


#------------------------------------------------------------------
#		Base propagator class
#-----------------------------------------------------------------

class Base_Symplectic_4 (Base_Symplectic_Single_Iteration_4) :
	"""
	This base class implements the 4th order symplectic propagator.
	
	Virtual properties used by this class: 
		x_init : The initial conditions in the phase space [x1,x2,..., p1,p2,p3...]
	"""
	def Run(self):
		"""
		This function executes the iterative process of the propagation of trajectories 
		"""
		tRange = np.linspace(self.t0, self.T, self.DIM_time)
                NextX = gpuarray.to_gpu(self.x_init)
                tCounter = 0

		if self.fileName <> '' : 
                        f = h5py.File(self.fileName)

			if not isinstance(self.degrees_of_freedom, int) :
				raise TypeError ("The atribute  <degrees_of_freedom> must be integer.")
			
			if self.degrees_of_freedom % 2 <> 0 :
				raise ValueError ("The atribute <degrees_of_freedom> has to be an even integer.\n \
				(For example,if you have a 1D system with the cannonical varibales (x, p) then degrees_of_freedom=2.)")			
		
			# Make sure that the attribute <degrees_of_freedom> is a positive integer
			self.degrees_of_freedom = np.abs(self.degrees_of_freedom)
			
			data_set = f.create_dataset( 'DegreesOfFreedom' , (1,1) , dtype=np.int, compression=self.compression)
                        data_set[:] = self.degrees_of_freedom

                        try :
                                getattr(self, 'density')
                                data_set = f.create_dataset( 'Densities' , self.density.shape, dtype=self.density.dtype, compression=self.compression)
                                data_set[:] = self.density
                        except AttributeError : pass

                        # Save the string contaning potential, if it was defined
                        try :
                                getattr(self, 'potential_str')
                                data_set = f.create_dataset( 'Potential' , (1,1), dtype=h5py.new_vlen(str), compression=self.compression)
                                data_set[:] = self.potential_str
                        except AttributeError : pass

                        try :
                                getattr(self, 'hamiltonian_str')
                                data_set = f.create_dataset( 'Hamiltonian', (1,1), dtype=h5py.new_vlen(str), compression=self.compression)
                                data_set[:] = self.hamiltonian_str
                        except AttributeError : pass

                for t in tRange:
                        if self.fileName <> '' and tCounter%self.skipFrame == 0:
                                frame =  NextX.get()
                                data_set = f.create_dataset( str(tCounter), frame.shape, dtype=frame.dtype, compression=self.compression)
                                data_set[:] = frame
                        tCounter += 1
                        NextX = self.Iteration( NextX )

                if self.fileName <> '' :
                        f.close()
		return NextX.get()
	

	def Set_TimeTrack(self, T ,t0=0 ,DIM_time = 1000, skipFrame=1, fileName='', compression=None) : 
        	"""
                T        :  final time
		t0	 : initial time. Default = 0.
                DIM_time :  Number of time points. Default = 1000
                density  :  Flat array of densities in the phase-space
                filename :  Nane of the hdf5 output file
                skipFrame:  How many frames to skip and save a frame. Save all: Default =  1 
		
		Virtual properties used by this class :

			x_init       :  initial phase-space coordinate

        	"""
		self.skipFrame = skipFrame
		self.compression = compression
		self.fileName = fileName
		self.T  = T
		self.t0 = t0
		self.DIM_time = DIM_time
		self.dt = (T - t0)/( DIM_time-1. );	
	

#--------------------------------------------------------
#                     Utilities
#-------------------------------------------------------

def PhaseSpaceMesh2D(xMin,xMax,yMin,yMax,pxMin,pxMax,pyMin,pyMax,DIM1):
	pyRange_pxRange_yRange_xRange = np.mgrid[0:DIM1,0:DIM1,0:DIM1,0:DIM1 ]/float(DIM1-1.)
        x_Mesh  =  pyRange_pxRange_yRange_xRange[3]
        y_Mesh  =  pyRange_pxRange_yRange_xRange[2]
        px_Mesh =  pyRange_pxRange_yRange_xRange[1]
        py_Mesh =  pyRange_pxRange_yRange_xRange[0]
        x_Mesh  =   (xMax-xMin)*x_Mesh  + xMin
        y_Mesh  =   (yMax-yMin)*y_Mesh  + yMin
        px_Mesh = (pxMax-pxMin)*px_Mesh + pxMin
        py_Mesh = (pyMax-pyMin)*py_Mesh + pyMin
	return x_Mesh, y_Mesh, px_Mesh, py_Mesh


#----------------------------------------------------------------------------------------------
#     Experimental functions
#----------------------------------------------------------------------------------------------
def GPUTake( out_GPUptr , input_GPUptr , initial_input_index , final_input_index, dtype ):
	typeSize = dtype.itemsize
	cuda.memcpy_dtod( out_GPUptr , input_GPUptr + typeSize*initial_input_index , typeSize*(final_input_index - initial_input_index) )
  
def GPUReplace( output_GPUptr , input_GPUptr , initial_output_index, input_size , dtype):
	typeSize = dtype.itemsize
	cuda.memcpy_dtod( output_GPUptr +  typeSize*initial_output_index   , input_GPUptr , typeSize*input_size)













	




