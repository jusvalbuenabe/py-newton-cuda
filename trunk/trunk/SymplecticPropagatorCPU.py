import numpy as np
#--------------------------------------------------------------
#                     CPU 
#--------------------------------------------------------------

def SymplecticDT(X,dt,dTdp,c):
        DIM = X.__len__()
        floatSize = 8
        q = np.frombuffer(X,np.float,DIM/2,0)
        p = np.frombuffer(X,np.float,DIM/2, floatSize*DIM/2 )
        qNext = q + c*dt*dTdp(p)
        return np.append(qNext,p)

def SymplecticDV(X,dt,dVdq,d):
        DIM = X.__len__()
        floatSize = 8
        q = np.frombuffer(X,np.float,DIM/2,0)
        p = np.frombuffer(X,np.float,DIM/2, floatSize*DIM/2 )
        pNext = p - d*dt*dVdq(q)
        return np.append(q,pNext)
#---------------------------------------------------------------
#            First Order
#---------------------------------------------------------------

def SymplecticPropagatorStep1( X , dt , dTdp, dVdq):
        """
        Symplectic propagator step, order 1.
        """
        NextX =     SymplecticDV(X, dt,dVdq,1.)
        return SymplecticDT( NextX ,dt,dTdp,1.)

def SymplecticPropagator1( X ,  dTdp, dVdq, t0,T, DIM_time , FileName ):
        f = h5py.File(FileName)
        DIM_qp =  X.__len__()
        dt = (T-t0)/(DIM_time-1.);
        tRange = np.linspace(t0,T,DIM_time)
        trajectory  = np.zeros( (DIM_time,DIM_qp), float )
        NextX = X
        tCounter = 0
        for t in tRange:
                trajectory[tCounter] = NextX
                tCounter+=1
                NextX = SymplecticPropagatorStep1( NextX , dt , dTdp, dVdq)
        f.create_dataset("pq", data = trajectory   )
        f.close()
        return trajectory

------------------------------------------------------------------
#              Second Order
#------------------------------------------------------------------             

def SymplecticPropagatorStep2( X , dt , dTdp, dVdq):
        """
        Symplectic propagator, order 2.
        """
        NextX =  SymplecticDT( SymplecticDV(      X,dt,dVdq,1./2.)  , dt,dTdp, 1.)
        NextX =                  SymplecticDV(NextX,dt,dVdq,1./2.)
        return NextX


def SymplecticPropagator2( X ,  dTdp, dVdq, t0,T, DIM_time , FileName ):
        f = h5py.File(FileName)
        DIM_qp =  X.__len__()
        dt = (T-t0)/(DIM_time-1.);
        tRange = np.linspace(t0,T,DIM_time)
        trajectory  = np.zeros( (DIM_time,DIM_qp), float )
        NextX = X
        tCounter = 0
        for t in tRange:
                trajectory[tCounter] = NextX
                tCounter+=1
                NextX = SymplecticPropagatorStep2( NextX , dt , dTdp, dVdq)
        f.create_dataset("pq", data = trajectory   )
        f.close()
        return trajectory

#------------------------------------------------------------
#           Third Order
#------------------------------------------------------------

def SymplecticPropagatorStep3( X , dt , dTdp, dVdq):
        """
        Symplectic propagator, order 2.
        """
        NextX = SymplecticDT( SymplecticDV(    X,dt, dVdq, 2./3.)  , dt,dTdp,  7./24.)
        NextX = SymplecticDT( SymplecticDV(NextX,dt, dVdq,-2./3.)  , dt,dTdp,  3./4.)
        NextX = SymplecticDT( SymplecticDV(NextX,dt, dVdq, 1.)     , dt,dTdp, -1./24.)
        return NextX

def SymplecticPropagator3( X ,  dTdp, dVdq, t0,T, DIM_time , FileName ):
        f = h5py.File(FileName)
        DIM_qp =  X.__len__()
        dt = (T-t0)/(DIM_time-1.);
        tRange = np.linspace(t0,T,DIM_time)
        trajectory  = np.zeros( (DIM_time,DIM_qp), float )
        NextX = X
        tCounter = 0
        for t in tRange:
                trajectory[tCounter] = NextX
                tCounter+=1
                NextX = SymplecticPropagatorStep3( NextX , dt , dTdp, dVdq)
        f.create_dataset("pq", data = trajectory   )
        f.close()
        return trajectory

#------------------------------------------------------------
#           Fourth Order
#------------------------------------------------------------

def SymplecticPropagatorStep4( X , dt , dTdp, dVdq):
        """
        Symplectic propagator, order 2.
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
        NextX = SymplecticDT( SymplecticDV(    X,dt, dVdq, d1 )  , dt,dTdp,  c1 )
        NextX = SymplecticDT( SymplecticDV(NextX,dt, dVdq, d2 )  , dt,dTdp,  c2 )
        NextX = SymplecticDT( SymplecticDV(NextX,dt, dVdq, d3 )  , dt,dTdp,  c3 )
        NextX = SymplecticDT( SymplecticDV(NextX,dt, dVdq, d4 )  , dt,dTdp,  c4 )
        return NextX

def SymplecticPropagator4( X ,  dTdp, dVdq, t0,T, DIM_time , FileName ):
        f = h5py.File(FileName)
        DIM_qp =  X.__len__()
        dt = (T-t0)/(DIM_time-1.);
        tRange = np.linspace(t0,T,DIM_time)
        trajectory  = np.zeros( (DIM_time,DIM_qp), float )
        NextX = X
        tCounter = 0
        for t in tRange:
                trajectory[tCounter] = NextX
                tCounter+=1 
                NextX = SymplecticPropagatorStep4( NextX , dt , dTdp, dVdq)
        f.create_dataset("pq", data = trajectory   )
        f.close()
        return trajectory

