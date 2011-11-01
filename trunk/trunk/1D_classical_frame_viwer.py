#!/usr/local/epd/bin/python

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import matplotlib.cm as cm

import Tkinter as Tk
import tkMessageBox

import h5py
import numpy as np


class Classical_Frame_Viewer_1D (Tk.Frame) :
	"""
	Classical_Frame_Viewer_1D -- Plot the probability distributions obtained by solving 
	the Liouville equation. 
	"""

	def __init__ (self, master, hdf5_file_name) :
		"""
		Constructor of Classical_Frame_Viewer_1D class.

		ARGUMENTS
		---------
			master		-- the main frame
			hdf5_file_name 	-- the name of the HDF5 file for viewing  	
		"""
		
		# Open the HDF5 file
		self.HDF5_File = h5py.File (hdf5_file_name, 'r')

		# Check whether the file with the data in the file has the correct number of degrees of freedom
		data_set = self.HDF5_File['DegreesOfFreedom']
		if data_set[0][0] <> 2 :	
			tkMessageBox.showerror ("Loading file ["+hdf5_file_name+"]...", \
				"The data stored in the file ["+hdf5_file_name+"] is not 1D. This viewer can display only 1D data.")
			raise ValueError ("The data stored in the file is not 1D. This viewer can display only 1D data.")

		# Collect the names of all the frames in the HDF5 file
		self.frame_names = []
		for x in self.HDF5_File.items() :
			if x[0] == 'DegreesOfFreedom' : 
				# We already processed this data set
				pass

			elif x[0] == 'Potential' :
				# The potential energy is stored in the HDF5 file as string
				data_set = self.HDF5_File['Potential']
				self.Potential = data_set[0][0]
	
				if not isinstance(self.Potential, str) : 
					tkMessageBox.showerror ("Loading file ["+hdf5_file_name+"]...", \
						"[Potential] must be string.")
					raise ValueError ("[Potential] must be string")
			
			elif x[0] == 'Hamiltonian' :
				# The Hamiltonian is stored in the HDF5 file as string
				data_set = self.HDF5_File['Hamiltonian']
				self.Hamiltonian = data_set[0][0]
	
				if not isinstance(self.Hamiltonian, str) : 
					tkMessageBox.showerror ("Loading file ["+hdf5_file_name+"]...", \
						"[Hamiltonian] must be string.")
					raise ValueError ("[Hamiltonian] must be string")
	
			elif x[0] == 'Densities' :
				# The probability density tags for each particle is saved in the HDF5 file
				data_set = self.HDF5_File['Densities']
				self.Densities = data_set[...]

				if len(self.Densities.shape) <> 1 : self.Densities = self.Densities.flatten()
			else :
				try :
					self.frame_names.append (int(x[0])) 
				except ValueError :
					tkMessageBox.showwarning ("Loading file ["+hdf5_file_name+"]...", \
		"The HDF5 File contains data set [" + x[0] +"] that will ignored because of unrecognized name.") 
	
		# The consistency check
		if len(self.frame_names) == 0 :
			tkMessageBox.showerror ("Loading file ["+hdf5_file_name+"]...", \
				"No frames are saved in file ["+hdf5_file_name+"]")
			raise ValueError ("No frames are in HDF5 file")
	
		# Sorting the frame's names in the increasing order
		self.frame_names.sort ()

		# Remember master for calling timer
		self.master = master
	
		# Create GUI
		Tk.Frame.__init__ (self, master) 
		self.pack (expand=1, fill=Tk.BOTH)
		self.createWidgets ()


	def createWidgets (self) :
		"""
		Create GUI
		"""
		
		# First row
		self.frame1 = Tk.Frame (self)
		self.frame1.pack (side="top")
	
		Tk.Label (self.frame1, text="Frame ").pack (side="left")

		# Variable for storing the frame number
		self.frame_number = Tk.IntVar ()
		self.frame_number.set (0)

		Tk.Spinbox (self.frame1, from_=0, to=len(self.frame_names)-1, increment=1, \
			textvariable=self.frame_number).pack (side="left")

		Tk.Button (self.frame1, text="Plot individual frame", \
			command=self.plot_probability_distribution).pack (side="left") 

		# Variable that determines whether to plot the potential 
		self.ToPlotPotential = Tk.IntVar ()
		
		cbutton1 = Tk.Checkbutton (self.frame1, text="Plot potential", variable=self.ToPlotPotential)
		cbutton1.pack (side="left")

		# Check whether the HDF5 contained the potential
		try :
			getattr(self, "Potential")
			self.ToPlotPotential.set (1)

		except AttributeError:
			self.ToPlotPotential.set (0)
			cbutton1["state"] = Tk.DISABLED

		# Second row
		self.frame2 = Tk.Frame (self)
		self.frame2.pack (side="top")
	
		Tk.Label (self.frame2, text="Plot's type ").pack (side="left")
		
		# Variable for saving the type of plot 
		self.plot_type = Tk.IntVar ()
		
		self.__type_x_px__		= 0
		self.__type_x_histogram__ 	= 1
		self.__type_px_histogram__ 	= 2

		Tk.Radiobutton (self.frame2, text="x / px", value=self.__type_x_px__, \
			variable=self.plot_type).pack (anchor=Tk.W, side="left")
	
		Tk.Radiobutton (self.frame2, text="x histogram", value=self.__type_x_histogram__, \
			variable=self.plot_type).pack (anchor=Tk.W, side="left")
		
		Tk.Radiobutton (self.frame2, text="px histogram", value=self.__type_px_histogram__, \
			variable=self.plot_type).pack (anchor=Tk.W, side="left")
		
		# Third row
		self.frame3 = Tk.Frame (self)
		self.frame3.pack (side="top")	
		
		Tk.Label (self.frame3, text="Plot's style ").pack (side="left")
		
		# Variable for storing the style of plot
		self.plot_style = Tk.IntVar ()
		
		self.__style_tagged_hexbin__		= 0
		self.__style_tagged_log_hexbin__	= 1
		self.__style_points__			= 2
		self.__style_hexbin__			= 3
		self.__style_log_hexbin__ 		= 4
		
		rb2 = Tk.Radiobutton (self.frame3, text="hexbin densities", \
			value=self.__style_tagged_hexbin__, variable=self.plot_style)
		rb2.pack (anchor=Tk.W, side="left")

		rb3 = Tk.Radiobutton (self.frame3, text="log hexbin densities", \
			value=self.__style_tagged_log_hexbin__, variable=self.plot_style)
		rb3.pack (anchor=Tk.W, side="left")

		# Turn off the options that require Densities dataset if it was not loaded 
		try :
			getattr (self, "Densities") 
		except AttributeError :
			rb2["state"] = Tk.DISABLED
			rb3["state"] = Tk.DISABLED
			self.plot_style.set (self.__style_points__)	

		# The rest plot styles that do not require the Densities dataset  
		Tk.Radiobutton (self.frame3, text="points ", value=self.__style_points__, \
			variable=self.plot_style).pack (anchor=Tk.W, side="left")
		
		Tk.Radiobutton (self.frame3, text="hexbin ", value=self.__style_hexbin__, \
			variable=self.plot_style).pack (anchor=Tk.W, side="left")

		Tk.Radiobutton (self.frame3, text="log hexbin ", value=self.__style_log_hexbin__, \
			variable=self.plot_style).pack (anchor=Tk.W, side="left")

		# Forth row
		self.frame4 = Tk.Frame (self)
		self.frame4.pack (side="top")

		# Define the variable to determine whether to save frames while playing an animation
		self.to_save_frames = Tk.IntVar ()
 		self.to_save_frames.set (0)

		Tk.Checkbutton (self.frame4, text="Save frames as sequence of PNG images while playing animation", \
				variable=self.to_save_frames).pack (side="left")

		# Fifth row
		
		# Define the variable for controlling the size of the image	
		self.resize_to_first_frame_extent = Tk.IntVar ()
		self.resize_to_first_frame_extent.set (1)

		self.frame5 = Tk.Frame (self)
		self.frame5.pack (side="top")

		Tk.Checkbutton (self.frame5, text="Always keep image's extent constant", \
				variable=self.resize_to_first_frame_extent).pack (side="left")
	
		# Sixth row
		self.frame6 = Tk.Frame (self)
		self.frame6.pack (side="top")

		# Define the variable for step size in reducing the number of elements of arrays 
		# to accelerate animation
		self.speed_up_step = Tk.IntVar ()
		self.speed_up_step.set (1)
 
		Tk.Label (self.frame6, text="Speed up animation (by reducing number of particles)").pack (side="left")

		Tk.Spinbox (self.frame6, from_=1, to=100, increment=1, \
				textvariable=self.speed_up_step).pack (side="left")
		
		# Seventh row
		self.frame7 = Tk.Frame (self)
		self.frame7.pack (side="top")

		phb = Tk.Button (self.frame7, text="Plot Hamiltonians", command=self.plot_hamiltonians)
		phb.pack (side="left")
	
		# If the Hamiltonian has not been loaded, then disable the Hamiltonian plotting facility	
		try : 
			getattr (self, "Hamiltonian")
		except AttributeError : 
			phb["state"] = Tk.DISABLED
 
		Tk.Button (self.frame7, text="Go to beginning", \
			command=self.pressed_reset_button).pack (side="left")

		# The button for playing and stopping animation

		# Constants for different state of the animation button
		self.__start_animation__ = "Start animation"
		self.__stop_animation__ = "Stop animation"

		self.animation_button = Tk.Button (self.frame7, text=self.__start_animation__, \
			command=self.play_animation)
		self.animation_button.pack (side="left")

	
		# Defying the frame for drawing
		self.drawing_frame = Tk.Frame (self)
		self.drawing_frame.pack (fill=Tk.BOTH)		
		
		# Define the size of the image
		self.PlotDPI = 90
		self.PlotWidth = int (self.winfo_screenwidth()/2)
		self.PlotHeight = int (65*self.winfo_screenheight()/100)

		self.f = Figure(figsize=(self.PlotWidth/self.PlotDPI, self.PlotHeight/self.PlotDPI), dpi=self.PlotDPI)
		
		self.canvas = FigureCanvasTkAgg(self.f, master=self.drawing_frame)
		self.canvas.get_tk_widget().pack(expand=1, fill=Tk.BOTH) 

		# Embed the toolbox from matplotlib
		toolbar = NavigationToolbar2TkAgg( self.canvas, self.drawing_frame  )
		toolbar.update()
		self.canvas._tkcanvas.pack(expand=1, fill=Tk.BOTH)
		
		# Plot the first frame. It is need for saving the extent of the first frame
		self.plot_probability_distribution () 


	def plot_hamiltonians (self) :
		"""
		Plot Hamiltonians of all the particles as a function of time to check whether the integration method 
		obtain to physical results
		"""
		# Load the first frame to find out the size of elements
		data_set = self.HDF5_File[ str(self.frame_names[0]) ]
		PD_x_px = data_set[...]
		
		# Allocating the memory for all the Hamiltonians
		Hamiltonains = np.zeros ((len(self.frame_names), int(PD_x_px.size/2)), dtype=PD_x_px.dtype) 

		# Calculate the Hamiltonians for each frame
		k = 0
		for fn in self.frame_names :
			data_set = self.HDF5_File[ str(fn) ]
			PD_x_px = data_set[...]
			PD_x_px = np.ascontiguousarray (PD_x_px) 
			x, px = self.extract_x_px (PD_x_px)	
			Hamiltonains[k, :] = eval(self.Hamiltonian)
			k += 1

		# plotting the Hamiltonians 
		self.f.clear ()

		ax = self.f.add_subplot (111, axisbg='k')
		ax.imshow (Hamiltonains.T, origin='lower', interpolation=None, aspect='auto')
		ax.set_xlabel ('frame\'s number')
		ax.set_ylabel ('particle\'s number')
		ax.set_title ('Plot of Hamiltonians for all particles')

		self.canvas.show(); self.canvas.get_tk_widget().update_idletasks()	

	
	def play_animation (self, event=None) :	
		"""
		The method that is been called when the "play/stop animation" button is clicked 
		"""
		if self.animation_button["text"] == self.__stop_animation__ :
			# the animation will be stopped  
			self.animation_button["text"] = self.__start_animation__
		else :  
			# the animation will be started
			self.animation_button["text"] = self.__stop_animation__ 
			self.update_animation_frame ()


	def update_animation_frame (self) :
		"""
		This methods draws an animation frame if the animation is being played
		"""	
		# Stop the animation if requested
		if self.animation_button["text"] == self.__start_animation__ :
			# Reset the saving of animation
			try : del self.anim_dir_name	
			except AttributeError : pass
			return
		
		# All the subsequent code will be called to continue the animation
		
		# Increment the number of current frame
		self.frame_number.set (self.frame_number.get()+1)
		
		# Consistency check for the current frame number
		if self.frame_number.get() < 0 : self.frame_number.set (0)

		if self.frame_number.get() >= len(self.frame_names)-1 :
			self.frame_number.set (len(self.frame_names)-1) 
			# Stop animation if it is the last frame
			self.animation_button["text"] = self.__start_animation__

		# Plot the current frame
		self.plot_probability_distribution ()
	
		# Set this function to be executed again after 200 Ms
		self.master.after (200, self.update_animation_frame)


	def extract_x_px (self, A) :
		"""
		This method extracts arrays that contain pointers to the data about x, y, px, and py 
		"""
		# Get a pointer to x coordinates
		PD_x = np.frombuffer (A.data, dtype=A.dtype, count=int(A.size/2), offset=0)

		# Get a pointer to x momenta
		PD_px = np.frombuffer (A.data, dtype=A.dtype, count=int(A.size/2), offset=int(A.nbytes/2))

		return [PD_x, PD_px]


	def plot_probability_distribution (self, even=None) :
		"""
		This method plots the probability distribution stored from frame self.frame_number
		"""

		# Retrieving the probability distribution from frame
		data_set = self.HDF5_File[ str(self.frame_names[self.frame_number.get()]) ]
		PD_x_px = data_set[...]
		PD_x_px = np.ascontiguousarray (PD_x_px)

		if PD_x_px.size % 2 <> 0 :
			tkMessageBox.showerror ("Loading probability distribution from file ["+hdf5_file_name+"]...", \
		"Error! A probability distribution is not valid.")
			raise ValueError ('Probability distribution is not valid')

		# Extracting probability distributions for coordinates (x, y) and momenta (px, py)
		PD_x, PD_px = self.extract_x_px (PD_x_px) 

		# If this is the very first frame, then save its extent
		try :
			getattr (self, "first_frame_extent")
		except AttributeError :
			self.first_frame_extent = [PD_x.min(), PD_x.max(), PD_px.min(), PD_px.max()]	

		# Reduce the dimensionality of the arrays, if requested
		step = self.speed_up_step.get()
		if step <> 1:
			PD_x = PD_x[::step]; PD_px = PD_px[::step]; 

		try :
			if step <> 1 : Densities_ = self.Densities[::step]
			else : Densities_ = self.Densities
		except AttributeError : pass
 
		# Plotting
		self.f.clear ()
		
		ax = self.f.add_subplot (111, axisbg='k')

		# Get the requested frame's extent 
		if self.resize_to_first_frame_extent.get () : fig_extent = self.first_frame_extent
		else :	fig_extent = [PD_x.min(), PD_x.max(), PD_px.min(), PD_px.max()]

		# Prepare the corresponding type of plot
		if self.plot_type.get() == self.__type_x_px__ :
			ax.set_xlabel ('$x$')
			ax.set_ylabel ('$p_x$')
			ax.set_title ("Plot of probability distributions $P(x,p_x)$")
			
			# Display the requested style of plot
			if self.plot_style.get() == self.__style_tagged_hexbin__ :
				ax.hexbin (PD_x, PD_px, Densities_, extent=fig_extent)
				
			elif self.plot_style.get() == self.__style_tagged_log_hexbin__	:
				ax.hexbin (PD_x, PD_px, Densities_, bins='log', extent=fig_extent)
		
			elif self.plot_style.get() == self.__style_points__ :
				ax.plot (PD_x, PD_px, '.w', clip_box=Bbox(fig_extent))
		
			elif self.plot_style.get() == self.__style_hexbin__ :
				ax.hexbin (PD_x, PD_px, extent=fig_extent)
		
			elif self.plot_style.get() == self.__style_log_hexbin__ :
				ax.hexbin (PD_x, PD_px, bins='log', extent=fig_extent)

			else : raise NotImplementedError ("Unknown style of plot")

		elif self.plot_type.get() == self.__type_x_histogram__ :
			ax.set_xlabel ('$x$')	
			ax.set_ylabel ('Probability density')
			ax.set_title ("Histogram of reduced probability density $P(x)$")
			ax.hist (PD_x, bins=int(np.sqrt(PD_x.size)), range=[fig_extent[0], fig_extent[1]])

		elif self.plot_type.get() == self.__type_px_histogram__ :
			ax.set_xlabel ('$p_x$')
			ax.set_ylabel ('Probability density')
			ax.set_title ("Histogram of reduced probability density $P(p_x)$")
			ax.hist (PD_px, bins=int(np.sqrt(PD_x.size)), range=[fig_extent[0], fig_extent[1]])

		else : raise NotImplementedError ("Unknown type of plot")  

		# Plot the potential, if needed
		if self.ToPlotPotential.get () and (self.plot_type.get() == self.__type_x_histogram__ \
			or self.plot_type.get() == self.__type_x_px__ ) :

			# Reduce the number of elements in the Potential array, if requested 
			x = np.linspace (fig_extent[0], fig_extent[1], 100)

			# Evaluate the potential by executing the commands in string self.Potential 
			ax_ = ax.twinx()
			ax_.plot (x, eval(self.Potential), 'w-', clip_box=Bbox(fig_extent))
			ax_.set_ylabel ('Potential')
				
		# Save a current frame, if requested 
		if self.to_save_frames.get() :

			# Prepare the directory for storing the animation, if needed  			
			try : getattr(self, "anim_dir_name")
			except AttributeError :
				# Delete the directory with the animations, if it exists
				self.anim_dir_name = "1D_classical_animation"  
				from shutil import rmtree
				rmtree (self.anim_dir_name, ignore_errors=True)		 	
				# Make a directory where figures will be saved
				from os import mkdir
				mkdir (self.anim_dir_name)

			# Finally, saving the figure
			self.f.savefig (self.anim_dir_name + "/classical_pd_" \
					+ str(self.frame_names[self.frame_number.get()]) + ".png")

		self.canvas.show(); self.canvas.get_tk_widget().update_idletasks()	
	

	def pressed_reset_button (self, event=None) :
		"""
		Go-to-the-beginning (or Reset) button was clicked
		"""
		self.frame_number.set (0)


	def __del__ (self) :
		# Closing the HDF5 file
		self.HDF5_File.close ()


if __name__ == '__main__' :
	import sys

	if len(sys.argv) == 2 :
		root = Tk.Tk ()
		root.wm_title ('Plotting classical probability distributions from file [' + sys.argv[1] + ']')
		Classical_Frame_Viewer_1D (master=root, hdf5_file_name=sys.argv[1]).mainloop ()	
	else :
		print '\n' + sys.argv[0] + "\t<the name of the HDF5 file>\n\n\
View snapshots of the quantum dynamics of a 1D classical system stored in the specified HDF5 file\n"		
