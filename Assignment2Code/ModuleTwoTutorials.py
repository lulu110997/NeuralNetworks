import numpy as np
import matplotlib.pyplot as plt 
import math 
import time

def FourPointOne(): # To fix the indexing issue with the cycle error plot
	x1 = np.array([[10, 2, -1]]).T
	x2 = np.array([[2, -5, -1]]).T
	x3 = np.array([[-5, 5, -1]]).T
	x4 = np.array([[-2, -5, -1]]).T
	x = np.hstack([x1,x2,x3,x4]) # Each input is stored in a matrix as a column vector
	d1 = np.array([[1,-1,-1,-1]]).T
	d2 = np.array([[-1,1,-1,-1]]).T
	d3 = np.array([[-1,-1,1,-1]]).T
	d4 = np.array([[-1,-1,-1,1]]).T
	d = np.hstack([d1, d2, d3, d4]) # Each desired response is stored as a matrix as a column vector
	w = np.array([[1,2,1], [2, -1, 2], [1, 3, -1], [2, 4, 0]], dtype='float16') # Create a matrix where each row is a weight vector
	c = 0.1
	inputIndex = 0
	ce = np.zeros((19,1))
	e = 0
	cyclesCount = 0
			

	for steps in range(0,72):
		# Step 1: Feed in the first pattern input into all four neurons and compare the neuron output to the desired output.
		# Use the fixed correction rule for the neurons that output the wrong classification
		z = np.zeros((1,4)) # Create an array to store the output of the activation neuron
		for weightIndex in range(4): 
			z[:,weightIndex] = (np.sign(np.matmul(w[weightIndex],x[:,inputIndex][np.newaxis].T))) # Feed the input in each neuron
			if z[:,weightIndex] != d[inputIndex][weightIndex]: # Compare if the linear classifier output is correct. If not, use the fixed correction rule
				# Need to reshape some np arrays into a column vector, hence the transpose operations. 
				# The newaxis is used to turn a 1D array into a 2D array so we can transpose the array 
				w[weightIndex] = (w[weightIndex][np.newaxis].T + 0.5*c*(d[inputIndex][weightIndex]-z[:,weightIndex])*x[:,inputIndex][np.newaxis].T).T
	
		if inputIndex < 3: 
		# Use the next pattern input. Else, recycle the pattern once the last input has been reached.
		# Calculate the pattern error using the updated weight values for this cycle (note that one cycle is when 
		# all four inputs have been fed into the linear classifier)
			inputIndex += 1
		else: 
			inputIndex = 0
			for inputIndexErrCalc in range(0,4):
				for weightIndexErrCalc in range(0,4):
					z[:,weightIndexErrCalc] = (np.sign(np.matmul(w[weightIndexErrCalc],x[:,inputIndexErrCalc][np.newaxis].T)))
					e = e + 0.5*((d[inputIndexErrCalc][weightIndexErrCalc]-z[:,weightIndexErrCalc]) ** 2)
			ce[cyclesCount+1,0] = e # Save the cycle error in the array for plotting
			e = 0 # Reset the cycle error
			cyclesCount += 1

	print("Terminating training after ", steps+1, "steps. Final weight vector is \n")
	print(w)
	ceFig, ceAxes = plt.subplots()  
	ceAxes.set_title("Cycle Error")  
	ceAxes.grid(True, which='both') 
	ceAxes.plot(np.arange(1,19),ce[1:19,0]) 
	ceAxes.set_xlabel('Cycles')  
	ceAxes.set_ylabel('Cycle Error)')
	plt.show()

def FivePointTwo():
	x = np.array([[1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1],
		[-1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1 ,-1 ,-1],
		[1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1]])
	d = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
	n = 0.8

	# Weight of the hidden layer
	w = np.array([[-0.3124, 0.6585, 0.2860, -0.4668],
		[0.0834, -0.9701, 0.0923, 0.9402],
		[0.0456, -0.4516, 0.8357, -0.5065]])

	# Weight of the first layer (input weights)
	wp = np.array([[0.6880, 0.6342, 0.4139, 0.1648, 0.3914, -0.6576, -0.8402, 0.5660, -0.9873, -0.8663, -0.5683, 0.3181, -0.4394, 0.3722, -0.7924, -0.5586, -0.6795],
		[0.3880, -0.9558, 0.4157, 0.5034, -0.4410, -0.9915, -0.6738, 0.4934, 0.1696, 0.0880, -0.2285, 0.4186, 0.6319, -0.3797, 0.0419, -0.8894, 0.4475],
		[-0.0885, -0.6785, -0.1267, 0.9831, 0.5077, -0.0086, 0.8066, -0.1271, 0.0370, -0.2205, 0.2537, 0.0773, -0.2562, -0.9813, 0.5518, -0.8587, -0.4060]])
	for cycle in range(0,3):
		vp = np.matmul(wp,x[cycle][np.newaxis].T) # Calculate the first neuron input. The resulting value of the multiplication is dependent on the input matrix sizes
		y = (1-np.exp(-vp))/(1+np.exp(-vp)) # Feed it into the activation function. Need to use np.exp because math.exp only takes in a single value (not an array)
		dy = 0.5*(1-y**2) # Calculate the slope

		# Repeat the steps above for the second layer
		# Need to augment the bias into the hidden layer's output, hence use np.append. 
		# Add a newaxis since append output is a 1D row vector array
		v = np.matmul(w,np.append(y, -1)[np.newaxis].T) 
		z = (1-np.exp(-v))/(1+np.exp(-v))
		dz = 0.5*(1-z**2)	

		r = d[cycle] - z.T # Calculate the error. Note how we don't need a newaxis since the resultant operations yielded 2D arrays
		delta = r.T*dz

		# Calculate the weight increment
		dp = np.matmul(delta.T,w[:,0:3])*dy.T # Error signal of output layer
		deltawp =n*dp.T*x[cycle,:]

		# Weight modifications
		w  = w + n*delta*np.append(y, -1).T
		wp = wp+deltawp

		print("the input weight is, \n", w, "\n for cycle number, ", cycle+1, "\n")
		print("the hidden layer weight is, \n", wp, "\n for cycle number, ", cycle+1, "\n")

	for cycle in range(0,3):
		vp = np.matmul(wp,x[cycle][np.newaxis].T) # Calculate the first neuron input. The resulting value of the multiplication is dependent on the input matrix sizes
		y = (1-np.exp(-vp))/(1+np.exp(-vp)) # Feed it into the activation function. Need to use np.exp because math.exp only takes in a single value (not an array)

		# Repeat the steps above for the second layer
		# Need to augment the bias into the hidden layer's output, hence use np.append. 
		# Add a newaxis since append output is a 1D row vector array
		v = np.matmul(w,np.append(y, -1)[np.newaxis].T) 
		z = (1-np.exp(-v))/(1+np.exp(-v))
		print("actual outputs of the network with continuous perceptrons: \n", z, "\n")

	for cycle in range(0,3):
		vp = np.matmul(wp,x[cycle][np.newaxis].T) # Calculate the first neuron input. The resulting value of the multiplication is dependent on the input matrix sizes
		y = np.sign(vp) # Feed it into the activation function. Need to use np.exp because math.exp only takes in a single value (not an array)

		# Repeat the steps above for the second layer
		# Need to augment the bias into the hidden layer's output, hence use np.append. 
		# Add a newaxis since append output is a 1D row vector array
		v = np.matmul(w,np.append(y, -1)[np.newaxis].T) 
		z = np.sign(v)
		print("outputs of the network with discrete perceptrons: \n", z, "\n")

# FourPointOne()
# FivePointTwo()