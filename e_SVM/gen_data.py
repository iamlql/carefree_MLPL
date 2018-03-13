import numpy as np
def gen_random(size = 100):
	xy = np.random.rand(size,2)
	z = np.random.randint(2, size = size)
	z[z==0] = -1
	return xy, z

def gen_xor(size = 100):
	x = np.random.randn(size)
	y = np.random.randn(size)
	z = np.ones(size)
	z[x*y < 0] = -1
	return np.c_[x,y].astype(np.float32), z

def gen_spin(size = 30):
	xs = np.zeros((size*4,2),dtype = np.float32)
	ys = np.zeros(size*4, dtype = np.int8)
	for i in np.arange(4):
		ix = range(size*i, size*(i+1))
		r = np.linspace(0.0, 1, size+1)[1:]
		t = np.linspace(2*i*np.pi/4,2*(i+4)*np.pi/4, size)+np.random.random(size = size)*0.1
		xs[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		ys[ix] = 2*(i%2)-1
	return xs, ys

def gen_linspace(scale = 1.0, bias = 0.0, size = 100):
	xyp = np.random.rand(size, 2)*scale+bias
	xyn = -1.0*np.random.rand(size,2)*scale+bias
	xy = np.vstack((xyp, xyn))
	z = np.vstack((np.ones((size,1)), -1*np.ones((size,1))))
	return np.random.permutation(np.hstack((xy, z)))
	
if __name__ == '__main__':
	xy = gen_linspace(100,100)
	print (np.random.permutation(xy))
	# print z


