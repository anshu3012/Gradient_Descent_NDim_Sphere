import numpy as np 
import math 

# the numper of dimneions in our hyperspace 
dimensions = 5 

# the number of points from which the arc lengths will be calculated
pts = 20
epochs =  1000
learning_rate = 0.001

# a vector to hold the gradients in each dimension and all the points
grad_vec = np.empty([dimensions, pts], dtype = float)

# a function to produce points on a sphere in n dimensions
# takes input as the number of points we need and the number of dimensions 
# returns a vector which stores the points column-wise
def sample_spherical(npoints, dim):
    vec = np.random.randn(dim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

# array to store random points on sphere 
points_array = sample_spherical(pts, dimensions)

# array to store the point which we need. it is also initialized
parameters = sample_spherical(1,dimensions)
parameters = np.reshape(parameters,(5,))



for i in range (epochs):  
    #iterating over all points  
    for j in range (pts):
        # gradient of acos(a.b) where a is the point we need to find.So find grad(cos(a.b)) wrt (a_x,a_y,a_z....)
        # grad(acos(a.b)) =  (-1/(sqrt(1-(a.b)**2))) * Partial_der(ab,wrt a)
        # grad(acos(a.b)) =  (-1/(sqrt(1-(a.b)**2))) * b
        grad_vec[:,j] = (-1/(np.sqrt(1-(np.dot(parameters, points_array[:,j]))**2)))*(points_array[:,j])

    # Now we have a gradient vector however that gradient is not along the surface of the sphere 
    # So we find a plane tangent to the surface of the sphere at a point (our parameter) and project the gradient along the surface 
    # The normal to the sphere (and the plane) at any point will be the gradient of the level surface i.e. the sphere 
    # Let us say that u is the gradient vector and v is the point (our parameters) where the gradient is calculated 
    # We can decompose u in such a way that it will have a component along the normal of plane i.e collinear comp and another component along the plane 
    # To get the compnent along the plane we subtract the u's collinear component from the main u vector 
    # Component of u along plane =  u - (u.v)*(unit v vector)
    # Now use this projected component of u (gradient vector) to tune our parameters

    summed_grad_vec = (np.sum(grad_vec, axis = 1))

    # gradient of level surfcae x^2+y^2+z^2.....
    normal_vec_to_plane = 2*(parameters) 

    # projection_of_grad_on_plane = summed_grad_vec - np.dot((np.dot(summed_grad_vec,normal_vec_to_plane)),normal_vec_to_plane) 
    projection_of_grad_on_plane = summed_grad_vec - (np.dot(summed_grad_vec,normal_vec_to_plane))*normal_vec_to_plane

    new_grad = projection_of_grad_on_plane
    parameters =  parameters - learning_rate*new_grad

# The point we got will be a little above the surface of the sphere, so we bring it back on the unit sphere by normalizing it 
parameters =  parameters/(np.sum(np.square(parameters)))
print(parameters)
# making sure the points lie on unit sphere 
print(np.sum(np.square(parameters)))
