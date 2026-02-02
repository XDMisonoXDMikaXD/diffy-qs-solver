import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as ln
from scipy import sparse
from scipy.sparse import linalg as sln

kappa = 0.5

def heatequation2D(time_span, u0, dt, points_interior, dims, b,f=None):
    t_init, t_end = time_span
    px, py = points_interior
    Lx, Ly = dims
    hx, hy = (Lx/(px+1),Ly/(py+1))

    interior = px*py
    
    t = np.arange(t_init, t_end + dt, dt)
    interval = len(t)

    u0 = np.atleast_1d(u0).astype(float)

    u = np.zeros((interior,interval), dtype = float)
    u[:,0] = u0

    b = b.astype(float)
    if f is None:
        f = np.zeros(interior)
    f = f.astype(float)
    
    #Define the 1D Laplacian in x and y
    Kx = sparse.diags([1,-2,1], [-1,0,1], shape = (px,px))
    Ky = sparse.diags([1,-2,1], [-1,0,1], shape = (py,py))

    Ix = sparse.identity(px)
    Iy = sparse.identity(py)

    #Define the 2D discrete Laplacian using the Kronecker product:
    IyKx = sparse.kron(Iy, Kx) 
    KyIx = sparse.kron(Ky, Ix)

    L = IyKx + KyIx

    I = sparse.identity(interior)

    rx, ry = (kappa * dt / hx**2, kappa * dt / hy**2)

    #Applying the Crank-Nicolson scheme.

    B =  I - 0.5*(rx*IyKx + ry*KyIx)
    B = B.tocsc()
    A =   I + 0.5*(rx*IyKx + ry*KyIx)
    A = A.tocsc()

    solver = sparse.linalg.splu(B)

    for i in range(interval - 1):
        rhs = A @ u[:,i] + dt*(b-f) 
        u[:,i+1] = solver.solve(rhs)
        

    return t, u

def build_dirichlet_bc(points_interior, dims, bc):

    py, px = points_interior
    Ly, Lx = dims

    hx = Lx/(px+1)
    hy = Ly/(py+1)

    interior = px * py
    b = np.zeros(interior)

    # Interior grid coordinates
    x = np.linspace(hx, Lx-hx, px)
    y = np.linspace(hy, Ly-hy, py)

    # LEFT boundary (x = 0)
    if bc.get("left") is not None:
        vals = bc["left"](y)
        for j in range(py):
            k = 0 + j*px
            b[k] += vals[j] / hx**2

    # RIGHT boundary (x = Lx)
    if bc.get("right") is not None:
        vals = bc["right"](y)
        for j in range(py):
            k = (px-1) + j*px
            b[k] += vals[j] / hx**2

    # BOTTOM boundary (y = 0)
    if bc.get("bottom") is not None:
        vals = bc["bottom"](x)
        for i in range(px):
            k = i + 0*px
            b[k] += vals[i] / hy**2

    # TOP boundary (y = Ly)
    if bc.get("top") is not None:
        vals = bc["top"](x)
        for i in range(px):
            k = i + (py-1)*px
            b[k] += vals[i] / hy**2

    return b

#Define grid
px,py = (199, 199)
size = (20, 20)

#Define initial conditions
u0_2d = np.zeros((px,py)) #initiate

    #ring of points initial conditions:
u0_2d[px//2, py//2] = 10
u0_2d[px//4, py//4] = 5
u0_2d[3*px//4, 3*py//4] = 5
u0_2d[px//4, 3*py//4] = 5
u0_2d[3*px//4, py//4] = 5
u0_2d[100, 25] = 5
u0_2d[25, 100] = 5
u0_2d[100, 175] = 5
u0_2d[175, 100] = 5


#The initial condition must be flat to input into the function.
u0_flat = u0_2d.ravel()


b0 = build_dirichlet_bc((px,py), size, bc = {
    "left":   lambda y: 0*y,
    "right":  lambda y: 0*y,
    "bottom": lambda x: 0*x,
    "top":    lambda x: 0*x
})
b1 = build_dirichlet_bc((px,py), size, bc = {
    "left":   lambda y: 0*y+10,
    "right":  lambda y: 0*y,
    "bottom": lambda x: 0*x+10,
    "top":    lambda x: 0*x
})

time, sol_flat = heatequation2D((0,1), u0_flat, 0.01, (px,py), size, b = b1)

import matplotlib.animation as animation
from matplotlib import cm

def animate_heat_3d(sol_flat, px, py, frames=None, interval_ms=50, save_path=None):
    """
    Animates a time-dependent 2D heat solution in a 3D surface plot.

    Args:
        sol_flat (np.ndarray): Flattened solution array (39601, 1001).
        px (int): Number of points in X dimension (e.g., 199).
        py (int): Number of points in Y dimension (e.g., 199).
        frames (list/range, optional): Which time indices to animate. Defaults to all.
        interval_ms (int): Delay between frames in milliseconds.
        save_path (str, optional): File path (e.g., 'heat_animation.mp4') to save video.
    """
    if frames is None:
        frames = range(sol_flat.shape[1]) # Animate all time steps

    # 1. Setup figure and 3D axes
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    x = np.linspace(0, 1, px)
    y = np.linspace(0, 1, py)
    X, Y = np.meshgrid(x, y)
    
    # Set fixed limits to prevent axis jumping during animation
    ax.set_zlim(np.min(sol_flat), np.max(sol_flat))
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Temperature (u)')
    ax.set_title(f'Heat Solution Simulation (Time Step: 0/{len(frames)})')

    # Initialize with the first frame's data
    Z = sol_flat[:, frames[0]].reshape((py, px))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.inferno, rcount=100, ccount=100)

    def update(frame_index):
        nonlocal surf
        # Remove the old surface plot
        if surf:
            surf.remove()
        
        # Get the new Z data from the correct time column and reshape
        current_Z = sol_flat[:, frame_index].reshape((py, px))
        
        # Plot the new surface
        surf = ax.plot_surface(X, Y, current_Z, cmap=cm.inferno, rcount=100, ccount=100)
        
        # Update the title with the current time step
        ax.set_title(f'Heat Solution Simulation (Time Step: {frame_index+1}/{len(frames)})')
        
        # Return the artist list required by FuncAnimation
        return [surf]

    # Create the animation object
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval_ms, blit=False, repeat=True
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        # Requires ffmpeg or similar backend installed separately
        ani.save(save_path, writer='ffmpeg', fps=30)
        print("Save complete.")
    else:
        plt.show()
    
    return ani

# Example Usage (You would use your actual sol_flat data):
# Assume you have sol_flat, px, py defined from your solver
ani = animate_heat_3d(sol_flat, px, py, interval_ms=0.01) 


