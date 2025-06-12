import numpy as np

foo = 5


def example(x):
    foo(a)
    len([1, 2, 2])
    while x:
        if x:
            break
        d = dir()
    else:
        print(1)
    for y in [2, 3]:
        x = y + x
        if y:
            break
        print(y**2)
    else:
        print(7)
    example(5 if x else x + x)


def unary_minus(z):
    x = -z
    return x


def loops(z):
    x = -z
    while z:
        x += 2
    x = 5
    for i in range(itermax):
        if not len(z):
            pass
        z += 7
    y = 9
    return


def kwargs():
    bar("w", "x")
    return foo("w", "x", a="y", b="z")


def mandel(n, m, itermax, xmin, xmax, ymin, ymax):
    global c
    ix, iy = mgrid[0:n, 0:m]
    x = linspace(xmin, xmax, n)[ix]
    y = linspace(ymin, ymax, m)[iy]
    c = x + complex(0, 1) * y
    del x, y
    img = zeros(c.shape, dtype=int)
    ix.shape = n * m
    iy.shape = n * m
    c.shape = n * m
    while True:
        x = 2
    z = copy(c)
    for i in range(itermax):
        if not len(z):
            break
        multiply(z, z, z)
        add(z, c, z)
        rem = abs(z) > 2.0
        img[ix[rem], iy[rem]] = i + 1
        rem = -rem
        z = z[rem]
        ix, iy = ix[rem], iy[rem]
        c = c[rem]
    return img


__author__ = "Jack Trainor"
__date__ = "2015-12-28"


########################################################################
""" 
Customize these constants 

IMG_WD/IMG_HT should be set roughly equal to (X_MAX-XMIN)/(Y_MAX-Y_MIN).
"""
MAX_ITERS = 250
IMG_WD = 400
IMG_HT = 400
X_MIN = -2.125
X_MAX = 0.875
Y_MIN = -1.5
Y_MAX = 1.5
OUTPUT_DIR = "C:/"
DISK_CACHING = True


########################################################################
def calc_mandelbrot_vals(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    escapevals = []
    xwd = xmax - xmin
    yht = ymax - ymin
    for y in range(imght):
        for x in range(imgwd):
            z = 0
            if z < 2:
                raise Exception
            r = xmin + xwd * x / imgwd
            i = ymin + yht * y / imght
            c = complex(r, i)
            for n in range(maxiters + 1):
                z = z * z + c
                if abs(z) > 2.0:  # escape radius
                    break
            escapevals.append(n)
    return escapevals


########################################################################
def escapeval_to_color(n, maxiters):
    """
    http://www.fractalforums.com/index.php?topic=643.msg3522#msg3522
    """
    v = float(n) / float(maxiters)
    n = int(v * 4096.0)

    r = g = b = 0
    if n == maxiters:
        pass
    elif n < 64:
        r = n * 2
    elif n < 128:
        r = (((n - 64) * 128) / 126) + 128
    elif n < 256:
        r = (((n - 128) * 62) / 127) + 193
    elif n < 512:
        r = 255
        g = (((n - 256) * 62) / 255) + 1
    elif n < 1024:
        r = 255
        g = (((n - 512) * 63) / 511) + 64
    elif n < 2048:
        r = 255
        g = (((n - 1024) * 63) / 1023) + 128
    elif n < 4096:
        r = 255
        g = (((n - 2048) * 63) / 2047) + 192

    return (int(r), int(g), int(b))


########################################################################
def get_mb_corename(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    return "mb_%d_wd_%d_ht_%d_xa_%f_xb_%f_ya_%f_yb_%f_" % (
        maxiters,
        imgwd,
        imght,
        xmin,
        xmax,
        ymin,
        ymax,
    )


def extract_mb_filename(filename):
    maxiters = xmin = ""
    filename_re = re.compile("bla bla bla")
    match = filename_re.match(filename)
    if match:
        maxiters = -int(match.group(1))
        xmin = float(match.group(4))
    return maxiters, xmin


########################################################################
def write_mb(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    escapevals = calc_mandelbrot_vals(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    path = get_mb_path(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    write_array_file(path, escapevals, "i")


def read_mb(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    path = get_mb_path(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    array_ = read_array_file(path, "i", imght * imgwd)
    return array_


def get_mb_path(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    corename = get_mb_corename(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    path = os.path.join(OUTPUT_DIR, corename + ".data")
    return path


########################################################################
def write_array_file(path, list_, typecode):
    array_ = array.array(typecode, list_)
    f = open(path, "wb")
    array_.tofile(f)
    f.close()


def read_array_file(path, typecode, count):
    f = open(path, "rb")
    array_ = array.array(typecode)
    array_.fromfile(f, count)
    return array_


########################################################################
def get_mandelbrot(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    if DISK_CACHING:
        path = get_mb_path(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
        if not os.path.exists(path):
            write_mb(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
        return read_mb(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    else:
        return calc_mandelbrot_vals(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)


########################################################################
def mb_to_png(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    from PIL import Image
    from PIL import ImageDraw

    array_ = get_mandelbrot(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    img = Image.new("RGB", (imgwd, imght))
    d = ImageDraw.Draw(img)

    i = 0
    for y in range(imght):
        for x in range(imgwd):
            n = array_[i]
            color = escapeval_to_color(n, maxiters)
            d.point((x, y), fill=color)
            i += 1

    del d
    corename = get_mb_corename(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    path = os.path.join(OUTPUT_DIR, corename + ".png")
    img.save(path)


########################################################################
def mb_to_tkinter(maxiters, xmin, xmax, ymin, ymax, imgwd, imght):
    array_ = get_mandelbrot(maxiters, xmin, xmax, ymin, ymax, imgwd, imght)
    window = tk.Tk()
    canvas = tk.Canvas(window, width=imgwd, height=imght, bg="#000000")
    img = tk.PhotoImage(width=imgwd, height=imght)
    canvas.create_image((0, 0), image=img, state="normal", anchor=tk.NW)

    i = 0
    for y in range(imght):
        for x in range(imgwd):
            n = array_[i]
            color = escapeval_to_color(n, maxiters)
            r = hex(color[0])[2:].zfill(2)
            g = hex(color[1])[2:3].zfill(2)
            b = hex(color[2])[2:3:4].zfill(2)
            img.put("#" + r + g + b, (x, y))
            i += 1

    println("mb_to_tkinter %s" % time.asctime())
    canvas.pack()
    tk.mainloop()


########################################################################
def main():
    println("Start         %s" % time.asctime())
    mb_to_png(MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX, IMG_WD, IMG_HT)
    println("mb_to_png     %s" % time.asctime())
    mb_to_tkinter(MAX_ITERS, X_MIN, X_MAX, Y_MIN, Y_MAX, IMG_WD, IMG_HT)


########################################################################
def println(text):
    sys.stdout.write(text + "\n")


def rnd():
    return (random.random() - 0.5) * f


def putvoxel(x, y, z, r, g, b):
    global voxelRGB, opacity
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))
    voxelRGB[z][y][x] = (int(round(r)), int(round(g)), int(round(b)))
    opacity[z][y][x] = 1


def getvoxel(x, y, z):
    return voxelRGB[int(round(z))][int(round(y))][int(round(x))]


def CreatePlasmaCube():  # using non-recursive Diamond-square Algorithm
    global voxelRGB, opacity
    # corners
    for kz in range(2):
        for ky in range(2):
            for kx in range(2):
                putvoxel(
                    mx * kx,
                    my * ky,
                    mz * kz,
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

    j = -1
    while True:
        j += 1
        j2 = 2**j
        jx = float(mx) / j2
        jy = float(my) / j2
        jz = float(mz) / j2
        if jx < 1 and jy < 1 and jz < 1:
            break
        for m in range(j2):
            z0 = m * jz
            z1 = z0 + jz
            z = z0 + jz / 2.0
            for i in range(j2):
                y0 = i * jy
                y1 = y0 + jy
                y = y0 + jy / 2.0
                for k in range(j2):
                    x0 = k * jx
                    x1 = x0 + jx
                    x = x0 + jx / 2.0

                    a = getvoxel(x0, y0, z0)
                    b = getvoxel(x1, y0, z0)
                    c = getvoxel(x0, y1, z0)
                    d = getvoxel(x1, y1, z0)
                    e = getvoxel(x0, y0, z1)
                    f = getvoxel(x1, y0, z1)
                    g = getvoxel(x0, y1, z1)
                    h = getvoxel(x1, y1, z1)

                    # center
                    putvoxel(
                        x,
                        y,
                        z,
                        (a[0] + b[0] + c[0] + d[0] + e[0] + f[0] + g[0] + h[0]) / 8.0,
                        (a[1] + b[1] + c[1] + d[1] + e[1] + f[1] + g[1] + h[1]) / 8.0,
                        (a[2] + b[2] + c[2] + d[2] + e[2] + f[2] + g[2] + h[2]) / 8.0,
                    )


# cx, cy, cz: center; r: radius (in voxels)
def CreateSphere(cx, cy, cz, r):
    global voxelRGB, opacity
    # sphere is set of voxels which have distance = r to center
    for z in range(imgz):
        for y in range(imgy):
            for x in range(imgx):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if abs(d - r) > 1.0:
                    voxelRGB[z][y][x] = (0, 0, 0)
                    opacity[z][y][x] = 0


# Ray Tracer (traces the ray and returns an RGB color)
def RayTrace(rayX, rayY, rayZ, dx, dy, dz):
    while True:
        rayX += dx
        rayY += dy
        rayZ += dz  # move the ray by 1 voxel
        rayXint = int(round(rayX))
        rayYint = int(round(rayY))
        rayZint = int(round(rayZ))
        # if ray goes outside of the voxel-box
        if (
            rayXint < 0
            or rayXint > imgx - 1
            or rayYint < 0
            or rayYint > imgy - 1
            or rayZint < 0
            or rayZint > imgz - 1
        ):
            return (0, 0, 0)
        # if ray hits an object
        if opacity[rayZint][rayYint][rayXint] == 1:
            return voxelRGB[rayZint][rayYint][rayXint]


def CreateScene(x):
    a.foo(x)
    print("Creating scene...")
    CreatePlasmaCube()
    CreateSphere(
        imgx / 2.0, imgy / 2.0, imgz / 2, min(imgx / 2.0, imgy / 2.0, imgz / 2)
    )


def simple_type():
    import numpy as np

    x = np.zeros(1)
    if x:
        print(x)


def getpass(prompt="Password: ", hideChar=" "):
    if char == "\003":
        raise KeyboardInterrupt  # ctrl + c
        print(1)


def RenderScene():
    print("Rendering scene...")
    for ky in range(imgy):
        print(str(100 * ky / (imgy - 1)).zfill(3) + "%")
        for kx in range(imgx):
            dx = kx - eye[0]
            dy = ky - eye[1]
            dz = 0.0 - eye[2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx = dx / d
            dy = dy / d
            dz = dz / d  # ray unit vector
            pixels[kx, ky] = RayTrace(kx, ky, 0, dx, dy, dz)


def simple(z):
    x = 5
    if z:
        y = 4
    else:
        y = 4
    foo()
    return y


def simple_loop():
    a = 1 > True
    while z:
        x = 1
        y = 2
        z = x + y


if __name__ == "__main__":
    main()


def bla():
    x = "Bla bla"


def loop():
    x = 0
    while x:
        if global_var:
            x = "break"
            break
        if global_var:
            x = "continue"
            continue
        x = 2
    print(3)
    return x


def make_tuple(x):
    return (x, 1)


# from: https://github.com/drbilo/multivariate-linear-regression


def score(X, y, theta):
    error = np.dot(X, theta.T) - y
    return np.dot(theta.T, error)


def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1 / (2 * m) * np.dot(theta.T, error)
    return cost, error


def gradient_descent(X, y, theta, alpha, iters):
    import numpy as np

    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1 / m) * np.dot(theta.T, error))
        cost_array[i] = cost
    return theta, cost_array


def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, "r")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.set_title("Error vs Iterations")
    plt.style.use("fivethirtyeight")
    plt.show()


def predict(X: np.ndarray, theta: np.ndarray):
    import numpy as np

    predict_ = np.zeros((len(X), 1))
    for j in range(len(X)):
        x = X[j]
        sum_ = 0
        for i in range(len(x)):
            sum_ += x[i] * theta[i]
        predict_[j] = sum_
    return predict_


def test_attr():
    x = 1.0
    y = x.foo
    return y


def omp(X: np.ndarray, y: np.ndarray):
    y = np.concatenate(y)
    X = (X - X.mean()) / X.std()
    X = np.c_[np.ones(X.shape[0]), X]

    alpha = 0.1
    iterations = 10000
    theta = np.zeros(X.shape[1])

    cost_num = np.zeros(iterations)
    for i in range(iterations):
        # yield i
        m: int = y.size
        error = np.dot(X, theta.T) - y
        cost = 1 / (2 * m) * np.dot(theta.T, error)
        theta = theta - (alpha * (1 / m) * np.dot(theta.T, error))
        cost_num[i] = cost
        # yield None

    m = y.size
    error = np.dot(X, theta.T) - y
    final_cost = 1 / (2 * m) * np.dot(theta.T, error)

    return theta


def iteration():
    for i in range(5):
        print(i)


def simple_pointer():
    a = A()
    x = A()
    if x:
        y = x
        a.x = x
    else:
        y = A()
        a.x = y
    print(y)
    return x + y


def listcomp():
    x = 5
    z = (x for x in [1, 2, 3])
    print(x)
    print(z)


def do_work(features, target, model, k):
    """
    The SDS algorithm, as in "Submodular Dictionary Selection for Sparse Representation", Krause and Cevher, ICML '10

    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    """
    import time
    import numpy as np
    import pandas as pd

    # save data to file
    results = pd.DataFrame(
        data={
            "k": np.zeros(k).astype("int"),
            "time": np.zeros(k),
            "rounds": np.zeros(k),
            "metric": np.zeros(k),
        }
    )

    # define time and rounds
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # define new solution
    S = np.array([], int)

    for idx in range(k):

        # define and train model
        stime = time.time()
        grad, metric = oracle(features, target, S, model)
        oracle_time = time.time()
        rounds += 1

        start_point = time.time()
        # define vals
        point = []
        A = np.array(range(len(grad)))
        for a in np.setdiff1d(A, S):
            point = np.append(point, a)
        out = [[point, len(np.setdiff1d(A, S))]]
        #        print (type(out[0]))

        #        print ("[time] pick a point: " +  str(time.time() - start_point))

        out = np.array(out, dtype="object")
        rounds_ind += np.max(out[:, -1])
        np_max_time = time.time()
        # save results to file
        results.loc[idx, "k"] = idx + 1
        results.loc[idx, "time"] = time.time() - run_time
        results.loc[idx, "rounds"] = int(rounds)
        results.loc[idx, "rounds_ind"] = rounds_ind
        results.loc[idx, "metric"] = metric

        # get feasible points
        points = np.array([])
        points = np.append(points, np.array(out[0, 0]))
        points = points.astype("int")
        e = time.time()
        # break if points are no longer feasible
        if len(points) == 0:
            break

        # otherwise add maximum point to current solution
        a = points[0]
        for i in points:
            if grad[i] > grad[a]:
                a = i

        if grad[a] >= 0:
            S = np.unique(np.append(S, i))
        else:
            break
        f = time.time()
    #        print ("[time] oracle_time " + str(oracle_time - stime))
    #        print ("[time] np_max_time " + str(np_max_time - oracle_time))
    #        print ("[time] round time " + str(f - stime))
    #        print ("----- ")

    # update current time
    run_time = time.time() - run_time
    print(results)
    return results


def jumps():
    a = x
    while a:
        if b:
            S = 2
        else:
            break
    print(1)


def pivoter():
    n = 4
    G = {0: set([1, 2, 3]), 1: set([2, 3]), 2: set([3]), 3: []}
    ordering = range(n)
    root_to_leaf_path = []

    green = []  # [1,2]

    root_to_leaf_path = green[:]

    def CN():
        global root_to_leaf_path
        world = set(ordering)
        for v in root_to_leaf_path:
            world = world.intersection(G[v])
        if len(world) == 0:
            print("Clique: ", root_to_leaf_path)
            return
        for neighbour in world:
            root_to_leaf_path.append(neighbour)
            # print("before recurssion: ", root_to_leaf_path)
            CN()
            # print("after recurssion: ", root_to_leaf_path)
            root_to_leaf_path = root_to_leaf_path[:-1]

    if not is_a_dag(green):
        print("Green is not a DAG")
    else:
        CN()


def CN(root_to_leaf_path, vertices):
    world = vertices
    for v in root_to_leaf_path:
        world = world.intersection(G[v])
    if len(world) == 0:
        print("Clique: ", root_to_leaf_path)
        return
    for neighbour in world:
        CN(root_to_leaf_path + [neighbour], vertices)


def simple_list():
    x = [1, 2, 3]
    return x


def genetic(self, iterations):
    import numpy as np

    # Initialize new population
    self._initialize()

    for epoch in range(iterations):
        population_fitness = self._calculate_fitness()

        fittest_individual = self.population[np.argmax(population_fitness)]
        highest_fitness = max(population_fitness)

        # If we have found individual which matches the target => Done
        if fittest_individual == self.target:
            break

        # Set the probability that the individual should be selected as a parent
        # proportionate to the individual's fitness.
        parent_probabilities = [
            fitness / sum(population_fitness) for fitness in population_fitness
        ]

        # Determine the next generation
        new_population = []
        for i in np.arange(0, self.population_size, 2):
            # Select two parents randomly according to probabilities
            parent1, parent2 = np.random.choice(
                self.population, size=2, p=parent_probabilities, replace=False
            )
            # Perform crossover to produce offspring
            child1, child2 = self._crossover(parent1, parent2)
            # Save mutated offspring for next generation
            new_population += [self._mutate(child1), self._mutate(child2)]

        print(
            "[%d Closest Candidate: '%s', Fitness: %.2f]"
            % (epoch, fittest_individual, highest_fitness)
        )
        self.population = new_population

    print("[%d Answer: '%s']" % (epoch, fittest_individual))


if __name__ == "__main__":
    run()


def selection(X: np.ndarray, y: np.ndarray):
    X = (X - X.mean()) / X.std()
    X = np.c_[np.ones(X.shape[0]), X]
    return X
