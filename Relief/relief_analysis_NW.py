import numpy as np
import os
from progressbar import ProgressBar
from multiprocessing import Pool


def find_grid(coord, wh_x, wh_y, cellsize):
    gridd = []
    for i in range(0, len(coord)):  # grid of 4 points
        if np.fabs(wh_x - coord[i][0]) <= cellsize and np.fabs(wh_y - coord[i][1]) <= cellsize:
            gridd.append(coord[i])
    return gridd


def read_file(file):
    with open(file, 'r') as f:  # read xyz file to a matrix
        x = f.readlines()
        coord = np.zeros([len(x), 3])
        for a in range(len(x)):
            lines = x[a].split(',')
            for b in range(0, 3):
                coord[a][b] = lines[b]
    return coord


def find_z(grid, x, y):
    for i in range(len(grid)):
        if grid[i][0] == x:
            if grid[i][1] == y:
                zet = grid[i][2]
    return zet


def find_xyz(grid):
    x = [grid[0][0], grid[1][0], grid[2][0], grid[3][0]]
    y = [grid[0][1], grid[1][1], grid[2][1], grid[3][1]]
    x1 = np.min(x)
    y1 = np.min(y)
    x2 = x1
    y2 = np.max(y)
    x3 = np.max(x)
    y3 = y2
    x4 = x3
    y4 = y1
    z1 = find_z(grid, x1, y1)
    z2 = find_z(grid, x2, y2)
    z3 = find_z(grid, x3, y3)
    z4 = find_z(grid, x4, y4)
    return [[x1, x2, x3, x4],
            [y1, y2, y3, y4],
            [z1, z2, z3, z4]]


# interpolacja IDW, na podstawie znalezionych i uporządkowanych 4 najbliższych punktów o znanej wysokości
def IDW(file, x0, y0, e, cell):
    grid = find_grid(file, x0, y0, cell)
    a = np.array(find_xyz(grid))
    u = (x0 - a[0, 0]) / cell
    v = (y0 - a[1, 0]) / cell
    u_ = 1 - u
    v_ = 1 - v
    d1 = np.sqrt(u ** 2 + v ** 2) + e
    d2 = np.sqrt(u_ ** 2 + v ** 2) + e
    d3 = np.sqrt(u_ ** 2 + v_ ** 2) + e
    d4 = np.sqrt(u ** 2 + v_ ** 2) + e
    z0 = (a[2, 0] / d1 ** 2 + a[2, 1] / d2 ** 2 + a[2, 2] / d3 ** 2 + a[2, 3] / d4 ** 2) / (1 / d1 ** 2 + 1 / d2 ** 2 + 1 / d3 ** 2 + 1 / d4 ** 2)
    return z0


# znalezienie zredukowanej odległości między kołami ze względu na nachylenie poprzeczne terenu
def find_ne_distance_btw_wheels(xt1, yt1, zt1, track_wdt):
    x0t2 = xt1 + track_wdt * np.sqrt(2)/2
    y0t2 = yt1 + track_wdt * np.sqrt(2)/2
    z0t2 = IDW(file_arr, x0t2, y0t2, e, cell)
    dH = z0t2 - zt1
    d12 = track_wdt * np.cos(np.arctan(dH / track_wdt))
    return d12

# znalezienie zredukowanej odległości między kołami ze względu na nachylenie poprzeczne terenu
def find_sw_distance_btw_wheels(xt1, yt1, zt1, track_wdt):
    x0t2 = xt1 - track_wdt * np.sqrt(2)/2
    y0t2 = yt1 - track_wdt * np.sqrt(2)/2
    z0t2 = IDW(file_arr, x0t2, y0t2, e, cell)
    dH = z0t2 - zt1
    d12 = track_wdt * np.cos(np.arctan(dH / track_wdt))
    return d12

# znalezienie zredukowanego rozstawu osi ze względu na nachylenie terenu
def find_se_distance_btw_wheels(xt1, yt1, zt1, wheelbase):
    x0t2 = xt1 + wheelbase * np.sqrt(2)/2
    y0t2 = yt1 - wheelbase * np.sqrt(2)/2
    z0t2 = IDW(file_arr, x0t2, y0t2, e, cell)
    dH = z0t2 - zt1
    d12 = wheelbase * np.cos(np.arctan(dH / wheelbase))
    return d12

# znalezienie zredukowanego rozstawu osi ze względu na nachylenie terenu
def find_nw_distance_btw_wheels(xt1, yt1, zt1, wheelbase):
    x0t2 = xt1 - wheelbase * np.sqrt(2)/2
    y0t2 = yt1 + wheelbase * np.sqrt(2)/2
    z0t2 = IDW(file_arr, x0t2, y0t2, e, cell)
    dH = z0t2 - zt1
    d12 = wheelbase * np.cos(np.arctan(dH / wheelbase))
    return d12


# obliczenie parametrów równania ogólnego powierzchni
def plane_equation(plane):
    a1 = plane[1, 0] - plane[0, 0]
    a2 = plane[1, 0] - plane[2, 0]
    b1 = plane[1, 1] - plane[0, 1]
    b2 = plane[1, 1] - plane[2, 1]
    c1 = plane[1, 2] - plane[0, 2]
    c2 = plane[1, 2] - plane[2, 2]
    ab1_ab2 = [b1 * c2 - c1 * b2, c1 * a2 - c2 * a1, a1 * b2 - b1 * a2]
    A = ab1_ab2[0]
    B = ab1_ab2[1]
    C = ab1_ab2[2]
    D1 = -1 * (A * plane[0, 0] + B * plane[0, 1] + C * plane[0, 2])
    D2 = D1 - ride_hgt * np.sqrt(A ** 2 + B ** 2 + C ** 2)
    return A, B, C, D1, D2


def plane_equation_reverse(plane):
    a1 = plane[1, 0] - plane[0, 0]
    a2 = plane[1, 0] - plane[2, 0]
    b1 = plane[1, 1] - plane[0, 1]
    b2 = plane[1, 1] - plane[2, 1]
    c1 = plane[1, 2] - plane[0, 2]
    c2 = plane[1, 2] - plane[2, 2]
    ab1_ab2 = [b1 * c2 - c1 * b2, c1 * a2 - c2 * a1, a1 * b2 - b1 * a2]
    A = ab1_ab2[0]
    B = ab1_ab2[1]
    C = ab1_ab2[2]
    D1 = -1 * (A * plane[0, 0] + B * plane[0, 1] + C * plane[0, 2])
    D2 = ride_hgt * np.sqrt(A ** 2 + B ** 2 + C ** 2) + D1
    return A, B, C, D1, D2


# obliczenie zredukowanych wartości boków pojazdu - z przodu, z tyłu i z boku kół
def reduced_dimensions(xt1, yt1, zt1, wheelbase, track_wdt, side, front, back):
    d_nw = find_nw_distance_btw_wheels(xt1, yt1, zt1, track_wdt)
    delta_west = d_nw / track_wdt
    side_red = side * delta_west
    d_se = find_se_distance_btw_wheels(xt1, yt1, zt1, wheelbase)
    delta_north = d_se / wheelbase
    front_red = front * delta_north
    back_red = back * delta_north
    return side_red, front_red, back_red, d_nw, d_se


# obliczenie współrzędnych podwozia
def chassis_coordinates_NW(x, y, z, wheelbase, track_wdt, side_dst, front_dst, back_dst):
    side, front, back, d_west, d_north = reduced_dimensions(x, y, z, wheelbase, track_wdt, side_dst, front_dst, back_dst)  # zredukowane odległości podwozia pojazdu
    prw2 = np.sqrt(2) / 2
    x1 = x - prw2 * (side + front)
    y1 = y - prw2 * (side - front)
    x2 = x1 + (2 * side + d_west) * prw2
    y2 = y1 + (2 * side + d_west) * prw2
    x3 = x1 + (front + back + d_north) * prw2
    y3 = y1 - (front + back + d_north) * prw2
    x4 = x3 + (2 * side + d_west) * prw2
    y4 = y3 + (2 * side + d_west) * prw2
    results = np.array([[x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4]])
    return results


# wyznaczanie wysokości punktu na powierzchni
def point_plane_hgt(x, y, A, B, C, D2):
    z1 = -A / C * x - B / C * y - D2 / C
    return z1


# obliczanie współrzędnych kół, gdy mamy podany środek pojazdu
def find_wheel_coords(xs, ys, zs):
    dnw = find_nw_distance_btw_wheels(xs + e, ys + e, zs, wheelbase / 2) * np.sqrt(2) / 2
    dse = find_se_distance_btw_wheels(xs + e, ys + e, zs, wheelbase / 2) * np.sqrt(2) / 2
    five_ptx = xs - dnw
    five_pty = ys + dnw
    five_z = IDW(file_arr, five_ptx, five_pty, e, cell)
    six_ptx = xs + dse
    six_pty = ys - dse
    six_z = IDW(file_arr, six_ptx, six_pty, e, cell)
    dsw5 = find_sw_distance_btw_wheels(five_ptx + e, five_pty + e, five_z, track_wdt / 2) * np.sqrt(2) / 2
    dne5 = find_ne_distance_btw_wheels(five_ptx + e, five_pty + e, five_z, track_wdt / 2) * np.sqrt(2) / 2
    dsw6 = find_sw_distance_btw_wheels(six_ptx + e, six_pty + e, six_z, track_wdt / 2) * np.sqrt(2) / 2
    dne6 = find_ne_distance_btw_wheels(six_ptx + e, six_pty + e, six_z, track_wdt / 2) * np.sqrt(2) / 2
    wh1x = five_ptx - dsw5
    wh1y = five_pty - dsw5
    wh1z = IDW(file_arr, wh1x, wh1y, e, cell)
    wh2x = five_ptx + dne5
    wh2y = five_pty + dne5
    wh2z = IDW(file_arr, wh2x, wh2y, e, cell)
    wh3x = six_ptx - dsw6
    wh3y = six_pty - dsw6
    wh3z = IDW(file_arr, wh3x, wh3y, e, cell)
    wh4x = six_ptx + dne6
    wh4y = six_pty + dne6
    wh4z = IDW(file_arr, wh4x, wh4y, e, cell)
    wh_coord = np.array([[wh1x, wh1y, wh1z],
                         [wh2x, wh2y, wh2z],
                         [wh3x, wh3y, wh3z],
                         [wh4x, wh4y, wh4z]])
    return wh_coord

def isPinRectangle(r, p):
    """
        r: A list of four points, each has a x- and a y- coordinate
        P: A point
    """

    areaRectangle = np.sqrt((r[1][1] - r[0][1]) ** 2 + (r[1][0] - r[0][0]) ** 2) * np.sqrt((r[2][1] - r[0][1]) ** 2 + (r[2][0] - r[0][0]) ** 2)

    ABP = 0.5 * np.fabs(
        r[0][0] * (r[1][1] - p[1])
        + r[1][0] * (p[1] - r[0][1])
        + p[0] * (r[0][1] - r[1][1])
    )

    BCP = 0.5 * np.fabs(
        r[1][0] * (r[2][1] - p[1])
        + r[2][0] * (p[1] - r[1][1])
        + p[0] * (r[1][1] - r[2][1])
    )
    CDP = 0.5 * np.fabs(
        r[2][0] * (r[3][1] - p[1])
        + r[3][0] * (p[1] - r[2][1])
        + p[0] * (r[2][1] - r[3][1])
    )
    DAP = 0.5 * np.fabs(
        r[3][0] * (r[0][1] - p[1])
        + r[0][0] * (p[1] - r[3][1])
        + p[0] * (r[3][1] - r[0][1])
    )
    return areaRectangle >= (ABP + BCP + CDP + DAP)


# jazda na północ: znajdowanie współrzędnych wszystkich kół, przyjmując współrzedną z grida jako
def microrelief(length):
    file_result = np.c_[file_arr, np.ones(len(file_arr))]
    err_no = 0
    for i in bar(length):
        try:
            result = 1
            xs = file_arr[i, 0]
            ys = file_arr[i, 1]
            zs = file_arr[i, 2]
            wh_coord = find_wheel_coords(xs, ys, zs)
            delta_z = []
            for r in range(4):  # obliczenie różnicy wysokości kół na podstawie ich współrzednych, po kolei lg, pg, ld, pd (lg - lewa górna)
                r += 1
                table = [1, 2, 3, 4]
                wh_tab = []
                for j in range(4):
                    if r != table[j]:
                        wh_tab.append(table[j] - 1)
                A, B, C, D1, D2 = plane_equation(wh_coord[wh_tab, :])
                r -= 1
                z = point_plane_hgt(wh_coord[r][0], wh_coord[r][1], A, B, C, D1)
                z_ter = IDW(file_arr, wh_coord[r][0], wh_coord[r][1], e, cell)
                if z - z_ter < 0:
                    dz = z
                else:
                    dz = z - z_ter
                delta_z.append(dz)
                # wskazuje różnicę wysokości opony nieleżącej na powierzchni i  w metrach po kolei dla opon: lg, pg, ld, pd
            tire_num = delta_z.index(np.min(delta_z))
            if np.min(delta_z) >= ride_hgt:
                result = 0
            table_res = []
            for k in range(4):
                if table[k] != tire_num:
                    table_res.append(table[k] - 1)
            A, B, C, D1, D2 = plane_equation(wh_coord[table_res, :])
            a2, b2, c2, d12, d22 = plane_equation_reverse(wh_coord[table_res, :])
            ch_coord = chassis_coordinates_NW(wh_coord[0, 0], wh_coord[0, 1], wh_coord[0, 2], track_wdt, wheelbase, side_dst, front_dst, back_dst)
            x_spacing = np.linspace(ch_coord[0, 0], ch_coord[3, 0], 5)
            y_spacing = np.linspace(ch_coord[1, 1], ch_coord[2, 1], 5)
            for c in range(len(x_spacing)):
                for v in range(len(y_spacing)):
                    x = x_spacing[c]
                    y = y_spacing[v]
                    p = [x, y]
                    if isPinRectangle(ch_coord, p):
                        z_terrain = IDW(file_arr, x, y, e, cell)
                        z_plane = point_plane_hgt(x, y, A, B, C, D2)
                        z_plane_2 = point_plane_hgt(x, y, a2, b2, c2, d22)
                        if z_plane_2 > z_plane:
                            z_plane = z_plane_2
                        if z_terrain >= z_plane:
                            result = 0
                            break
            if result == 0:
                file_result[i, 3] = 0
            print(result)
        except:
            err_no += 1
            print(str(err_no) + " Bad coords: " + str(i))
            pass
    return file_result

def microrelief_chunk(length, file_arr):
    chunks = [length[i::processes] for i in range(processes)]
    pool = Pool(processes=processes)
    res = pool.map(microrelief, chunks)
    return res


os.chdir('C:\\Users\\wojo1\\Desktop\\Doktorat\\Microrelief\\Data\\Analysis_1')
filename = 'nmt.xyz'
file_arr = read_file(filename)
processes = 4
file_tab = np.c_[np.arange(len(file_arr)) + 1, file_arr]
process_lgt = np.arange(len(file_arr))
cell = 1  # wielkość piksela
e = 0.00001  # współczynnik powodujący niezerowanie się odległości
wheelbase = 2.9  # rozstaw osi
track_wdt = 1.65  # rozstaw kół
ride_hgt = 0.22  # prześwit
side_dst = 0.15  # odległość od boku do koła
front_dst = 0.75  # odległość od przodu do koła
back_dst = 1  # odległość od tyłu do koła
bar = ProgressBar()

if __name__ == "__main__":
    result = microrelief_chunk(process_lgt, file_arr)
    tab1 = result[:][:][0]
    tab2 = result[:][:][1]
    tab3 = result[:][:][2]
    tab4 = result[:][:][3]
    table_result = np.concatenate((tab1, tab2, tab3, tab4), axis=0)
    np.savetxt('test_NW.txt', table_result, delimiter=',')

# w kierunkach zmiana: linijka 294 (wheelbase z track_wdt miejscami zamiana, funkcje: find wheel coords, chassis coords, nazwa pliku