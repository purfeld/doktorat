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
def find_east_distance_btw_wheels(xt1, yt1, zt1, track_wdt):
    x0t2 = xt1 + track_wdt
    y0t2 = yt1
    z0t2 = IDW(file_arr, x0t2, y0t2, e, cell)
    dH = z0t2 - zt1
    d12 = track_wdt * np.cos(np.arctan(dH / track_wdt))
    return d12


# znalezienie zredukowanego rozstawu osi ze względu na nachylenie terenu
def find_south_distance_btw_wheels(xt1, yt1, zt1, wheelbase):
    x0t2 = xt1
    y0t2 = yt1 - wheelbase
    z0t2 = IDW(file_arr, x0t2, y0t2, e, cell)
    dH = z0t2 - zt1
    d12 = wheelbase * np.cos(np.arctan(dH / wheelbase))
    return d12


# znalezienie współrzędnych 3 punktów, na podstawie których zostanie utworzone równanie powierzchni
def plane_points():
    zt1 = IDW(file_arr, xt1, yt1, e, cell)
    dX = find_east_distance_btw_wheels(xt1, yt1, zt1, track_wdt)
    dY = find_south_distance_btw_wheels(xt1, yt1, zt1, wheelbase)
    xt2 = xt1 + dX
    yt2 = yt1
    xt3 = xt1
    yt3 = yt1 - dY
    zt2 = IDW(file_arr, xt2, yt2, e, cell)
    zt3 = IDW(file_arr, xt3, yt3, e, cell)
    plane = np.array([[xt1, yt1, zt1],
                      [xt2, yt2, zt2],
                      [xt3, yt3, zt3]])
    return plane


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
def reduced_dimensions(xt1, yt1, wheelbase, track_wdt, side, front, back):
    zt1 = IDW(file_arr, xt1, yt1, e, cell)
    d_west = find_east_distance_btw_wheels(xt1, yt1, zt1, track_wdt)
    delta_west = d_west / track_wdt
    side_red = side * delta_west
    d_north = find_south_distance_btw_wheels(xt1, yt1, zt1, wheelbase)
    delta_north = d_north / wheelbase
    front_red = front * delta_north
    back_red = back * delta_north
    return side_red, front_red, back_red, d_west, d_north


# obliczenie współrzędnych podwozia
def chassis_coordinates_W(x, y, wheelbase, track_wdt, side_dst, front_dst, back_dst):
    side, front, back, d_west, d_north = reduced_dimensions(x, y, wheelbase, track_wdt, side_dst, front_dst, back_dst)  # zredukowane odległości podwozia pojazdu
    x1 = x - front
    y1 = y + side
    x2 = x + d_west + back
    y2 = y1
    x3 = x1
    y3 = y - d_north - side
    x4 = x2
    y4 = y3
    results = np.array([[x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4]])
    return results


# wyznaczenie wysokości punktów skrajnych podwozia
def chassis_points_hgt(x, y):
    coord_arr = chassis_coordinates_W(x, y)
    A, B, C, D1, D2 = plane_equation(plane_points())
    z1 = -A / C * coord_arr[0, 0] - B / C * coord_arr[0, 1] - D2 / C
    z2 = -A / C * coord_arr[1, 0] - B / C * coord_arr[1, 1] - D2 / C
    z3 = -A / C * coord_arr[2, 0] - B / C * coord_arr[2, 1] - D2 / C
    z4 = -A / C * coord_arr[3, 0] - B / C * coord_arr[3, 1] - D2 / C
    res = np.array([z1, z2, z3, z4])
    return res


# wyznaczanie wysokości punktu na powierzchni
def point_plane_hgt(x, y, A, B, C, D2):
    z1 = -A / C * x - B / C * y - D2 / C
    return z1


# obliczanie współrzędnych kół, gdy mamy podany środek pojazdu
def find_wheel_coords(xs, ys, zs):
    wh1x = xs - find_east_distance_btw_wheels(xs + e, ys + e, zs, wheelbase / 2)
    wh1y = ys + find_south_distance_btw_wheels(xs + e, ys + e, zs, track_wdt / 2)
    wh1z = IDW(file_arr, wh1x, wh1y, e, cell)
    wh2x = xs + find_east_distance_btw_wheels(xs + e, ys + e, zs, wheelbase / 2)
    wh2y = ys + find_south_distance_btw_wheels(xs + e, ys + e, zs, track_wdt / 2)
    wh2z = IDW(file_arr, wh2x, wh2y, e, cell)
    wh3x = xs - find_east_distance_btw_wheels(xs + e, ys + e, zs, wheelbase / 2)
    wh3y = ys - find_south_distance_btw_wheels(xs + e, ys + e, zs, track_wdt / 2)
    wh3z = IDW(file_arr, wh3x, wh3y, e, cell)
    wh4x = xs + find_east_distance_btw_wheels(xs + e, ys + e, zs, wheelbase / 2)
    wh4y = ys - find_south_distance_btw_wheels(xs + e, ys + e, zs, track_wdt / 2)
    wh4z = IDW(file_arr, wh4x, wh4y, e, cell)
    wh_coord = np.array([[wh1x, wh1y, wh1z],
                         [wh2x, wh2y, wh2z],
                         [wh3x, wh3y, wh3z],
                         [wh4x, wh4y, wh4z]])
    return wh_coord


os.chdir('C:\\Users\\wojo1\\Desktop\\Doktorat\\Microrelief\\Data\\Analysis_1')
filename = 'nmt.xyz'
file_arr = read_file(filename)
processes = 4
file_tab = np.c_[np.arange(len(file_arr)) + 1, file_arr]
process_lgt = np.arange(len(file_arr))
cell = 1  # wielkość piksela
xt1 = 574965.0  # współrzędna x lewego górnego koła
yt1 = 5552741.0  # współrzędna y lewego górnego koła
e = 0.00001  # współczynnik powodujący niezerowanie się odległości
wheelbase = 2.9  # rozstaw osi
track_wdt = 1.65  # rozstaw kół
ride_hgt = 0.22  # prześwit
side_dst = 0.15  # odległość od boku do koła
front_dst = 0.75  # odległość od przodu do koła
back_dst = 1  # odległość od tyłu do koła
bar = ProgressBar()
# A, B, C, D1, D2 = plane_equation(plane_points())  # parametry równań powierzchni [A, B, C] - wektor normalny powierzchni

# print(A, B, C, D1, D2)
# print(IDW(file_arr, xt1, yt1, e, cell))
# print(-A/C * xt1 - B/C * yt1 - D2/C)
# print(chassis_points_hgt(xt1, yt1))

# ch_coord = chassis_coordinates(xt1, yt1)
#
# x_spacing = np.linspace(ch_coord[0, 0], ch_coord[1, 0], 10)
# y_spacing = np.linspace(ch_coord[0, 1], ch_coord[2, 1], 25)
# A, B, C, D1, D2 = plane_equation(plane_points())
# bad_coord = []
# for i in bar(range(len(x_spacing))):
#     for j in range(len(y_spacing)):
#         x = x_spacing[i]
#         y = y_spacing[j]
#         z_terrain = IDW(file_arr, x, y, e, cell)
#         z_plane = point_plane_hgt(x, y, A, B, C, D2)
#         if z_terrain >= z_plane:
#             print("\nBad coordinates: X = " + str(x) + " Y = " + str(y), z_plane, z_terrain)
#             bad_coord.append([x, y])
# print(bad_coord)

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
            # if tire_num == 1 or tire_num == 4:
            #     which_tire = 2
            # else:
            #     which_tire = 1
            # print(which_tire)
            table_res = []
            for k in range(4):
                if table[k] != tire_num:
                    table_res.append(table[k] - 1)
            A, B, C, D1, D2 = plane_equation(wh_coord[table_res, :])
            a2, b2, c2, d12, d22 = plane_equation_reverse(wh_coord[table_res, :])
            ch_coord = chassis_coordinates_W(wh_coord[0, 0], wh_coord[0, 1], track_wdt, wheelbase, side_dst, front_dst, back_dst)
            x_spacing = np.linspace(ch_coord[0, 0], ch_coord[1, 0], 5)
            y_spacing = np.linspace(ch_coord[0, 1], ch_coord[2, 1], 2)
            for c in range(len(x_spacing)):
                for v in range(len(y_spacing)):
                    x = x_spacing[c]
                    y = y_spacing[v]
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

if __name__ == "__main__":
    result = microrelief_chunk(process_lgt, file_arr)
    tab1 = result[:][:][0]
    tab2 = result[:][:][1]
    tab3 = result[:][:][2]
    tab4 = result[:][:][3]
    table_result = np.concatenate((tab1, tab2, tab3, tab4), axis=0)
    np.savetxt('test_W.txt', table_result, delimiter=',')

# w kierunkach zmiana: linijka 294 (wheelbase z track_wdt miejscami zamiana, funkcje: find wheel coords, chassis coords, nazwa pliku