import numpy as np

from .utils import get_vtk_object_from_scalar_data


def addMaxData(lat, lon, scalar_values, scalar_name):

    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_max = np.zeros((r, c))
    vertex_identifiers = np.zeros(r * c)

    rect = get_vtk_object_from_scalar_data(lon, lat, scalar_values, scalar_name)
    # pv.RectilinearGrid(lon, lat)
    # scalar = scalar_values[::-1, :].ravel()
    # rect.point_arrays["v"] = scalar

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            vertex_identifiers[k] = k + 1
            k += 1
            max_flag = 1

            if check[i][j] == 1:
                continue

            else:
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if scalar_values[x][y] > scalar_values[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                if j == c - 1:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, 0]:
                            if (0 <= x < r) and (0 <= y < c):
                                if scalar_values[x][y] > scalar_values[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if scalar_values[x][y] > scalar_values[i][j]:
                                    max_flag = 0
                                else:
                                    check[x][y] = 1

            if max_flag == 1:
                is_max[i][j] = 1
                check[i][j] = 1
                count += 1

    cell_number = rect.GetNumberOfCells()
    cell_id = np.zeros(cell_number)
    for i in range(cell_number):
        cell_id[i] = i

    rect.point_arrays["is max"] = is_max[::-1, :].ravel()
    rect.point_arrays["Vertex_id"] = vertex_identifiers
    rect.cell_arrays["Cell_{}".format(scalar_name)] = cell_id
    # print("max_count", count)
    return (rect, is_max)


def addMinData(lat, lon, scalar_values, scalar_name):

    # scalar_negative = np.negative(scalar_values)

    r, c = scalar_values.shape
    check = np.zeros((r, c))
    is_min = np.zeros((r, c))
    vertex_identifiers = np.zeros(r * c)

    rect = get_vtk_object_from_scalar_data(lon, lat, scalar_values, scalar_name)

    # rect = pv.RectilinearGrid(lon, lat)
    # scalar = scalar_values[::-1, :].ravel()
    # rect.point_arrays["v"] = scalar

    count = 0
    k = 0

    for i in range(r):
        for j in range(c):

            k += 1
            min_flag = 1

            if check[i][j] == 1:
                continue

            else:
                if j == 0:
                    for x in [i - 1, i, i + 1]:
                        for y in [c - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if scalar_values[x][y] < scalar_values[i][j]:
                                    min_flag = 0
                                else:
                                    check[x][y] = 1

                if j == c - 1:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, 0]:
                            if (0 <= x < r) and (0 <= y < c):
                                if scalar_values[x][y] < scalar_values[i][j]:
                                    min_flag = 0
                                else:
                                    check[x][y] = 1

                else:
                    for x in [i - 1, i, i + 1]:
                        for y in [j - 1, j, j + 1]:
                            if (0 <= x < r) and (0 <= y < c):
                                if scalar_values[x][y] < scalar_values[i][j]:
                                    min_flag = 0
                                else:
                                    check[x][y] = 1

                if min_flag == 1 and i != 0:
                    is_min[i][j] = 1
                    check[i][j] = 1
                    count += 1

    cell_number = rect.GetNumberOfCells()
    cell_id = np.zeros(cell_number)
    for i in range(cell_number):
        cell_id[i] = i

    rect.point_arrays["is min"] = is_min[::-1, :].ravel()
    rect.point_arrays["Vertex_id"] = vertex_identifiers
    rect.cell_arrays["Cell_{}".format(scalar_name)] = cell_id
    # print("min points", count)
    return (rect, is_min)


def interpolateCellVals(inputs, scalar_name):

    numCells = inputs.GetNumberOfCells()
    scalar_v = inputs.GetPointData().GetArray(scalar_name)
    cell_scalars = vtk.vtkFloatArray()
    cell_scalars.SetNumberOfComponents(1)
    cell_scalars.SetNumberOfTuples(numCells)
    cell_scalars.SetName("Cell {}".format(scalar_name))

    for i in range(numCells):
        cell = inputs.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        func_value = 0
        for j in range(num_points):
            pid = cell.GetPointId(j)
            func_value += scalar_v.GetTuple1(pid)
        func_value /= num_points
        cell_scalars.SetTuple1(i, func_value)

    inputs.GetCellData().AddArray(cell_scalars)
    return inputs


def interpolateCellVals_min(inputs, scalar_name):

    numCells = inputs.GetNumberOfCells()
    scalar_v = inputs.GetPointData().GetArray(scalar_name)
    cell_scalars = vtk.vtkFloatArray()
    cell_scalars.SetNumberOfComponents(1)
    cell_scalars.SetNumberOfTuples(numCells)
    cell_scalars.SetName("Cell {}".format(scalar_name))

    for i in range(numCells):
        cell = inputs.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        func_value = 0
        for j in range(num_points):
            pid = cell.GetPointId(j)
            func_value += scalar_v.GetTuple1(pid)
        func_value /= num_points
        cell_scalars.SetTuple1(i, func_value)

    inputs.GetCellData().AddArray(cell_scalars)
    return inputs


def clipDataset(dataset, scalar_name, scalar_val):

    clip_dataset = vtk.vtkClipDataSet()
    dataset.GetPointData().SetScalars(dataset.GetPointData().GetArray(scalar_name))
    clip_dataset.SetValue(scalar_val)
    clip_dataset.SetInputData(dataset)
    clip_dataset.Update()
    return clip_dataset.GetOutput()


def clipDataset_min(dataset, scalar_name, scalar_val):

    clip_dataset = vtk.vtkClipDataSet()
    dataset.GetPointData().SetScalars(dataset.GetPointData().GetArray(scalar_name))
    clip_dataset.SetValue(scalar_val)
    clip_dataset.SetInputData(dataset)
    clip_dataset.InsideOutOn()
    clip_dataset.Update()
    return clip_dataset.GetOutput()
