import trimesh
import numpy as np
import scipy.ndimage as ndimage
import heapq
import plotly.graph_objects as go

# --- 1. ФУНКЦИИ АЛГОРИТМА И ПОДГОТОВКИ ---

def prepare_grid(mesh_file, grid_shape, pitch, pipe_diameter):
    print("Вокселизация 3D-модели...")
    mesh = trimesh.load(mesh_file)
    voxelized = mesh.voxelized(pitch=pitch).fill()
    grid = np.zeros(grid_shape, dtype=bool)
    
    for p in voxelized.points:
        x, y, z = int(p[0] / pitch), int(p[1] / pitch), int(p[2] / pitch)
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            grid[x, y, z] = True

    radius_voxels = int(np.ceil((pipe_diameter / 2.0) / pitch))
    if radius_voxels > 0:
        print(f"Расширение стен на {radius_voxels} ячеек для клиренса трубы...")
        z, y, x = np.ogrid[-radius_voxels:radius_voxels+1, -radius_voxels:radius_voxels+1, -radius_voxels:radius_voxels+1]
        kernel = x**2 + y**2 + z**2 <= radius_voxels**2
        grid = ndimage.binary_dilation(grid, structure=kernel).astype(bool)
    return grid

def create_safe_zones(grid, start, goal, pipe_diameter, pitch):
    """
    Вырезает сферические 'безопасные зоны' вокруг старта и финиша.
    Это позволяет точкам лежать прямо на поверхностях оборудования.
    """
    safe_radius = int(np.ceil((pipe_diameter / 2.0) / pitch)) + 1 # Чуть больше радиуса трубы
    
    for pt in [start, goal]:
        cx, cy, cz = pt
        for dx in range(-safe_radius, safe_radius + 1):
            for dy in range(-safe_radius, safe_radius + 1):
                for dz in range(-safe_radius, safe_radius + 1):
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    # Проверка границ матрицы
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                        # Если ячейка попадает в радиус безопасной зоны - делаем её проходимой
                        if dx**2 + dy**2 + dz**2 <= safe_radius**2:
                            grid[nx, ny, nz] = False
    return grid

def astar_matrix(grid, start, goal, pipe_diameter, pitch):
    class Node:
        def __init__(self, pos, direction, parent=None, g=0, h=0):
            self.pos = pos
            self.direction = direction
            self.parent = parent
            self.g = g
            self.f = g + h
        def __lt__(self, other): return self.f < other.f

    open_list = []
    closed_set = set()
    
    heapq.heappush(open_list, Node(start, (0,0,0), None, 0, 0))
    g_scores = {(start, (0,0,0)): 0}
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    # Штраф за излом (поворот) трубы
    turn_penalty = 15 + (pipe_diameter / pitch) * 5 

    while open_list:
        current = heapq.heappop(open_list)
        if current.pos == goal:
            path = []
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1]

        state = (current.pos, current.direction)
        if state in closed_set: continue
        closed_set.add(state)

        for dx, dy, dz in directions:
            nx, ny, nz = current.pos[0] + dx, current.pos[1] + dy, current.pos[2] + dz
            new_dir = (dx, dy, dz)
            neighbor_pos = (nx, ny, nz)

            if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]):
                continue
            if grid[nx, ny, nz]:
                continue

            move_cost = 1
            if current.direction != (0,0,0) and current.direction != new_dir:
                move_cost += turn_penalty

            tentative_g = current.g + move_cost
            neighbor_state = (neighbor_pos, new_dir)

            if neighbor_state not in g_scores or tentative_g < g_scores[neighbor_state]:
                h = abs(nx - goal[0]) + abs(ny - goal[1]) + abs(nz - goal[2])
                g_scores[neighbor_state] = tentative_g
                heapq.heappush(open_list, Node(neighbor_pos, new_dir, current, tentative_g, h))
    return None

def simplify_path(path):
    """ Удаляет промежуточные точки, оставляя только углы поворотов """
    if not path or len(path) <= 2: return path
    simplified = [path[0]]
    for i in range(1, len(path)-1):
        v1 = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1], path[i][2]-path[i-1][2])
        v2 = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1], path[i+1][2]-path[i][2])
        if v1 != v2: 
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified

def draw_results(original_mesh_file, path, pipe_diameter):
    print("Создание 3D-модели трубы и рендер...")
    fig = go.Figure()
    
    mesh = trimesh.load(original_mesh_file)
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        color='gray', opacity=0.3, name="Помещение", hoverinfo='skip'
    ))

    pipe_meshes = []
    radius = pipe_diameter / 2.0
    
    for i in range(len(path)):
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(path[i])
        pipe_meshes.append(sphere)
        
        if i < len(path) - 1:
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            vec = p2 - p1
            length = np.linalg.norm(vec)
            
            if length > 0:
                cylinder = trimesh.creation.cylinder(radius=radius, height=length)
                rot_matrix = trimesh.geometry.align_vectors([0, 0, 1], vec)
                cylinder.apply_transform(rot_matrix)
                cylinder.apply_translation((p1 + p2) / 2.0)
                pipe_meshes.append(cylinder)

    full_pipe = trimesh.util.concatenate(pipe_meshes)
    fig.add_trace(go.Mesh3d(
        x=full_pipe.vertices[:, 0], y=full_pipe.vertices[:, 1], z=full_pipe.vertices[:, 2],
        i=full_pipe.faces[:, 0], j=full_pipe.faces[:, 1], k=full_pipe.faces[:, 2],
        color='blue', opacity=1.0, name=f"Труба Ø{pipe_diameter}"
    ))
    
    fig.update_layout(
        title=f"Трассировка трубы (Диаметр: {pipe_diameter} мм)", 
        scene=dict(aspectmode='data')
    )
    fig.write_html('result.html')
    print("Визуализация сохранена в 'result.html'.")

def get_matrix_index(real_coords, pitch):
    return (int(real_coords[0] / pitch), int(real_coords[1] / pitch), int(real_coords[2] / pitch))

# --- 2. НАСТРОЙКИ ВАШЕГО ФАЙЛА ---

YOUR_STL_FILE = "grisha.stl"   
PIPE_DIAMETER = 1000.0          # Диаметр трубы 
PITCH = 100.0                  

# Точки теперь можно ставить прямо в упор к стенам и оборудованию
REAL_START = (3200.0, 11200.0, 3200.0)  
REAL_GOAL = (700.0, 1200.0, 5222.0)   

# --- 3. ЗАПУСК ПРОГРАММЫ ---

print(f"Загрузка файла {YOUR_STL_FILE}...")
mesh = trimesh.load(YOUR_STL_FILE)

min_b, max_b = mesh.bounds
if any(min_b < 0):
    mesh.apply_translation(-min_b) 
    YOUR_STL_FILE = "shifted_" + YOUR_STL_FILE
    mesh.export(YOUR_STL_FILE)
    min_b, max_b = mesh.bounds 

grid_shape = (
    int(max_b[0] / PITCH) + 5, 
    int(max_b[1] / PITCH) + 5, 
    int(max_b[2] / PITCH) + 5
)

START = get_matrix_index(REAL_START, PITCH)
GOAL = get_matrix_index(REAL_GOAL, PITCH)

print("\nНачинаем вокселизацию и подготовку пространства...")
grid = prepare_grid(YOUR_STL_FILE, grid_shape, PITCH, PIPE_DIAMETER)

# ВЫЗОВ НОВОЙ ФУНКЦИИ: Очищаем места старта и финиша от "бетона"
grid = create_safe_zones(grid, START, GOAL, PIPE_DIAMETER, PITCH)

print("Запуск алгоритма A* (Поиск пути с учетом штрафов на изгиб)...")
path = astar_matrix(grid, START, GOAL, PIPE_DIAMETER, PITCH)

if path:
    print(f"✅ Путь найден! Изначально вокселей: {len(path)}")
    simplified_path = simplify_path(path)
    print(f"✅ Маршрут оптимизирован. Узлов поворота: {len(simplified_path)}")
    real_path = [(p[0]*PITCH, p[1]*PITCH, p[2]*PITCH) for p in simplified_path]
    draw_results(YOUR_STL_FILE, real_path, PIPE_DIAMETER)
else:
    print("❌ Ошибка: Путь не найден.")