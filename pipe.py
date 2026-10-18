import numpy as np

class Node: 
    def __init__(self, nodeid, x, y, z, support=False):
        self.nodeid = nodeid
        self.coords = np.array([x, y, z], dtype=float)
        # 6 степеней свободы: [X, Y, Z, rot_X, rot_Y, rot_Z]
        self.matrix = np.array([False, False, False, False, False, False])
        self.is_support = support 


class pipeElement: 
    def __init__(self, node1: Node, node2: Node, diamerterR: float, diamerterr: float, E: float, p: float):
        self.node1 = node1
        self.node2 = node2
        self.diamerterR = diamerterR
        self.diamerterr = diamerterr
        self.E = E
        self.p = p
        self.alpha_t = 1.2e-5  # Коэффициент теплового расширения стали
        
        self.apipe = (diamerterR**2 - diamerterr**2) * np.pi / 4
        self.m = self.apipe * p
        self.weight = self.m * 9.81 
        
        self.I = (diamerterR**4 - diamerterr**4) * np.pi / 64 #Осевой момент инерции
        self.J = (diamerterR**4 - diamerterr**4) * np.pi / 32 # Полярный момент инерции
        self.W = (diamerterR**4 - diamerterr**4) * np.pi / 32 / diamerterR # Момент сопротивления сечения
        
        vector = self.node2.coords - self.node1.coords
        self.length = np.linalg.norm(vector)

    def moment_local(self): 
        L = self.length
        E = self.E
        G = E / (2 * (1 + 0.3))

        k_ax  = (E * self.apipe) / L
        k_tor = (G * self.J) / L
        k_b1  = (12 * E * self.I) / (L**3)
        k_b2  = (6 * E * self.I) / (L**2)
        k_b3  = (4 * E * self.I) / L
        k_b4  = (2 * E * self.I) / L

        K11 = np.array([
            [k_ax,  0,     0,      0,     0,     0   ],
            [0,     k_b1,  0,      0,     0,     k_b2],
            [0,     0,     k_b1,   0,    -k_b2,  0   ],
            [0,     0,     0,      k_tor, 0,     0   ],
            [0,     0,    -k_b2,   0,     k_b3,  0   ],
            [0,     k_b2,  0,      0,     0,     k_b3]
        ])

        K22 = np.array([
            [k_ax,  0,     0,      0,     0,     0   ],
            [0,     k_b1,  0,      0,     0,    -k_b2],
            [0,     0,     k_b1,   0,     k_b2,  0   ],
            [0,     0,     0,      k_tor, 0,     0   ],
            [0,     0,     k_b2,   0,     k_b3,  0   ],
            [0,    -k_b2,  0,      0,     0,     k_b3]
        ])

        K12 = np.array([
            [-k_ax,  0,     0,      0,     0,     0   ],
            [ 0,    -k_b1,  0,      0,     0,     k_b2],
            [ 0,     0,    -k_b1,   0,    -k_b2,  0   ],
            [ 0,     0,     0,     -k_tor, 0,     0   ],
            [ 0,     0,     k_b2,   0,     k_b4,  0   ],
            [ 0,    -k_b2,  0,      0,     0,     k_b4]
        ])

        K21 = K12.T
        return np.block([[K11, K12], [K21, K22]])

    def moment_global(self):  
        K_local = self.moment_local() 
        dx = self.node2.coords[0] - self.node1.coords[0]
        dy = self.node2.coords[1] - self.node1.coords[1]
        dz = self.node2.coords[2] - self.node1.coords[2]
        
        cx, cy, cz = dx / self.length, dy / self.length, dz / self.length
        
        if np.isclose(cx, 0.0) and np.isclose(cy, 0.0):
            Lambda = np.array([[0, 0, np.sign(cz)], [0, 1, 0], [-np.sign(cz), 0, 0]])
        else:
            cxy = np.sqrt(cx**2 + cy**2)
            Lambda = np.array([[cx, cy, cz], [-cy/cxy, cx/cxy, 0], [-cx*cz/cxy, -cy*cz/cxy, cxy]])
            
        T = np.zeros((12, 12))
        T[0:3, 0:3] = Lambda
        T[3:6, 3:6] = Lambda
        T[6:9, 6:9] = Lambda
        T[9:12, 9:12] = Lambda
        
        return T.T @ K_local @ T


def split(pipe, pressure, progib, startid): 
    lmax1 = np.sqrt((8 * pressure * pipe.W) / pipe.weight)
    lmax2 = (384 * pipe.E * pipe.I * progib / (5 * pipe.weight))**(1/4)
    l = min(lmax1, lmax2) 
    
    if pipe.length <= l:
        return [pipe], []
        
    sengents = int(np.ceil(pipe.length / l))
    new_pipes = []
    new_nodes = []
    current_node = pipe.node1
    currentid = startid
    step_vector = (pipe.node2.coords - pipe.node1.coords) / sengents 
    
    for i in range(1, sengents):
        new_coords = pipe.node1.coords + step_vector * i 
        new_node = Node(currentid, new_coords[0], new_coords[1], new_coords[2], support=True)
        new_node.matrix[2] = True  
        new_nodes.append(new_node)
        currentid += 1 
        
        new_pipe = pipeElement(current_node, new_node, pipe.diamerterR, pipe.diamerterr, pipe.E, pipe.p)
        new_pipes.append(new_pipe)
        current_node = new_node
        
    last_pipe = pipeElement(current_node, pipe.node2, pipe.diamerterR, pipe.diamerterr, pipe.E, pipe.p)
    new_pipes.append(last_pipe)
    
    return new_pipes, new_nodes


# ==========================================
# 3. МАТРИЧНЫЙ РЕШАТЕЛЬ ТЕПЛОВОГО УДАРА (ОБНОВЛЕННЫЙ)
# ==========================================

def solve_thermal_shock(nodes, pipes, delta_T):
    N = len(nodes)
    K_global = np.zeros((6*N, 6*N))
    F_global = np.zeros(6*N)
    
    for pipe in pipes:
        idx1 = nodes.index(pipe.node1) * 6
        idx2 = nodes.index(pipe.node2) * 6
        
        K_elem = pipe.moment_global()
        K_global[idx1:idx1+6, idx1:idx1+6] += K_elem[0:6, 0:6]
        K_global[idx1:idx1+6, idx2:idx2+6] += K_elem[0:6, 6:12]
        K_global[idx2:idx2+6, idx1:idx1+6] += K_elem[6:12, 0:6]
        K_global[idx2:idx2+6, idx2:idx2+6] += K_elem[6:12, 6:12]
        
        # Температурная нагрузка
        F_th = pipe.E * pipe.apipe * pipe.alpha_t * delta_T
        dx, dy, dz = pipe.node2.coords - pipe.node1.coords
        cx, cy, cz = dx/pipe.length, dy/pipe.length, dz/pipe.length
        
        f_vec = np.array([F_th*cx, F_th*cy, F_th*cz, 0, 0, 0])
        F_global[idx1:idx1+6] -= f_vec
        F_global[idx2:idx2+6] += f_vec
        
        # ДОБАВЛЕН УЧЕТ ВЕСА ТРУБЫ (распределяем пополам на Z-ось каждого узла)
        F_global[idx1+2] -= pipe.weight / 2.0
        F_global[idx2+2] -= pipe.weight / 2.0

    K_original = K_global.copy()
    F_original = F_global.copy()

    # Наложение граничных условий
    for i, node in enumerate(nodes):
        idx = i * 6
        for dof_index in range(6):
            if node.matrix[dof_index] == True: 
                global_dof = idx + dof_index
                K_global[global_dof, :] = 0
                K_global[:, global_dof] = 0
                K_global[global_dof, global_dof] = 1.0
                F_global[global_dof] = 0.0

    # Проверка обусловленности (чтобы ловить летающие в воздухе трубы)
    cond_val = np.linalg.cond(K_global)
    if cond_val > 1e12:
        print(f"\n⚠️ ВНИМАНИЕ: Матрица плохо обусловлена ({cond_val:.2e})! Возможно, системе не хватает опор.")

    # Решение системы
    U_global = np.linalg.solve(K_global, F_global)
    Reactions = K_original @ U_global - F_original
    
    max_axial_force = 0.0
    critical_node = None
    
    print("\n--- РЕАКЦИИ В ОПОРАХ ---")
    for i, node in enumerate(nodes):
        if not node.is_support:
            continue
            
        idx = i * 6
        node_R = Reactions[idx:idx+6]
        
        # Находим примыкающую трубу, чтобы проецировать силу на ее ось
        connected_pipes = [p for p in pipes if p.node1 == node or p.node2 == node]
        if not connected_pipes: continue
        
        pipe = connected_pipes[0]
        vec = pipe.node2.coords - pipe.node1.coords
        vec = vec / np.linalg.norm(vec)
        
        # Осевая сила (проекция), вертикальная сила (реакция от веса) и момент
        F_axial = abs(np.dot(node_R[0:3], vec))
        F_z = abs(node_R[2]) 
        M_norm = np.linalg.norm(node_R[3:6])
        
        print(f"Опора ID {node.nodeid}: Осевая сила = {F_axial:,.0f} Н, Вертикальная (вес) = {F_z:,.0f} Н, Момент = {M_norm:,.0f} Н*м")
        
        if F_axial > max_axial_force:
            max_axial_force = F_axial
            critical_node = node
            
    print(f"\n[КРИТИЧЕСКАЯ ОПОРА] ID {critical_node.nodeid} перегружена по оси: {max_axial_force:,.0f} Н")
    
    # Возвращаем еще и перемещения для расчета напряжений
    return critical_node, U_global


# ==========================================
# НОВАЯ ФУНКЦИЯ: РАСЧЕТ НАПРЯЖЕНИЙ В ТРУБАХ (МПа)
# ==========================================
def calculate_stresses(nodes, pipes, U_global, delta_T):
    print("\n--- НАПРЯЖЕНИЯ В ТРУБАХ ---")
    max_stress_global = 0.0
    critical_pipe = None
    
    for pipe in pipes:
        idx1 = nodes.index(pipe.node1) * 6
        idx2 = nodes.index(pipe.node2) * 6
        
        U_elem = np.zeros(12)
        U_elem[0:6] = U_global[idx1:idx1+6]
        U_elem[6:12] = U_global[idx2:idx2+6]
        
        dx, dy, dz = pipe.node2.coords - pipe.node1.coords
        cx, cy, cz = dx/pipe.length, dy/pipe.length, dz/pipe.length
        if np.isclose(cx, 0.0) and np.isclose(cy, 0.0):
            Lambda = np.array([[0, 0, np.sign(cz)], [0, 1, 0], [-np.sign(cz), 0, 0]])
        else:
            cxy = np.sqrt(cx**2 + cy**2)
            Lambda = np.array([[cx, cy, cz], [-cy/cxy, cx/cxy, 0], [-cx*cz/cxy, -cy*cz/cxy, cxy]])
            
        T = np.zeros((12, 12))
        T[0:3, 0:3] = Lambda
        T[3:6, 3:6] = Lambda
        T[6:9, 6:9] = Lambda
        T[9:12, 9:12] = Lambda
        
        # Перевод в локальные координаты
        u_local = T @ U_elem
        
        K_loc = pipe.moment_local()
        F_th = pipe.E * pipe.apipe * pipe.alpha_t * delta_T
        F_init_local = np.array([-F_th, 0, 0, 0, 0, 0, F_th, 0, 0, 0, 0, 0])
        
        # Внутренние усилия
        F_internal = K_loc @ u_local - F_init_local
        
        N = abs(F_internal[0])
        M_y = F_internal[4]
        M_z = F_internal[5]
        M_bend = np.sqrt(M_y**2 + M_z**2)
        
        # Sigma = N / A + M / W
        sigma_N = N / pipe.apipe
        sigma_M = M_bend / pipe.W
        sigma_total = sigma_N + sigma_M
        
        sigma_MPa = sigma_total / 1e6 
        
        if sigma_MPa > max_stress_global:
            max_stress_global = sigma_MPa
            critical_pipe = pipe
            
    print(f"[МАКСИМАЛЬНОЕ НАПРЯЖЕНИЕ] В трубе {critical_pipe.node1.nodeid}->{critical_pipe.node2.nodeid}: {max_stress_global:.1f} МПа")
    
    if max_stress_global > 240:
        print("❌ ВНИМАНИЕ: Напряжения превышают предел текучести Ст3 (240 МПа). Трубу деформирует!")
    else:
        print("✅ ОК: Напряжения в безопасной зоне.")


# ==========================================
# 4. ФУНКЦИЯ ВРЕЗКИ КОМПЕНСАТОРА
# ==========================================
def compensator(nodes, pipes, critical_node, W=2.0, H=2.0):
    if critical_node == nodes[0] or critical_node == nodes[-1]:
        support_nodes = [n for n in nodes if n.is_support and n != nodes[0] and n != nodes[-1]]
        if support_nodes:
            critical_node = support_nodes[0]

    pipe_in = next(p for p in pipes if p.node2 == critical_node)
    pipe_out = next(p for p in pipes if p.node1 == critical_node)
    
    dir_vec = pipe_out.node2.coords - pipe_in.node1.coords
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
    if abs(dir_vec[0]) > 0.5 or abs(dir_vec[1]) > 0.5:
        ortho_vec = np.array([-dir_vec[1], dir_vec[0], 0.0])
    else:
        ortho_vec = np.array([1.0, 0.0, 0.0])
        
    C = critical_node.coords
    coord_N1 = C - dir_vec * (W / 2.0)
    coord_N2 = coord_N1 + ortho_vec * H
    coord_N3 = coord_N2 + dir_vec * W
    coord_N4 = coord_N3 - ortho_vec * H
    
    max_id = max(n.nodeid for n in nodes)
    
    n1 = Node(max_id + 1, coord_N1[0], coord_N1[1], coord_N1[2], support=True)
    n1.matrix[2] = True 
    n4 = Node(max_id + 4, coord_N4[0], coord_N4[1], coord_N4[2], support=True)
    n4.matrix[2] = True
    
    n2 = Node(max_id + 2, coord_N2[0], coord_N2[1], coord_N2[2], support=False)
    n3 = Node(max_id + 3, coord_N3[0], coord_N3[1], coord_N3[2], support=False)
    
    D, d, E, rho = pipe_in.diamerterR, pipe_in.diamerterr, pipe_in.E, pipe_in.p
    
    p1 = pipeElement(pipe_in.node1, n1, D, d, E, rho)
    p2 = pipeElement(n1, n2, D, d, E, rho)
    p3 = pipeElement(n2, n3, D, d, E, rho)
    p4 = pipeElement(n3, n4, D, d, E, rho)
    p5 = pipeElement(n4, pipe_out.node2, D, d, E, rho)
    
    pipes.remove(pipe_in)
    pipes.remove(pipe_out)
    nodes.remove(critical_node)
    
    nodes.extend([n1, n2, n3, n4])
    pipes.extend([p1, p2, p3, p4, p5])
    
    return nodes, pipes


if __name__ == "__main__":
    node_A = Node(1, 0, 0, 0, support=True)
    node_A.matrix = np.array([True, True, True, True, True, True])
    
    node_B = Node(2, 23.0, 0, 0, support=True)
    node_B.matrix = np.array([True, True, True, True, True, True])
    
    long_pipe = pipeElement(node_A, node_B, 0.219, 0.203, 2e11, 7850) 
    result_pipes, result_nodes = split(long_pipe, pressure=140e6, progib=0.015, startid=3)
    
    all_nodes = [node_A] + result_nodes + [node_B]
    all_pipes = result_pipes
    
    print("=== ЭТАП 1: РАСЧЕТ ИСХОДНОЙ ТРАССЫ ===")
    critical_node, U_global_1 = solve_thermal_shock(all_nodes, all_pipes, delta_T=100)
    calculate_stresses(all_nodes, all_pipes, U_global_1, delta_T=100)
    
    print(f"\n=== ЭТАП 2: ВРЕЗКА КОМПЕНСАТОРА В УЗЕЛ {critical_node.nodeid} ===")
    all_nodes, all_pipes = compensator(all_nodes, all_pipes, critical_node, W=2.0, H=2.0)
    
    print("\n=== ЭТАП 3: ПОВТОРНЫЙ РАСЧЕТ ПОСЛЕ ВРЕЗКИ КОМПЕНСАТОРА ===")
    _, U_global_2 = solve_thermal_shock(all_nodes, all_pipes, delta_T=100)
    calculate_stresses(all_nodes, all_pipes, U_global_2, delta_T=100)