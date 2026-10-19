"""
pipe_v2.py - переработанная версия pipe.py.

Что делает:
  1. Строит трассу по списку точек (первая и последняя - анкеры).
  2. Расставляет направляющие опоры по допустимому пролёту (из веса).
  3. Считает МКЭ два случая: вес+давление (длительная нагрузка) и температура.
  4. По усилиям в КАЖДОМ элементе считает напряжения и сравнивает с допускаемыми
     (упрощённо, по духу ASME B31.3 / статьи Yeo et al.).
  5. Находит самый нагруженный прямой участок и пробует поставить в него
     П-компенсатор; оставляет вставку, только если нагрузка реально упала.

Допущения (важно!): допускаемые напряжения - условные (Sh = Sc = 140 МПа),
коэффициенты интенсификации напряжений в отводах (SIF) не учитываются,
расчёт линейный (без потери устойчивости), трение в опорах не учтено.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

G = 9.81


# ==========================================================
# 1. ДАННЫЕ: сечение, допускаемые значения, узлы, элементы
# ==========================================================

@dataclass
class Section:
    D: float                    # наружный диаметр, м
    d: float                    # внутренний диаметр, м
    E: float = 2.0e11           # модуль Юнга, Па
    rho: float = 7850.0         # плотность стали, кг/м3
    alpha: float = 1.2e-5       # коэфф. линейного расширения, 1/К
    nu: float = 0.3
    rho_fluid: float = 1000.0   # плотность среды (вода), кг/м3
    insulation: float = 0.0     # масса изоляции, кг/м
    P: float = 1.6e6            # внутреннее давление, Па

    @property
    def A(self):
        return math.pi * (self.D**2 - self.d**2) / 4

    @property
    def I(self):
        return math.pi * (self.D**4 - self.d**4) / 64

    @property
    def J(self):
        return 2 * self.I

    @property
    def W(self):                # момент сопротивления при изгибе
        return self.I / (self.D / 2)

    @property
    def q(self):                # погонный вес: сталь + среда + изоляция, Н/м
        fluid = math.pi * self.d**2 / 4 * self.rho_fluid
        return (self.A * self.rho + fluid + self.insulation) * G

    @property
    def sigma_P(self):          # продольное напряжение от давления, Па
        return self.P * self.d**2 / (self.D**2 - self.d**2)


@dataclass
class Limits:
    Sh: float = 140e6           # допускаемое при рабочей температуре, Па (условно)
    Sc: float = 140e6           # допускаемое при монтажной температуре, Па (условно)
    f: float = 1.0              # коэффициент циклов
    sag: float = 0.015          # допустимый прогиб, м

    @property
    def SA(self):               # допускаемое напряжение компенсации (B31.3, ур. 1a)
        return self.f * (1.25 * self.Sc + 0.25 * self.Sh)


class Node:
    """kind: None - свободный узел, 'rest' - скользящая опора (только z),
    'guide' - направляющая опора, 'anchor' - анкер."""

    def __init__(self, nodeid, x, y, z, kind=None, fixed=None):
        self.nodeid = nodeid
        self.coords = np.array([x, y, z], dtype=float)
        self.kind = kind
        # fixed[i] = True, если i-я степень свободы (ux,uy,uz,rx,ry,rz) закреплена
        self.fixed = np.zeros(6, dtype=bool) if fixed is None else np.array(fixed, dtype=bool)


def make_anchor(node):
    node.kind = "anchor"
    node.fixed[:] = True


def make_rest(node):
    """Скользящая опора под отводом: держит только вес (z), всё остальное свободно."""
    node.kind = "rest"
    node.fixed[:] = False
    node.fixed[2] = True


def make_guide(node, direction):
    """Направляющая: держит вес (z) и боковое смещение, вдоль трубы свободна."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    node.kind = "guide"
    node.fixed[:] = False
    if abs(d[2]) > 0.7:                       # вертикальная труба: держим x и y
        node.fixed[0] = node.fixed[1] = True
    else:                                     # горизонтальная: держим z и боковую ось
        node.fixed[2] = True
        node.fixed[1 if abs(d[0]) >= abs(d[1]) else 0] = True


class PipeElement:
    """Прямой балочный элемент (6 степеней свободы в каждом узле)."""

    def __init__(self, n1: Node, n2: Node, sec: Section, parent=None):
        self.n1, self.n2, self.sec, self.parent = n1, n2, sec, parent
        self.tag = None                       # 'loop' - элемент компенсатора
        v = n2.coords - n1.coords
        self.length = float(np.linalg.norm(v))
        self.Lambda = self._rotation(v / self.length)
        self.T = np.kron(np.eye(4), self.Lambda)          # глобальные -> локальные
        self.Kl = self._k_local()
        self.Kg = self.T.T @ self.Kl @ self.T

    @staticmethod
    def _rotation(c):
        cx, cy, cz = c
        if np.isclose(cx, 0.0) and np.isclose(cy, 0.0):
            s = np.sign(cz)
            return np.array([[0, 0, s], [0, 1, 0], [-s, 0, 0]], dtype=float)
        cxy = math.sqrt(cx**2 + cy**2)
        return np.array([[cx, cy, cz],
                         [-cy / cxy, cx / cxy, 0],
                         [-cx * cz / cxy, -cy * cz / cxy, cxy]])

    def _k_local(self):
        s, L = self.sec, self.length
        Gm = s.E / (2 * (1 + s.nu))
        ka, kt = s.E * s.A / L, Gm * s.J / L
        b1, b2 = 12 * s.E * s.I / L**3, 6 * s.E * s.I / L**2
        b3, b4 = 4 * s.E * s.I / L, 2 * s.E * s.I / L
        K11 = np.array([[ka, 0, 0, 0, 0, 0], [0, b1, 0, 0, 0, b2], [0, 0, b1, 0, -b2, 0],
                        [0, 0, 0, kt, 0, 0], [0, 0, -b2, 0, b3, 0], [0, b2, 0, 0, 0, b3]])
        K22 = np.array([[ka, 0, 0, 0, 0, 0], [0, b1, 0, 0, 0, -b2], [0, 0, b1, 0, b2, 0],
                        [0, 0, 0, kt, 0, 0], [0, 0, b2, 0, b3, 0], [0, -b2, 0, 0, 0, b3]])
        K12 = np.array([[-ka, 0, 0, 0, 0, 0], [0, -b1, 0, 0, 0, b2], [0, 0, -b1, 0, -b2, 0],
                        [0, 0, 0, -kt, 0, 0], [0, 0, b2, 0, b4, 0], [0, -b2, 0, 0, 0, b4]])
        return np.block([[K11, K12], [K12.T, K22]])

    # эквивалентные узловые нагрузки (в локальных осях элемента)
    def gravity_local(self):
        L = self.length
        wx, wy, wz = self.Lambda @ np.array([0.0, 0.0, -self.sec.q])
        f = np.zeros(12)
        f[0] = f[6] = wx * L / 2
        f[1] = f[7] = wy * L / 2
        f[5], f[11] = wy * L**2 / 12, -wy * L**2 / 12
        f[2] = f[8] = wz * L / 2
        f[4], f[10] = -wz * L**2 / 12, wz * L**2 / 12
        return f

    def thermal_local(self, dT):
        F = self.sec.E * self.sec.A * self.sec.alpha * dT
        f = np.zeros(12)
        f[0], f[6] = -F, F
        return f


# ==========================================================
# 2. РЕШАТЕЛЬ МКЭ
# ==========================================================

@dataclass
class Solution:
    U: np.ndarray
    R: np.ndarray
    eq: dict
    idx: dict


def solve(nodes, pipes, dT=0.0, gravity=False) -> Solution:
    idx = {id(n): i * 6 for i, n in enumerate(nodes)}
    N = 6 * len(nodes)
    K, F, eq = np.zeros((N, N)), np.zeros(N), {}
    for p in pipes:
        i1, i2 = idx[id(p.n1)], idx[id(p.n2)]
        dofs = np.r_[i1:i1 + 6, i2:i2 + 6]
        K[np.ix_(dofs, dofs)] += p.Kg
        f = np.zeros(12)
        if gravity:
            f += p.gravity_local()
        if dT:
            f += p.thermal_local(dT)
        F[dofs] += p.T.T @ f
        eq[id(p)] = f
    free = ~np.concatenate([n.fixed for n in nodes])
    U = np.zeros(N)
    try:
        U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    except np.linalg.LinAlgError as e:
        raise ValueError("Модель не закреплена (механизм): проверьте опоры") from e
    if not np.all(np.isfinite(U)) or np.abs(U).max() > 1e3:
        raise ValueError("Модель не закреплена (механизм): проверьте опоры")
    R = K @ U - F
    R[free] = 0.0
    return Solution(U, R, eq, idx)


def element_forces(sol: Solution, p: PipeElement):
    """Усилия на концах элемента в ЛОКАЛЬНЫХ осях: [N,Vy,Vz,Mt,My,Mz] x 2 конца."""
    i1, i2 = sol.idx[id(p.n1)], sol.idx[id(p.n2)]
    u = np.concatenate([sol.U[i1:i1 + 6], sol.U[i2:i2 + 6]])
    return p.Kl @ (p.T @ u) - sol.eq[id(p)]


def mesh(nodes, pipes, max_len=1.0):
    """Дробит проектную модель на мелкие элементы - чтобы видеть напряжения внутри пролёта."""
    mnodes, mp = [], {}
    for n in nodes:
        m = Node(n.nodeid, *n.coords, kind=n.kind, fixed=n.fixed)
        mnodes.append(m)
        mp[id(n)] = m
    mpipes = []
    for p in pipes:
        k = max(1, math.ceil(p.length / max_len - 1e-9))
        prev = mp[id(p.n1)]
        for i in range(1, k + 1):
            if i == k:
                cur = mp[id(p.n2)]
            else:
                c = p.n1.coords + (p.n2.coords - p.n1.coords) * i / k
                cur = Node(-1, *c)
                mnodes.append(cur)
            mpipes.append(PipeElement(prev, cur, p.sec, parent=p))
            prev = cur
    return mnodes, mpipes


# ==========================================================
# 3. ПРОВЕРКА: напряжения и прогиб
# ==========================================================

@dataclass
class Evaluation:
    util_sust: dict         # id(проектной трубы) -> S_L / Sh
    util_exp: dict          # id(проектной трубы) -> S_E / S_A
    max_SL: float
    max_SE: float
    sag: float
    anchors: list
    lim: Limits
    viz: dict = None        # данные для графиков (см. pipe_viz.py)

    @property
    def max_util(self):
        return max(max(self.util_sust.values()), max(self.util_exp.values()))

    @property
    def ok(self):
        return self.max_util <= 1.0 and self.sag <= self.lim.sag

    def pipe_util(self, p):
        return max(self.util_sust[id(p)], self.util_exp[id(p)])


def evaluate(nodes, pipes, dT, lim: Limits, mesh_len=1.0) -> Evaluation:
    mn, mp = mesh(nodes, pipes, mesh_len)
    sg = solve(mn, mp, gravity=True)          # вес + давление
    st = solve(mn, mp, dT=dT)                 # температура
    us = {id(p): 0.0 for p in pipes}
    ue = {id(p): 0.0 for p in pipes}
    maxSL = maxSE = 0.0
    el_u = {}
    for p in mp:
        s = p.sec
        el_u[id(p)] = [0.0, 0.0]
        fg, ft = element_forces(sg, p), element_forces(st, p)
        for sl in (slice(0, 6), slice(6, 12)):
            a, b = fg[sl], ft[sl]
            # длительные нагрузки: (осевое + давление + изгиб) и кручение (ср. с Eq.7 статьи)
            sb = math.hypot(a[4], a[5]) / s.W
            SL = math.hypot(abs(a[0]) / s.A + s.sigma_P + sb, abs(a[3]) / s.W)
            # температурные напряжения (осевая сила в B31.3 не входит)
            SE = math.hypot(math.hypot(b[4], b[5]) / s.W, abs(b[3]) / s.W)
            key = id(p.parent)
            us[key] = max(us[key], SL / lim.Sh)
            ue[key] = max(ue[key], SE / lim.SA)
            maxSL, maxSE = max(maxSL, SL), max(maxSE, SE)
            el_u[id(p)][0] = max(el_u[id(p)][0], SL / lim.Sh)
            el_u[id(p)][1] = max(el_u[id(p)][1], SE / lim.SA)
    sag = max(0.0, -float(sg.U[2::6].min()))
    anchors = []
    for n in mn:
        if n.kind == "anchor":
            i = sg.idx[id(n)]
            anchors.append(dict(id=n.nodeid,
                                weight_F=float(np.linalg.norm(sg.R[i:i + 3])),
                                therm_F=float(np.linalg.norm(st.R[i:i + 3])),
                                therm_M=float(np.linalg.norm(st.R[i + 3:i + 6]))))
    return Evaluation(us, ue, maxSL, maxSE, sag, anchors, lim,
                      viz=_collect_viz(mn, mp, sg, st, el_u))


def _collect_viz(mn, mp, sg, st, el_u):
    """Данные для графиков: элементы вдоль трассы (от первого анкера), опоры, перемещения."""
    adj = {}
    for p in mp:
        adj.setdefault(id(p.n1), []).append(p)
        adj.setdefault(id(p.n2), []).append(p)
    start = next(n for n in mn if n.kind == "anchor")
    cur, used, order = start, set(), []
    while True:
        cand = [p for p in adj.get(id(cur), []) if id(p) not in used]
        if not cand:
            break
        p = cand[0]
        used.add(id(p))
        order.append((p, p.n1 is cur))
        cur = p.n2 if p.n1 is cur else p.n1
    elements, node_s, s = [], {id(start): 0.0}, 0.0
    for p, fwd in order:
        a, b = (p.n1, p.n2) if fwd else (p.n2, p.n1)
        ia, ib = sg.idx[id(a)], sg.idx[id(b)]
        elements.append(dict(c1=a.coords.copy(), c2=b.coords.copy(),
                             d1=st.U[ia:ia + 3].copy(), d2=st.U[ib:ib + 3].copy(),
                             z1=float(sg.U[ia + 2]), z2=float(sg.U[ib + 2]),
                             s0=s, s1=s + p.length, uL=el_u[id(p)][0], uE=el_u[id(p)][1],
                             loop=p.parent.tag == "loop"))
        s += p.length
        node_s[id(b)] = s
    supports = []
    for n in mn:
        if n.kind:
            i = sg.idx[id(n)]
            supports.append(dict(kind=n.kind, s=node_s.get(id(n), float("nan")),
                                 coords=n.coords.copy(), Fz=float(sg.R[i + 2]),
                                 F_th=float(np.linalg.norm(st.R[i:i + 3]))))
    return dict(elements=elements, supports=supports, length=s)


def summary(nodes, pipes, ev: "Evaluation"):
    """Сводные показатели трассы (для таблиц и сравнения до/после)."""
    adj = {}
    for p in pipes:
        adj.setdefault(id(p.n1), []).append(p)
        adj.setdefault(id(p.n2), []).append(p)
    bends = 0
    for n in nodes:
        a = adj.get(id(n), [])
        if n.kind == "anchor" or len(a) != 2:
            continue
        v1 = (a[0].n1.coords + a[0].n2.coords) / 2 - n.coords
        v2 = (a[1].n1.coords + a[1].n2.coords) / 2 - n.coords
        if (v1 / np.linalg.norm(v1)) @ (v2 / np.linalg.norm(v2)) > -0.98:
            bends += 1
    return dict(
        length=sum(p.length for p in pipes),
        mass=sum(p.sec.A * p.sec.rho * p.length for p in pipes),
        bends=bends,
        loops=sum(1 for p in pipes if p.tag == "loop") // 3,
        guides=sum(1 for n in nodes if n.kind == "guide"),
        rests=sum(1 for n in nodes if n.kind == "rest"),
        max_util=ev.max_util, SL=ev.max_SL, SE=ev.max_SE, sag=ev.sag, ok=ev.ok,
        anchors=ev.anchors)


# ==========================================================
# 4. ТОПОЛОГИЯ: прямые участки, вставка узлов
# ==========================================================

@dataclass
class Run:
    start: Node
    end: Node
    pipes: list
    nodes: list
    length: float
    direction: np.ndarray


def straight_runs(nodes, pipes):
    """Разбиение трассы на прямые участки между анкерами, поворотами и тройниками."""
    adj = {id(n): [] for n in nodes}
    for p in pipes:
        adj[id(p.n1)].append((p, p.n2))
        adj[id(p.n2)].append((p, p.n1))
    special = set()
    for n in nodes:
        a = adj[id(n)]
        if n.kind == "anchor" or len(a) != 2:
            special.add(id(n))
            continue
        v1 = a[0][1].coords - n.coords
        v2 = a[1][1].coords - n.coords
        if (v1 / np.linalg.norm(v1)) @ (v2 / np.linalg.norm(v2)) > -0.98:   # изгиб > ~10 град
            special.add(id(n))
    runs, seen = [], set()
    for s in nodes:
        if id(s) not in special:
            continue
        for p, nxt in adj[id(s)]:
            if id(p) in seen:
                continue
            cp, cn, prev, cur = [p], [s, nxt], p, nxt
            seen.add(id(p))
            while id(cur) not in special:
                cand = [(q, m) for q, m in adj[id(cur)] if q is not prev]
                if not cand:
                    break
                q, m = cand[0]
                cp.append(q)
                cn.append(m)
                seen.add(id(q))
                prev, cur = q, m
            chord = cn[-1].coords - s.coords
            if np.linalg.norm(chord) < 1e-9:
                continue
            runs.append(Run(s, cn[-1], cp, cn, sum(x.length for x in cp),
                            chord / np.linalg.norm(chord)))
    return runs


def ensure_node_at(nodes, pipes, P, tol=1e-6):
    """Узел в точке P: существующий или новый (с разрезанием трубы)."""
    for n in nodes:
        if np.linalg.norm(n.coords - P) < tol:
            return n
    for p in pipes:
        a, b = p.n1.coords, p.n2.coords
        ab = b - a
        t = float(np.dot(P - a, ab) / np.dot(ab, ab))
        if 0 < t < 1 and np.linalg.norm(a + t * ab - P) < tol:
            new = Node(max(n.nodeid for n in nodes) + 1, *P)
            nodes.append(new)
            pipes.remove(p)
            pipes.extend([PipeElement(p.n1, new, p.sec), PipeElement(new, p.n2, p.sec)])
            return new
    raise ValueError("Точка не лежит на трассе")


def build_route(points, sec: Section):
    nodes = [Node(i + 1, *pt) for i, pt in enumerate(points)]
    make_anchor(nodes[0])
    make_anchor(nodes[-1])
    pipes = [PipeElement(a, b, sec) for a, b in zip(nodes, nodes[1:])]
    return nodes, pipes


# ==========================================================
# 5. ОПОРЫ ПО ВЕСУ
# ==========================================================

def max_span(sec: Section, lim: Limits):
    """Допустимый пролёт: min(по прочности, по прогибу) - те же формулы, что в старом split()."""
    q = sec.q
    L_stress = math.sqrt(8 * max(lim.Sh - sec.sigma_P, 1.0) * sec.W / q)
    L_sag = (384 * sec.E * sec.I * lim.sag / (5 * q)) ** 0.25
    return min(L_stress, L_sag)


def place_supports(nodes, pipes, lim: Limits):
    """Направляющие на горизонтальных участках через равные пролёты + скользящие под горизонтальными отводами."""
    for run in straight_runs(nodes, pipes):
        if abs(run.direction[2]) > 0.5:
            continue
        n = math.ceil(run.length / max_span(run.pipes[0].sec, lim) - 1e-9)
        for i in range(1, n):
            P = run.start.coords + run.direction * run.length * i / n
            node = ensure_node_at(nodes, pipes, P)
            if node.kind is None:
                make_guide(node, run.direction)
    # пролёт не должен "перепрыгивать" через горизонтальный отвод: ставим под него опору
    adj = {}
    for p in pipes:
        adj.setdefault(id(p.n1), []).append(p)
        adj.setdefault(id(p.n2), []).append(p)
    for n in nodes:
        a = adj.get(id(n), [])
        if n.kind is not None or len(a) != 2:
            continue
        v1 = (a[0].n1.coords + a[0].n2.coords) / 2 - n.coords
        v2 = (a[1].n1.coords + a[1].n2.coords) / 2 - n.coords
        u1, u2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        if abs(u1[2]) < 0.5 and abs(u2[2]) < 0.5 and u1 @ u2 > -0.98:
            make_rest(n)


# ==========================================================
# 6. КОМПЕНСАТОР
# ==========================================================

def _loop_direction(d, plane):
    if abs(d[2]) > 0.7:                                  # вертикальная труба -> петля вдоль x
        return np.array([1.0, 0.0, 0.0])
    if plane == "vertical":
        return np.array([0.0, 0.0, 1.0])
    o = np.cross([0.0, 0.0, 1.0], d)                     # горизонтальная петля (нет воздушных пробок)
    return o / np.linalg.norm(o)


def insert_loop(nodes, pipes, run: Run, W=3.0, H=4.0, plane="horizontal"):
    """П-компенсатор шириной W и вылетом H в середине прямого участка."""
    d = run.direction
    ortho = _loop_direction(d, plane)
    mid = run.start.coords + d * run.length / 2
    n1 = ensure_node_at(nodes, pipes, mid - d * W / 2)
    n4 = ensure_node_at(nodes, pipes, mid + d * W / 2)
    t1 = float(np.dot(n1.coords - run.start.coords, d))
    t4 = float(np.dot(n4.coords - run.start.coords, d))

    def on_run(n):
        r = n.coords - run.start.coords
        return np.linalg.norm(r - d * np.dot(r, d)) < 1e-6

    inside = [p for p in pipes if on_run(p.n1) and on_run(p.n2)
              and min(np.dot(p.n1.coords - run.start.coords, d), np.dot(p.n2.coords - run.start.coords, d)) >= t1 - 1e-6
              and max(np.dot(p.n1.coords - run.start.coords, d), np.dot(p.n2.coords - run.start.coords, d)) <= t4 + 1e-6]
    drop_nodes = {id(n) for p in inside for n in (p.n1, p.n2)} - {id(n1), id(n4)}
    for p in inside:
        pipes.remove(p)
    nodes[:] = [n for n in nodes if id(n) not in drop_nodes]

    base = max(n.nodeid for n in nodes) + 1
    n2 = Node(base, *(n1.coords + ortho * H))
    n3 = Node(base + 1, *(n4.coords + ortho * H))
    nodes.extend([n2, n3])
    sec = run.pipes[0].sec
    loop_pipes = [PipeElement(n1, n2, sec), PipeElement(n2, n3, sec), PipeElement(n3, n4, sec)]
    for lp in loop_pipes:
        lp.tag = 'loop'
    pipes.extend(loop_pipes)
    make_guide(n1, d)
    make_guide(n4, d)
    if abs(ortho[2]) < 0.5:          # горизонтальные плечи не висят в воздухе: опоры под углами петли
        make_rest(n2)
        make_rest(n3)


def run_util(run: Run, ev: Evaluation):
    return max(ev.pipe_util(p) for p in run.pipes)


def add_compensators(nodes, pipes, dT, lim: Limits, max_loops=3, W=3.0,
                     heights=(4.0, 5.0, 6.0, 8.0, 10.0), plane="horizontal",
                     min_gain=0.05, log=print):
    """Цикл: самый нагруженный участок -> компенсатор -> пересчёт.

    Для участка перебираем вылет петли H (из heights) и берём наименьший, при котором
    загрузка <= 1; если такого нет - лучший из вариантов. Вставка остаётся, только если
    загрузка упала минимум на min_gain и прогиб не вышел за допуск; иначе пробуем
    следующий по нагрузке участок."""
    ev = evaluate(nodes, pipes, dT, lim)
    history = [ev.max_util]
    for it in range(1, max_loops + 1):
        if ev.max_util <= 1.0:
            break
        ranked = sorted(straight_runs(nodes, pipes), key=lambda r: -run_util(r, ev))
        improved = False
        for r in ranked:
            if r.length < W + 1.0:
                continue                                  # не помещается петля
            variants = []
            for H in heights:
                tn, tp = copy.deepcopy((nodes, pipes))
                tr = next(x for x in straight_runs(tn, tp)
                          if x.start.nodeid == r.start.nodeid and x.end.nodeid == r.end.nodeid)
                insert_loop(tn, tp, tr, W, H, plane)
                variants.append((H, tn, tp, evaluate(tn, tp, dT, lim)))
            good = [v for v in variants if v[3].max_util <= 1.0 and v[3].sag <= max(lim.sag, ev.sag)]
            best = min(good, key=lambda v: v[0]) if good else min(variants, key=lambda v: v[3].max_util)
            H, tn, tp, tev = best
            gain_ok = tev.max_util < ev.max_util * (1 - min_gain)
            sag_ok = tev.sag <= max(lim.sag, ev.sag)       # не ухудшаем прогиб сверх допуска
            if gain_ok and sag_ok:
                log(f"  итерация {it}: участок {r.start.nodeid}->{r.end.nodeid} "
                    f"(L={r.length:.1f} м, загрузка {run_util(r, ev):.2f}) -> компенсатор "
                    f"{W:.0f}x{H:.0f} м; максимум {ev.max_util:.2f} -> {tev.max_util:.2f}")
                nodes[:], pipes[:] = tn, tp
                ev, improved = tev, True
                history.append(ev.max_util)
                break
        if not improved:
            log(f"  итерация {it}: ни один компенсатор не улучшает результат - остановка")
            break
    return ev, history


# ==========================================================
# 7. ОТЧЁТ И ГРАФИК
# ==========================================================

def report(nodes, pipes, ev: Evaluation, title):
    print(f"\n--- {title} ---")
    print(f"  напряжение длительное  S_L max = {ev.max_SL/1e6:7.1f} МПа (допуск {ev.lim.Sh/1e6:.0f})")
    print(f"  напряжение температурное S_E max = {ev.max_SE/1e6:7.1f} МПа (допуск {ev.lim.SA/1e6:.0f})")
    print(f"  максимальная загрузка = {ev.max_util:.2f}   прогиб = {ev.sag*1000:.1f} мм "
          f"(допуск {ev.lim.sag*1000:.0f})   => {'ПРОХОДИТ' if ev.ok else 'НЕ ПРОХОДИТ'}")
    for r in sorted(straight_runs(nodes, pipes), key=lambda r: -run_util(r, ev))[:4]:
        print(f"  участок {r.start.nodeid:>3}->{r.end.nodeid:<3} L={r.length:5.1f} м  "
              f"загрузка {run_util(r, ev):.2f}")
    for a in ev.anchors:
        print(f"  анкер {a['id']}: от веса {a['weight_F']:,.0f} Н; от температуры "
              f"{a['therm_F']:,.0f} Н, момент {a['therm_M']:,.0f} Н*м")


def plot_route(nodes, pipes, title="", path=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for p in pipes:
        c = np.vstack([p.n1.coords, p.n2.coords])
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color="#1f77b4", linewidth=4)
    for kind, color, marker, label in (("anchor", "red", "s", "Анкер"),
                                       ("guide", "green", "^", "Направляющая"),
                                       ("rest", "orange", "v", "Скользящая")):
        pts = np.array([n.coords for n in nodes if n.kind == kind])
        if len(pts):
            ax.scatter(*pts.T, color=color, marker=marker, s=90, label=label)
    ax.set_xlabel("X, м"); ax.set_ylabel("Y, м"); ax.set_zlabel("Z, м")
    ax.set_title(title); ax.legend()
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ==========================================================
# 8. ДЕМО
# ==========================================================

def design(points, sec: Section, lim: Limits, dT, title, plot_prefix=None):
    nodes, pipes = build_route(points, sec)
    place_supports(nodes, pipes, lim)
    ev0 = evaluate(nodes, pipes, dT, lim)
    nodes0, pipes0 = copy.deepcopy((nodes, pipes))          # снимок "до"
    report(nodes, pipes, ev0, f"{title}: ДО")
    if plot_prefix:
        plot_route(nodes, pipes, f"{title}: до", f"{plot_prefix}_before.png")
    ev1, hist = add_compensators(nodes, pipes, dT, lim)
    if len(hist) > 1:
        report(nodes, pipes, ev1, f"{title}: ПОСЛЕ")
        if plot_prefix:
            plot_route(nodes, pipes, f"{title}: после", f"{plot_prefix}_after.png")
    return dict(nodes=nodes, pipes=pipes, ev_after=ev1, nodes_before=nodes0,
                pipes_before=pipes0, ev_before=ev0, dT=dT, lim=lim)


if __name__ == "__main__":
    sec = Section(D=0.219, d=0.203)
    lim = Limits()
    print(f"Допустимый пролёт: {max_span(sec, lim):.1f} м, погонный вес {sec.q:.0f} Н/м")

    # Случай 1: геометрия из старого сценария - проходит без компенсатора
    design([(0, 0, 0), (40, 0, 0), (40, 10, 0), (40, 10, 8)], sec, lim, 100, "Случай 1 (старая трасса)")

    # Случай 2: длинное плечо и короткая жёсткая перемычка - проверка должна не пройти
    design([(0, 0, 0), (60, 0, 0), (60, 3, 0), (60, 3, 4)], sec, lim, 150, "Случай 2 (жёсткая трасса)")
