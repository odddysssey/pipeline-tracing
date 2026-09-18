import trimesh
import plotly.graph_objects as go

# Укажите ваш файл
FILE_NAME = "grisha.stl" 

mesh = trimesh.load(FILE_NAME)

# Если модель в минусовых координатах, сдвигаем ее в (0,0,0) как в основном скрипте
min_b = mesh.bounds[0]
if any(min_b < 0):
    mesh.apply_translation(-min_b)

vertices = mesh.vertices
faces = mesh.faces

fig = go.Figure(data=[go.Mesh3d(
    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    color='lightgray', opacity=0.5
)])

fig.update_layout(
    title="Наведите мышку на модель, чтобы узнать координаты X, Y, Z",
    scene=dict(
        xaxis=dict(tickformat=".0f"), # Отключает сокращения (k, M) и убирает нули после запятой
        yaxis=dict(tickformat=".0f"),
        zaxis=dict(tickformat=".0f"),
        aspectmode='data' # Гарантирует правильные пропорции комнаты (чтобы она не растягивалась)
    )
)
fig.write_html('viewer.html')
print("Готово! Откройте файл viewer.html и найдите нужные координаты.")