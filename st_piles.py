import numpy as np
import streamlit as st
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
from scipy.spatial import distance_matrix, Delaunay, cKDTree
import math
import tempfile
import os
import ezdxf
import csv
import time
from io import StringIO, BytesIO

# Настройки страницы
st.set_page_config(page_title="Создание свайного поля", layout="wide")

# Исходные данные по умолчанию
default_vertices = [
    [0, 0],
    [0, 19500],
    [10890, 19500],
    [27118.1006, 3271.8994],
    [16299.3669, -7546.8344],
    [8752.5325, 0]
]


# =============================================
# ФУНКЦИИ ГЕНЕРАЦИИ СЕТОК
# =============================================

def generate_hexagonal_grid(poly, min_dist, target_points):
    step = min_dist
    min_x, min_y, max_x, max_y = poly.bounds

    points = []
    y = min_y
    even = False

    while y <= max_y:
        x = min_x if not even else min_x + step / 2
        while x <= max_x:
            point = Point(x, y)
            if poly.contains(point):
                points.append([x, y])
            x += step
        y += step * math.sqrt(3) / 2
        even = not even

    points = np.array(points)

    if len(points) > 1:
        tree = cKDTree(points)
        pairs = tree.query_pairs(min_dist)

        if pairs:
            displacement = np.zeros_like(points)
            for i, j in pairs:
                vec = points[i] - points[j]
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    direction = vec / (dist + 1e-6)
                    displacement[i] += direction * (min_dist - dist) / 2
                    displacement[j] -= direction * (min_dist - dist) / 2

            new_points = points + displacement * 0.3
            for i in range(len(new_points)):
                if poly.contains(Point(new_points[i])):
                    points[i] = new_points[i]

    return points


def generate_rectangular_grid(poly, min_dist, target_points):
    min_x, min_y, max_x, max_y = poly.bounds
    area = poly.area
    density = target_points / area
    step = math.sqrt(1 / density)

    points = []
    y = min_y
    while y <= max_y:
        x = min_x
        while x <= max_x:
            point = Point(x, y)
            if poly.contains(point):
                points.append([x, y])
            x += step
        y += step

    points = np.array(points)

    if len(points) > 1:
        tree = cKDTree(points)
        pairs = tree.query_pairs(min_dist)

        if pairs:
            displacement = np.zeros_like(points)
            for i, j in pairs:
                vec = points[i] - points[j]
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    direction = vec / (dist + 1e-6)
                    displacement[i] += direction * (min_dist - dist) / 2
                    displacement[j] -= direction * (min_dist - dist) / 2

            new_points = points + displacement * 0.3
            for i in range(len(new_points)):
                if poly.contains(Point(new_points[i])):
                    points[i] = new_points[i]

    return points


def generate_offset_rect_grid(poly, min_dist, target_points):
    min_x, min_y, max_x, max_y = poly.bounds
    area = poly.area
    density = target_points / area
    step = math.sqrt(1 / density)

    points = []
    even_row = False
    y = min_y
    while y <= max_y:
        x = min_x if not even_row else min_x + step / 2
        while x <= max_x:
            point = Point(x, y)
            if poly.contains(point):
                points.append([x, y])
            x += step
        y += step
        even_row = not even_row

    points = np.array(points)

    if len(points) > 1:
        tree = cKDTree(points)
        pairs = tree.query_pairs(min_dist)

        if pairs:
            displacement = np.zeros_like(points)
            for i, j in pairs:
                vec = points[i] - points[j]
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    direction = vec / (dist + 1e-6)
                    displacement[i] += direction * (min_dist - dist) / 2
                    displacement[j] -= direction * (min_dist - dist) / 2

            new_points = points + displacement * 0.3
            for i in range(len(new_points)):
                if poly.contains(Point(new_points[i])):
                    points[i] = new_points[i]

    return points


def generate_triangular_grid(poly, min_dist, target_points):
    points = generate_rectangular_grid(poly, min_dist * 1.2, target_points)

    if len(points) < 3:
        return points

    tri = Delaunay(points)

    new_points = []
    for i, p in enumerate(points):
        neighbors = set()
        for simplex in tri.simplices:
            if i in simplex:
                neighbors.update(simplex)
        neighbors.discard(i)

        if neighbors:
            displacement = np.zeros(2)
            for n in neighbors:
                vec = points[n] - p
                displacement += vec / np.linalg.norm(vec)
            displacement /= len(neighbors)

            new_p = p + displacement * min_dist * 0.3
            if poly.contains(Point(new_p)):
                new_points.append(new_p)
            else:
                new_points.append(p)
        else:
            new_points.append(p)

    points = np.array(new_points)

    if len(points) > 1:
        tree = cKDTree(points)
        pairs = tree.query_pairs(min_dist)

        if pairs:
            displacement = np.zeros_like(points)
            for i, j in pairs:
                vec = points[i] - points[j]
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    direction = vec / (dist + 1e-6)
                    displacement[i] += direction * (min_dist - dist) / 2
                    displacement[j] -= direction * (min_dist - dist) / 2

            new_points = points + displacement * 0.2
            for i in range(len(new_points)):
                if poly.contains(Point(new_points[i])):
                    points[i] = new_points[i]

    return points


def generate_spiral_grid(poly, min_dist, target_points):
    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y

    max_radius = 0
    for x, y in poly.exterior.coords:
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist > max_radius:
            max_radius = dist

    points = []
    n = target_points
    for k in range(1, n + 1):
        radius = max_radius * math.sqrt(k / n)
        angle = 2 * np.pi * (1 + math.sqrt(5)) / 2 * k
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        point = Point(x, y)
        if poly.contains(point):
            points.append([x, y])

    points = np.array(points)

    if len(points) > 1:
        tree = cKDTree(points)
        pairs = tree.query_pairs(min_dist)

        if pairs:
            displacement = np.zeros_like(points)
            for i, j in pairs:
                vec = points[i] - points[j]
                dist = np.linalg.norm(vec)
                if dist < min_dist:
                    direction = vec / (dist + 1e-6)
                    displacement[i] += direction * (min_dist - dist) / 2
                    displacement[j] -= direction * (min_dist - dist) / 2

            new_points = points + displacement * 0.3
            for i in range(len(new_points)):
                if poly.contains(Point(new_points[i])):
                    points[i] = new_points[i]

    return points

# =============================================
# ИНТЕРФЕЙС STREAMLIT
# =============================================

st.title("Создание свайного поля")

# Основные колонки
col1, col2 = st.columns([1, 2])

with col1:
    with st.expander("Исходные данные", expanded=True):
        input_method = st.radio("Метод ввода:", ["Ручной ввод", "Загрузка DXF"])

        if input_method == "Ручной ввод":
            num_vertices = st.number_input("Количество вершин:", min_value=3, max_value=20, value=6)
            vertices = []
            for i in range(num_vertices):
                col_x, col_y = st.columns(2)
                with col_x:
                    x = st.number_input(f"X{i + 1}", value=default_vertices[i][0] if i < len(default_vertices) else 0.0)
                with col_y:
                    y = st.number_input(f"Y{i + 1}", value=default_vertices[i][1] if i < len(default_vertices) else 0.0)
                vertices.append([x, y])
        else:
            dxf_file = st.file_uploader("Загрузите DXF файл", type=["dxf"])
            vertices = []
            if dxf_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                        tmp.write(dxf_file.getvalue())
                        tmp_path = tmp.name

                    doc = ezdxf.readfile(tmp_path)
                    msp = doc.modelspace()

                    polylines = []
                    for entity in msp:
                        if entity.dxftype() in ('POLYLINE', 'LWPOLYLINE'):
                            polylines.append(entity)

                    if polylines:
                        polyline = polylines[0]
                        if polyline.dxftype() == 'LWPOLYLINE':
                            for point in polyline.get_points():
                                vertices.append([point[0], point[1]])
                        else:
                            for vertex in polyline.vertices():
                                vertices.append([vertex.dxf.location.x, vertex.dxf.location.y])

                        st.success(f"Загружено {len(vertices)} вершин из DXF")
                    else:
                        st.error("Отсутствует полилиния в DXF файле")

                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Ошибка загрузки файла DXF: {str(e)}")

    with st.expander("Параметры распределения", expanded=True):
        method = st.selectbox(
            "Метод распределения:",
            ["Шестиугольная сетка", "Прямоугольная сетка",
             "Квадратно-гнездовое распределение", "Треугольная сетка",
             "Спиральное распределение"]
        )

        points_count = st.number_input("Количество свай:", min_value=1, value=225)
        min_distance = st.number_input("Минимальное расстояние:", min_value=0.1, value=1200.0)
        border_offset = st.number_input("Отступ от контура:", min_value=0.0, value=600.0)

        if st.button("Выполнить построение", type="primary"):
            st.session_state.calculate = True
            st.session_state.vertices = vertices
            st.session_state.method = method
            st.session_state.points_count = points_count
            st.session_state.min_distance = min_distance
            st.session_state.border_offset = border_offset

    with st.expander("Экспорт результатов", expanded=True):
        if 'calculate' in st.session_state and st.session_state.calculate and hasattr(st.session_state, 'pile_points'):
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                if st.button("Экспорт в DXF"):
                    try:
                        doc = ezdxf.new('R2010')
                        msp = doc.modelspace()

                        # Добавляем контур фундамента
                        msp.add_lwpolyline(st.session_state.vertices, close=True)

                        # Добавляем точки свай
                        for point in st.session_state.pile_points:
                            msp.add_point((point[0], point[1]))

                        # Сохраняем DXF во временный файл
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
                            doc.saveas(tmp_file.name)
                            tmp_file_path = tmp_file.name

                        # Читаем временный файл и передаем в Streamlit
                        with open(tmp_file_path, "rb") as f:
                            dxf_bytes = f.read()

                        # Удаляем временный файл
                        os.unlink(tmp_file_path)

                        # Предлагаем скачать DXF
                        st.download_button(
                            label="Скачать DXF",
                            data=dxf_bytes,
                            file_name="pile_field.dxf",
                            mime="application/dxf"
                        )
                    except Exception as e:
                        st.error(f"Ошибка при экспорте в DXF: {str(e)}")

            with col_exp2:
                if st.button("Экспорт в CSV"):
                    try:
                        csv_data = StringIO()
                        writer = csv.writer(csv_data)
                        writer.writerow(['X', 'Y'])
                        for point in st.session_state.pile_points:
                            writer.writerow([point[0], point[1]])

                        st.download_button(
                            label="Скачать CSV",
                            data=csv_data.getvalue(),
                            file_name="pile_field.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Ошибка при экспорте в CSV: {str(e)}")
        else:
            st.info("Выполните расчёт для экспорта результатов")

with col2:
    # Область для графика
    if 'calculate' in st.session_state and st.session_state.calculate:
        vertices = st.session_state.vertices
        method = st.session_state.method
        points_count = st.session_state.points_count
        min_distance = st.session_state.min_distance
        border_offset = st.session_state.border_offset

        if len(vertices) < 3:
            st.error("Полигон должен содержать минимум 3 вершины")
        else:
            with st.spinner("Выполняется расчет..."):
                start_time = time.time()

                outer_poly = Polygon(vertices)
                inner_poly = outer_poly.buffer(-border_offset, join_style=2)

                if inner_poly.is_empty:
                    st.error("Контур размещения пуст. Измените значение отступа.")
                else:
                    method_map = {
                        "Шестиугольная сетка": "hexagonal",
                        "Прямоугольная сетка": "rectangular",
                        "Квадратно-гнездовое распределение": "offset_rect",
                        "Треугольная сетка": "triangular",
                        "Спиральное распределение": "spiral"
                    }

                    method_key = method_map[method]

                    if method_key == "hexagonal":
                        points = generate_hexagonal_grid(inner_poly, min_distance, points_count)
                    elif method_key == "rectangular":
                        points = generate_rectangular_grid(inner_poly, min_distance, points_count)
                    elif method_key == "offset_rect":
                        points = generate_offset_rect_grid(inner_poly, min_distance, points_count)
                    elif method_key == "triangular":
                        points = generate_triangular_grid(inner_poly, min_distance, points_count)
                    elif method_key == "spiral":
                        points = generate_spiral_grid(inner_poly, min_distance, points_count)
                    else:
                        points = np.array([])

                    # Сохраняем точки для экспорта
                    st.session_state.pile_points = points

                    # Проверка минимального расстояния между точками
                    min_dist_actual = 0
                    if len(points) > 1:
                        dist_matrix = distance_matrix(points, points)
                        np.fill_diagonal(dist_matrix, np.inf)
                        min_dist_actual = np.min(dist_matrix)

                    # Создаем фигуру Plotly
                    fig = go.Figure()

                    # Добавляем контур фундамента
                    x_outer, y_outer = zip(*vertices)
                    fig.add_trace(go.Scatter(
                        x=list(x_outer) + [x_outer[0]],
                        y=list(y_outer) + [y_outer[0]],
                        mode="lines",
                        line=dict(color='#4285F4', width=3),
                        name="Контур фундамента",
                        fill="toself",
                        fillcolor='rgba(66, 133, 244, 0.1)'
                    ))

                    # Добавляем контур размещения
                    if inner_poly.geom_type == 'Polygon':
                        x_inner, y_inner = zip(*inner_poly.exterior.coords)
                        fig.add_trace(go.Scatter(
                            x=x_inner,
                            y=y_inner,
                            mode="lines",
                            line=dict(color='#EA4335', width=2, dash='dash'),
                            name="Контур размещения",
                            fill="toself",
                            fillcolor='rgba(234, 67, 53, 0.05)'
                        ))
                    elif inner_poly.geom_type == 'MultiPolygon':
                        for geom in inner_poly.geoms:
                            x_inner, y_inner = zip(*geom.exterior.coords)
                            fig.add_trace(go.Scatter(
                                x=x_inner,
                                y=y_inner,
                                mode="lines",
                                line=dict(color='#EA4335', width=2, dash='dash'),
                                name="Контур размещения",
                                fill="toself",
                                fillcolor='rgba(234, 67, 53, 0.05)'
                            ))

                    # Добавляем сваи
                    if len(points) > 0:
                        fig.add_trace(go.Scatter(
                            x=points[:, 0],
                            y=points[:, 1],
                            mode="markers",
                            marker=dict(color='#34A853', size=8),
                            name=f"Сваи ({len(points)})"
                        ))

                    # Добавляем подписи вершин
                    for i, (x, y) in enumerate(vertices):
                        fig.add_annotation(
                            x=x,
                            y=y,
                            text=str(i + 1),
                            showarrow=False,
                            font=dict(color='#4285F4', size=12),
                            bordercolor="#4285F4",
                            borderwidth=1,
                            borderpad=4,
                            bgcolor="white",
                            opacity=0.8
                        )

                    # Настраиваем макет с фиксированными пропорциями
                    fig.update_layout(
                        title=dict(
                            text=f"<b>Метод: {method}</b><br>"
                                 f"Сваи: {len(points)} (требование: {points_count})<br>"
                                 f"Минимальное расстояние: {min_dist_actual:.2f} (требование ≥ {min_distance})",
                            x=0.5,
                            xanchor="center"
                        ),
                        xaxis_title="X координата",
                        yaxis_title="Y координата",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=20, r=20, t=100, b=20),
                        hovermode='closest',
                        autosize=True,
                        width=800,
                        height=600,
                        # Фиксируем пропорции осей
                        yaxis=dict(
                            scaleanchor="x",
                            scaleratio=1
                        )
                    )

                    # Отображаем график
                    st.plotly_chart(fig, use_container_width=True)

                    # Статистика
                    st.success(f"Расчет завершен за {time.time() - start_time:.2f} сек")
    else:
        st.info("Введите параметры и нажмите 'Выполнить построение'")
