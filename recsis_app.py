import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO

# Конфигурация путей
class PathConfig:
    BASE_DIR = Path(r"C:\Users\krask\Downloads")
    ARCHIVE_DIR = BASE_DIR / "archive-20250423T214020Z-001" / "archive" / "fashion-dataset"

    IMAGES = ARCHIVE_DIR / "fashion-dataset" / "images"
    STYLES = ARCHIVE_DIR / "styles.csv"
    FEATURES = BASE_DIR / "features (1).npy"
    COSINE_SIM = BASE_DIR / "cosine_sim (1).npy"


# Функция загрузки изображений из локальной папки
def load_image_from_path(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path)
        return img.resize(target_size)
    except Exception as e:
        st.error(f"Ошибка загрузки изображения из пути {img_path}: {e}")
        return None


@st.cache_data
def load_data(max_images=1000):
    config = PathConfig()

    # Загрузка styles.csv
    df = pd.read_csv(config.STYLES, on_bad_lines='warn')
    df['id'] = df['id'].astype(str)
    df['image_id'] = df['id'] + '.jpg'

    # Загрузка признаков
    features = np.load(config.FEATURES)

    # Загрузка матрицы косинусного сходства
    cosine_sim = np.load(config.COSINE_SIM)

    # Получаем список image_id в порядке файлов из папки
    image_files = [p.name for p in sorted(config.IMAGES.glob("*.jpg")) if p.name in df['image_id'].values]

    # Фильтруем DataFrame, оставляя только image_id из image_files
    df_filtered = df[df['image_id'].isin(image_files)].copy()

    # Сортируем df_filtered по порядку image_files
    df_filtered['image_id'] = pd.Categorical(df_filtered['image_id'], categories=image_files, ordered=True)
    df_filtered = df_filtered.sort_values('image_id').reset_index(drop=True)

    # Создаем словарь image_to_index: image_id -> индекс (позиция в features и cosine_sim)
    image_to_index = {image_id: idx for idx, image_id in enumerate(image_files)}

    # Создаем словарь с информацией о стилях
    styles_dict = df_filtered.set_index('image_id').to_dict('index')

    # Создаем словарь с путями к изображениям
    image_paths = {image_id: str(config.IMAGES / image_id) for image_id in image_files}

    # Проверка соответствия
    for idx, image_id in enumerate(image_files):
        df_image_id = df_filtered.iloc[idx]['image_id']
        if image_id != df_image_id:
            raise ValueError(f"Несоответствие индексов: image_id '{image_id}' != df image_id '{df_image_id}' на позиции {idx}")
    st.info("Проверка соответствия индексов прошла успешно.")

    return df_filtered, features, image_paths, cosine_sim, styles_dict, image_to_index


def show_recommendations(selected_id, df, image_paths, cosine_sim, styles_dict, image_to_index):
    try:
        image_id = str(selected_id) + '.jpg'

        if image_id not in styles_dict:
            st.error(f"Товар с ID {selected_id} не найден в данных.")
            return

        if image_id not in image_to_index:
            st.error(f"Для товара с ID {selected_id} нет данных о признаках.")
            return

        product_info = styles_dict[image_id]
        idx = image_to_index[image_id]

        similarity_scores = cosine_sim[idx]
        similar_indices = np.argsort(similarity_scores)[::-1][1:5]

        st.subheader("✨ Рекомендуемые товары")
        cols = st.columns(5)

        with cols[0]:
            img_path = image_paths[image_id]
            img = load_image_from_path(img_path)
            if img:
                st.image(img, use_container_width=True, caption="Выбранный товар")
                st.caption(f"ID: {product_info['id']}")
                st.caption(f"{product_info['articleType']} ({product_info['baseColour']})")
                st.caption(f"Пол: {product_info['gender']}")
                st.caption(f"Сезон: {product_info['season']}")

        for i, neighbor_idx in enumerate(similar_indices):
            with cols[i + 1]:
                try:
                    neighbor_image_id = df.iloc[neighbor_idx]['image_id']
                    neighbor_info = styles_dict[neighbor_image_id]

                    img_path = image_paths[neighbor_image_id]
                    img = load_image_from_path(img_path)

                    if img:
                        st.image(img, use_container_width=True,
                                 caption=f"Схожесть: {similarity_scores[neighbor_idx]:.2f}")
                        st.caption(f"ID: {neighbor_info['id']}")
                        st.caption(f"{neighbor_info['articleType']} ({neighbor_info['baseColour']})")
                        st.caption(f"Пол: {neighbor_info['gender']}")
                        st.caption(f"Сезон: {neighbor_info['season']}")
                except Exception as e:
                    st.error(f"Ошибка отображения рекомендации: {str(e)}")

        with st.expander("📝 Подробная информация о товаре"):
            st.json({
                "ID": int(product_info['id']),
                "Пол": product_info['gender'],
                "Категория": product_info['articleType'],
                "Тип": product_info['masterCategory'],
                "Цвет": product_info['baseColour'],
                "Сезон": product_info['season']
            })

    except Exception as e:
        st.error(f"Ошибка при поиске рекомендаций: {str(e)}")


st.set_page_config(page_title="👗 Рекомендации модных товаров", layout="wide")
st.title('👗 Рекомендательная система модных товаров')

try:
    df, features, image_paths, cosine_sim, styles_dict, image_to_index = load_data()

    num_images_to_display = 30
    df_limited = df.head(num_images_to_display).copy()

    with st.sidebar:
        st.header('🔍 Фильтры')

        gender = st.selectbox('Пол', ['Все'] + sorted(df_limited['gender'].dropna().unique()))
        category = st.selectbox('Категория', ['Все'] + sorted(df_limited['articleType'].dropna().unique()))
        color = st.selectbox('Цвет', ['Все'] + sorted(df_limited['baseColour'].dropna().unique()))
        season = st.selectbox('Сезон', ['Все'] + sorted(df_limited['season'].dropna().unique()))

    filtered_df = df_limited.copy()
    if gender != 'Все':
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    if category != 'Все':
        filtered_df = filtered_df[filtered_df['articleType'] == category]
    if color != 'Все':
        filtered_df = filtered_df[filtered_df['baseColour'] == color]
    if season != 'Все':
        filtered_df = filtered_df[filtered_df['season'] == season]

    st.subheader("🔎 Отфильтрованные товары")

    num_cols = 4
    cols = st.columns(num_cols)
    col_idx = 0
    displayed_count = 0

    for _, row in filtered_df.iterrows():
        if displayed_count >= num_images_to_display:
            break

        image_id = row['image_id']
        if image_id in image_paths:
            with cols[col_idx]:
                try:
                    img_path = image_paths[image_id]
                    img = load_image_from_path(img_path)

                    if st.button(f"ID: {row['id']}", key=f"btn_{row['id']}"):
                        st.session_state.selected_id = str(row['id'])

                    if img:
                        st.image(img, use_container_width=True)
                        st.caption(f"{row['articleType']} ({row['baseColour']})")
                        st.caption(f"Пол: {row['gender']}")
                        st.caption(f"Сезон: {row['season']}")

                    col_idx = (col_idx + 1) % num_cols
                    displayed_count += 1

                except Exception as e:
                    st.error(f"Ошибка отображения товара ID {row['id']}: {str(e)}")

    if 'selected_id' in st.session_state:
        show_recommendations(
            st.session_state.selected_id,
            df,
            image_paths,
            cosine_sim,
            styles_dict,
            image_to_index
        )
    else:
        st.warning("Нет товаров, соответствующих выбранным фильтрам")

except Exception as e:
    st.error(f"Ошибка при загрузке данных: {str(e)}")
    st.error("Проверьте:")
    st.error("1. Все ли файлы находятся в указанных папках?")
    st.error("2. Корректны ли пути в настройках PathConfig?")
