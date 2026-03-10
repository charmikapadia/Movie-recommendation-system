import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA & TRAIN MODEL (CACHED)
# --------------------------------------------------

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("recommend_strict.csv")

    df = df.rename(columns={
        "movie_name": "title",
        "year": "release_year",
        "votes": "vote_count"
    })

    df["overview"] = df["overview"].fillna("")
    df["genre"] = df["genre"].fillna("")
    df["rating"] = df["rating"].fillna(0)
    df["vote_count"] = df["vote_count"].fillna(0)

    # 🔴 CRITICAL FIX
    df["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    )
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    df["combined_features"] = (
        df["title"] + " " +
        df["genre"] + " " +
        df["overview"]
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2
    )

    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return df, similarity_matrix


# Load model
try:
    df, similarity_matrix = load_and_train_model()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------
def get_recommendations(
    movie_name,
    search_by,
    min_rating,
    min_votes,
    genre_filter,
    year_range,
    top_n
):
    result_df = df.copy()

    # ---- SIMILARITY MODE ----
    if search_by == "Similarity":
        matches = df[df["title"].str.lower() == movie_name.lower()]

        if matches.empty:
            return None, "Movie not found."

        idx = matches.index[0]

        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in scores[1:51]]
        similarity_scores = [i[1] for i in scores[1:51]]

        result_df = result_df.iloc[movie_indices].copy()
        result_df["similarity_score"] = similarity_scores
        result_df = result_df.sort_values(
            by="similarity_score",
            ascending=False
        )

    # ---- FILTERS ----
    if genre_filter != "All":
        result_df = result_df[
            result_df["genre"].str.contains(
                genre_filter,
                case=False,
                na=False
            )
        ]

    start_year, end_year = year_range
    result_df = result_df[
        (result_df["release_year"] >= start_year) &
        (result_df["release_year"] <= end_year)
    ]

    result_df = result_df[result_df["rating"] >= min_rating]
    result_df = result_df[result_df["vote_count"] >= min_votes]

    # ---- SORTING FOR OTHER MODES ----
    if search_by == "Top Rated":
        result_df = result_df.sort_values(
            by="rating",
            ascending=False
        )
    elif search_by == "Popularity":
        result_df = result_df.sort_values(
            by="vote_count",
            ascending=False
        )

    return result_df.head(top_n), None

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("### BCA Semester 6 | Machine Learning Project")
st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔍 Recommendation Mode")

    search_mode = st.radio(
        "Choose Mode",
        ["Similarity", "Top Rated", "Popularity"]
    )

    selected_movie = ""
    if search_mode == "Similarity":
        movie_list = (
            df.sort_values(
                by="release_year",
                ascending=False
            )["title"]
            .unique()
        )
        selected_movie = st.selectbox(
            "Select Movie",
            movie_list
        )

    st.markdown("---")
    st.header("⚙️ Filters")

    all_genres = sorted(df["genre"].dropna().unique())
    selected_genre = st.selectbox(
        "Genre",
        ["All"] + list(all_genres)
    )

    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())

    year_range = st.slider(
        "Release Year Range",
        min_year,
        max_year,
        (min_year, max_year)
    )

    min_rating = st.slider(
        "Minimum Rating",
        0.0,
        10.0,
        0.0,
        0.1
    )

    min_votes = st.slider(
        "Minimum Votes",
        0,
        int(df["vote_count"].max()),
        0,
        100
    )

    num_results = st.slider(
        "Number of Movies",
        1,
        20,
        6
    )

    st.markdown("---")
    run_btn = st.button(
        "Get Recommendations",
        type="primary"
    )

# ---------------- RESULTS ----------------
if run_btn:
    results, error = get_recommendations(
        movie_name=selected_movie,
        search_by=search_mode,
        min_rating=min_rating,
        min_votes=min_votes,
        genre_filter=selected_genre,
        year_range=year_range,
        top_n=num_results
    )

    if error:
        st.error(error)

    elif results.empty:
        st.warning("No movies found.")

    else:
        st.success(f"Found {len(results)} movies")

        cols = st.columns(3)

        for idx, (_, row) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                with st.container(border=True):

                    if pd.notna(row.get("poster_url", None)):
                        st.image(
                            row["poster_url"],
                            use_container_width=True
                        )
                    else:
                        st.image(
                            "https://via.placeholder.com/300x450?text=No+Poster",
                            use_container_width=True
                        )

                    st.subheader(row["title"])
                    st.caption(
                        f"{row['release_year']} | ⭐ {row['rating']}"
                    )
                    st.write(f"**Genre:** {row['genre']}")

                    with st.expander("Overview"):
                        st.write(row["overview"])
