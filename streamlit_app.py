import streamlit as st
st.set_page_config(
    page_title="Att-ResRoBERTa",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "role" not in st.session_state:
    st.session_state.role = None

ROLES = [None,"Admin","User"]

def login():
    st.image("./images/runmodelcover.jpg")
    st.subheader("Att-ResRoberta: Code-Switched Tagalog-English Text-Image Multimodal Sarcasm Detection Using Attention Mechanism Intermodal Incongruity")
    st.header("Log In")
    role = st.selectbox("Choose your role", ROLES)
    if st.button("Log in"):
        st.session_state.role = role
        st.rerun()

def logout():
    st.session_state.role = None
    st.rerun()

role = st.session_state.role

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
settings = st.Page("settings.py", title="Settings", icon=":material/settings:")

#pages template module
#--------------User-------------#
home = st.Page(
    "section/home.py",
    title="Home",
    icon=":material/home:",  # Changed to "home" for the home page
    default=(role == "User"),
)

dataset_profile = st.Page(
    "section/dataset_profile.py",
    title="Dataset Profile", 
    icon=":material/storage:",  # Changed to "storage" for dataset profile
)

about_model = st.Page(
    "section/about_model.py",
    title="About the Model", 
    icon=":material/insights:",  # Changed to "insights" to represent the model
)

authors = st.Page(
    "section/authors.py",
    title="About the Authors",
    icon=":material/people:",  # Changed to "people" to represent authors
)

run_model = st.Page(
    "section/run_model.py",
    title ="Run Att-ResRoBERTa",
    icon=":material/play_arrow:",  # Changed to "play_arrow" for running the model
)

#--------------Admin-------------#
admin_train = st.Page(
    "admin/train.py",
    title="Train the Model",
    icon=":material/build:",  # Changed to "build" to represent training (building) the model
    default=(role == "Admin"),
)


account_pages = [logout_page, settings]
user_pages = [home, dataset_profile, about_model, authors, run_model]
admin_pages = [admin_train]


st.logo("images/h1.png",size="large")

page_dict = {}

if st.session_state.role in ["User", "Admin"]:
    page_dict["User"] = user_pages
if st.session_state.role == "Admin":
    page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation({"Account": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(login)])

pg.run()