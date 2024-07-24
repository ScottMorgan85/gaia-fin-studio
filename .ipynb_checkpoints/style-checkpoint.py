import streamlit as st
from PIL import Image


ms = st.session_state
if "themes" not in ms: 
    ms.themes = {
        "current_theme": "light",
        "refreshed": True,
        
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "black",
            "theme.primaryColor": "#c98bdb",
            "theme.secondaryBackgroundColor": "#5591f5",
            "theme.textColor": "white",
            "button_face": "🌜"
        },

        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "white",
            "theme.primaryColor": "#5591f5",
            "theme.secondaryBackgroundColor": "#82E1D7",
            "theme.textColor": "#0a1464",
            "button_face": "🌞"
        }
    }

def change_theme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"): 
            st._config.set_option(vkey, vval)

    ms.themes["refreshed"] = False
    if previous_theme == "dark": 
        ms.themes["current_theme"] = "light"
    elif previous_theme == "light": 
        ms.themes["current_theme"] = "dark"

def render_theme_toggle_button():
    btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
    if st.button(btn_face, on_click=change_theme, key="unique_theme_toggle_button"):
        if ms.themes["refreshed"] == False:
            ms.themes["refreshed"] = True
            st.experimental_rerun()



# import streamlit as st

# def apply_styles(theme="light"):
#     if theme == "dark":
#         with open('assets/styles.css') as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#     else:
#         st.markdown("""
#             <style>
#             body {
#                 background-color: #FFFFFF;
#                 color: #000000;
#             }

#             h1, h2, h3, h4, h5, h6 {
#                 color: #000000;
#             }

#             .sidebar .sidebar-content {
#                 background-color: #F0F2F6;
#             }

#             .stButton>button {
#                 background-color: #4CAF50;
#                 color: white;
#                 border: none;
#                 padding: 10px 24px;
#                 text-align: center;
#                 display: inline-block;
#                 font-size: 16px;
#                 margin: 4px 2px;
#                 transition-duration: 0.4s;
#                 cursor: pointer;
#             }

#             .stButton>button:hover {
#                 background-color: white;
#                 color: black;
#                 border: 2px solid #4CAF50;
#             }

#             .main .block-container {
#                 padding-top: 0.5rem;
#             }
#             </style>
#         """, unsafe_allow_html=True)

# def toggle_theme():
#     if "theme" not in st.session_state:
#         st.session_state.theme = "light"
#     st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
#     apply_styles(st.session_state.theme)

# def get_current_theme():
#     return st.session_state.get("theme", "light")
