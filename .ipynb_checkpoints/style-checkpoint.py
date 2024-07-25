import streamlit as st

def initialize_theme():
    if "themes" not in st.session_state:
        st.session_state.themes = {
            "current_theme": "light",
            "refreshed": True,
            "light": {
                "theme.base": "dark",
                "theme.backgroundColor": "black",
                "theme.primaryColor": "#c98bdb",
                "theme.secondaryBackgroundColor": "#5591f5",
                "theme.textColor": "white",
                "button_face": "🌐"
            },
            "dark": {
                "theme.base": "light",
                "theme.backgroundColor": "white",
                "theme.primaryColor": "#5591f5",
                "theme.secondaryBackgroundColor": "#82E1D7",
                "theme.textColor": "#0a1464",
                "button_face": "🌕"
            }
        }

def change_theme():
    previous_theme = st.session_state.themes["current_theme"]
    tdict = st.session_state.themes["light"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)

    st.session_state.themes["refreshed"] = False
    st.session_state.themes["current_theme"] = "dark" if previous_theme == "light" else "light"

def render_theme_toggle_button():
    btn_face = st.session_state.themes["light"]["button_face"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]["button_face"]
    tooltip_text = "Toggle Dark Mode" if st.session_state.themes["current_theme"] == "light" else "Toggle Light Mode"
    
    # Display the button with a tooltip
    st.markdown(f"""
        <div class="tooltip">
            <button onclick="window.location.reload();">{btn_face}</button>
            <span class="tooltiptext">{tooltip_text}</span>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.themes["refreshed"] == False:
        st.session_state.themes["refreshed"] = True
        st.experimental_rerun()

# Initialize the theme when this module is imported
initialize_theme()