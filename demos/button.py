import streamlit as st

def main():
    # Set the title of the app
    st.title("Logo Buttons Example")

    # Paths or URLs to the logo images
    logo1_path = "/home/chetan/course_chat/demos/Untitled design/1.png"  # Placeholder URL for logo1
    logo2_path = "/home/chetan/course_chat/demos/Untitled design/2.png"  # Placeholder URL for logo2

    # Display the logos as buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(" ", key="logo1"):
            st.session_state.logo_clicked = "1"
        st.image(logo1_path, caption="Logo 1")

    with col2:
        if st.button(" ", key="logo2"):
            st.session_state.logo_clicked = "2"
        st.image(logo2_path, caption="Logo 2")

    # Display text based on the button clicked
    if "logo_clicked" in st.session_state:
        if st.session_state.logo_clicked == "1":
            st.write("You clicked the first logo!")
        elif st.session_state.logo_clicked == "2":
            st.write("You clicked the second logo!")
if __name__ == "__main__":
    main()