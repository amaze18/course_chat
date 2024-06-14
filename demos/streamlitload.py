import streamlit as st
import time

def main():
    st.title("Vidyarang Loading Screen")

    # Displaying content while loading
    with st.spinner("Loading..."):
        # Simulating a 10-second delay
        time.sleep(10)

        # Once loading is done, display the actual content
        st.success("Loading complete!")
        st.write("Welcome to Vidyarang")

        # Add more content here if needed

if __name__ == "__main__":
    main()
