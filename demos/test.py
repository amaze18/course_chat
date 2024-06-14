import streamlit as st

def main():
    st.title("Automatic File Upload Example")

    # Automatically read a file from the current working directory
    file_path = 'script-yt.txt'  # Replace with your file name
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        st.text_area("File Content", file_content, height=400)
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
