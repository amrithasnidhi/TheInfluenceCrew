with open('streamlit_dashboard.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('streamlit_dashboard.py', 'w', encoding='utf-8') as f:
    for i, line in enumerate(lines):
        if i == 270:
            f.write("if __name__ == '__main__':\n")
        if i >= 270:
            f.write("    " + line if line.strip() else "\n")
        else:
            f.write(line)
