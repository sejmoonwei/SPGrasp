import os


def generate_tree(root_dir, ignore_dirs=None, indent=""):
    if ignore_dirs is None:
        ignore_dirs = {"tensorboard", "output",'notebooks','demo'}

    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        is_last = i == len(entries) - 1
        prefix = "└── " if is_last else "├── "

        if os.path.isdir(path):
            if entry in ignore_dirs:
                continue
            print(indent + prefix + entry)
            with open("project_structure.txt", "a", encoding="utf-8") as f:
                f.write(indent + prefix + entry + "\n")
            generate_tree(path, ignore_dirs, indent + ("    " if is_last else "│   "))
        else:
            print(indent + prefix + entry)
            with open("project_structure.txt", "a", encoding="utf-8") as f:
                f.write(indent + prefix + entry + "\n")


if __name__ == "__main__":
    project_root = os.getcwd()  # 获取当前目录作为项目根目录
    with open("project_structure.txt", "w", encoding="utf-8") as f:
        f.write(f"Project structure of: {project_root}\n\n")
    print(f"Project structure of: {project_root}\n")
    generate_tree(project_root)
