import subprocess
from settings import g_blender_excutable_path


def main():
    subprocess.run([g_blender_excutable_path, '--background', '--python', 'render.py'])


if __name__ == '__main__':
    main()
