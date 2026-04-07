# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import arcade
from Menu import MenuView
import ctypes
# Constants
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

SCREEN_TITLE = "Game"
SCREEN_WIDTH = screensize[0]
SCREEN_HEIGHT = screensize[1]


def main():
    """Main function"""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.set_fullscreen()
    menu_view = MenuView.MenuView()
    window.show_view(menu_view)
    arcade.run()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
