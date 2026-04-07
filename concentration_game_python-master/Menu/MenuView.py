
import arcade
import arcade.gui
from Menu import InfoInput
# Constants
WIDTH = 1000
HEIGHT = 650

"""
First View
"""
class MenuView(arcade.View):
    def __init__(self):
        super().__init__()

        """
        Ui manager creation, stocks widgets for interactive purposes
        """
        self.ui_manager = arcade.gui.UIManager()
        self.ui_manager.enable()

        """
        Creation of the start button
        """
        start_button = arcade.gui.UIFlatButton(text="Start Game", width=200)
        start_button.on_click = self.on_button_click #assign function to run when the button is clicked
        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                child=start_button
            )
        )

    def on_draw(self):
        self.ui_manager.draw() #draw every widget in the ui manager
        arcade.set_background_color(arcade.color.BLACK) #black background
        arcade.draw_text("Focus game", WIDTH / 2.4, HEIGHT / 1.2, arcade.color.RED, 24)

    def on_button_click(self, event):
        """Clears the windows and calls the information input view ( event paramater needed even if not used )

        :param event: not used, used in case of specific click
        :return:
        """
        self.ui_manager.clear()
        self.window.show_view(InfoInput.InfoInput())


if __name__ == '__main__':
    pass
