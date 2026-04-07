import arcade
import arcade.gui
from Menu import LevelChoice

"""
View to later save the score, input first name and last name
"""
class InfoInput(arcade.View):
    def __init__(self):
        super().__init__()

        """
        Ui manager creation, stocks widgets for interactive purposes
        """
        self.gui_manager = arcade.gui.UIManager()
        self.gui_manager.enable()

        """
        Creation of a vertical box, widget to store widgets on top of each others
        """
        self.v_box = arcade.gui.UIBoxLayout(space_between=10)

        """
        Label and input for the first name
        """
        self.label_first_name = arcade.gui.UILabel(text="First Name")
        self.input_first_name = arcade.gui.UIInputText(text="Fill here", text_color=arcade.color.RED)

        """
        Label and input for the last name
        """
        self.label_last_name = arcade.gui.UILabel(text="Last Name")
        self.input_last_name = arcade.gui.UIInputText(text="Fill here", text_color=arcade.color.RED)

        """
        Button to send the info and display another view
        """
        self.validation_button = arcade.gui.UIFlatButton(text="Submit")
        self.validation_button.on_click = self.on_click_validation

        """
        Add every widgets to the vertical box
        """
        self.v_box.add(self.label_first_name)
        self.v_box.add(self.input_first_name)
        self.v_box.add(self.label_last_name)
        self.v_box.add(self.input_last_name)
        self.v_box.add(self.validation_button)

        """
        We add the v_box to an anchor widget to display on screen
        """
        self.gui_manager.add(
            arcade.gui.UIAnchorWidget(
                child=self.v_box
            )
        )

    def on_draw(self):
        self.clear()
        self.gui_manager.draw()

    def on_click_validation(self, event):
        """Clear, submit the info and open the next view, the level choice

        :param event: still not needed
        :return:
        """
        first_name = self.input_first_name.text
        last_name = self.input_last_name.text
        self.gui_manager.clear()
        self.clear()
        self.window.show_view(LevelChoice.LevelChoice(first_name, last_name))
